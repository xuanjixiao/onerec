# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import tensorboard
from components.utils.loggers import TrainLogger, ValidateLogger
from components.pruning.prune_mlt import MTL_Masker
from components.utils.loggers import Logger
import random

class TrainerMTL(object):

    def __init__(self,
                 datasets,
                 transform_functions,
                 train_fn,
                 validate_steps,
                 log_steps,
                 learning_rate,
                 train_epochs=1,
                 evaluator=None,
                 tester=None,
                 save_checkpoints_dir=None,
                 restore_checkpoint_path=None,
                 validate_at_start=True,
                 tensorboard_logdir=None,
                 train_hour=None,
                 test_hour=None,
                 validate_hour=None,
                 validate_duaring_training=False,
                 cut_name=None,
                 final_rate=0.1,
                 pruning_iter=3,
                 task_ids=[0,1]
                 ):
        self._datasets = datasets
        self._transform_functions = transform_functions
        self._train_fn = train_fn
        self._train_epochs = train_epochs
        self._save_checkpoints_dir = save_checkpoints_dir
        self._save_mask_dir=os.path.join(self._save_checkpoints_dir,"mask")
        self._restore_checkpoint_path = os.path.join(restore_checkpoint_path)
        self._validate_steps = validate_steps
        self._log_steps = log_steps
        self._learning_rate = learning_rate
        self._validate_at_start = validate_at_start
        self._evaluator = evaluator
        self._tester = tester
        self._train_hour = train_hour
        self._validate_hour = validate_hour
        self._test_hour = test_hour
        self._validate_duaring_training = validate_duaring_training
        self.task_lst=task_ids
        self._loss_lst,self._train_op_lst = self._build_train_graph()
        #self._train_op=self._apply_gradients_for_loss(0)
        self._tensorboard_logdirs=tensorboard_logdir
        self._valid_logger = ValidateLogger("Train", self._validate_hour , self._tensorboard_logdirs)
        self._test_logger = ValidateLogger("Valid", self._test_hour , self._tensorboard_logdirs)
        self._train_logger = TrainLogger(self._log_steps, self._train_hour, self._tensorboard_logdirs)
        self.cut_name=cut_name
        self.masks={}
        self._log=Logger(__name__)
        self._task_ids_lst=[]
        
        
    def get_cut_name(self):
        """must after session initialized"""
        if self.cut_name==None:
            return
        need_cut_names = list(set([s.strip() for s in self.cut_name.split(",")]))
        prune_names = []
        for v in tf.trainable_variables():
            if "biases" in v.name:
                continue
            for n in need_cut_names:
                if n in v.name:
                    prune_names.append(v.name)
                    break
        return prune_names  

    def get_mask_init_dic(self,prune_names):
        
        feed_init={}
        graph=tf.get_default_graph()
        for name in prune_names:
            mask_name=name.split("_w")[0]+"_m:0"
            mask_tensor=graph.get_tensor_by_name(mask_name)
            feed_init[mask_tensor]=np.ones(mask_tensor.shape,dtype="float32")
            self.masks[mask_name]=mask_tensor 
            #print("mask",mask_tensor)
        return feed_init   

    def run(self, sess=None):
        if sess is None:
            self._sess = self._create_and_init_session()
        else:
            self._sess = sess
        self.prune_names=self.get_cut_name()        #dlrm2_network/fc_2_w
        self._feed_init_mask=self.get_mask_init_dic(self.prune_names)#dlrm2_network/fc_2_m
        self._log.info("need prune weight:{}".format(self.prune_names))
        self._mlt_masker=MTL_Masker(self._sess,self.masks,self._save_mask_dir,self.prune_names,self._save_checkpoints_dir)
          
        self._train_loop()
            
        if sess is None:
            self._sess.close()

    def _create_and_init_session(self):
        #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _combine_loss(self,ctr_loss,cvr_loss):
        ctr_loss=ctr_loss
        cvr_loss=1.5*cvr_loss
        loss_lst=[ctr_loss,cvr_loss]
        return loss_lst

    def _build_train_graph(self):
        trainable_params = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        
        transform_fn = self._join_pipeline(self._transform_functions)
        loss_ctr,loss_cvr = self._train_fn(transform_fn(self._datasets.next_batch))
        #f_loss_ctr,loss_cvr = self._train_fn(transform_fn(self._datasets[1].next_batch))
        loss_lst = self._combine_loss(loss_ctr,loss_cvr)
        
        grads_ctr = tf.gradients(loss_lst[0], trainable_params)   #just for init optimizer's variable
        grads_cvr = tf.gradients(loss_lst[1],trainable_params)
        train_op_ctr = trainer.apply_gradients(list(zip(grads_ctr, trainable_params)))
        train_op_cvr = trainer.apply_gradients(list(zip(grads_cvr, trainable_params)))
        train_op_lst=[train_op_ctr,train_op_cvr]
        return loss_lst,train_op_lst



    def _train_loop(self):
        self._restore_checkpoint()
        #self._validate_at_start=False

        if self._validate_at_start:
            print("testing...")
            self._test(epoch=0, step=0,feed_mask=self._feed_init_mask)
        
        step = 0
        task_len=len(self.task_lst)
        
        for epoch in range(self._train_epochs):
            
            self._datasets.init(self._sess)
            empty_task=set()
            while len(empty_task)<task_len:
                for task_id in self.task_lst:
                    if task_id in empty_task:
                        continue
                
                    cur_task_id=task_id
                    feed_mask=self._mlt_masker.before_forward(cur_task_id)
                    #feed_mask=self._feed_init_mask #warmup
                    success = self._train_step(epoch, step,cur_task_id,feed_mask)
                     
                    if not success:
                        empty_task.add(cur_task_id)
                    else:
                        step += 1
                    if self._validate_duaring_training and step % self._validate_steps == 0:
                        for task_id in range(task_len):
                            self._log.info("validate for task:{}".format(task_id))
                            feed_mask=self._mlt_masker.before_forward(task_id)
                            self._validate(epoch=epoch, step=step+task_id,feed_mask=feed_mask)
                    #if step==2000:
                    #    self._save_checkpoint(0,"ckpt_init")

            self._train_logger._cleanup()
            for task_id in range(task_len):
                self._log.info("validate for task:{}".format(task_id))
                feed_mask=self._mlt_masker.before_forward(task_id)
                self._validate(epoch=epoch, step=step,feed_mask=feed_mask)
            self._log.info("validate without mask")
            self._validate(epoch=epoch, step=step,feed_mask=self._feed_init_mask)
            self._save_checkpoint(epoch + 1)

    def _train_step(self, epoch, step,task_id,feed_mask):
        try:
            t_start = time.time()
            
            loss,_ = self._sess.run([self._loss_lst[task_id],self._train_op_lst[task_id]],
            feed_dict=feed_mask)
            #self._train_op=self._apply_gradients(task_id)
            #_ = self._sess.run(self._train_op)
            #tf.get_default_graph().finalize()
            t_end = time.time()
            #print("cur task_id:{},cost time:{}".format(task_id,t_end-t_start))
            loss_dic={"loss_{}".format(task_id):loss}
            self._train_logger.log_info(loss_dic=loss_dic,
                                        time=t_end - t_start,
                                        size=self._datasets.batch_size,
                                        epoch=epoch,
                                        step=step + 1
                                        )        
             
            return True
        except tf.errors.OutOfRangeError:
            return False

    def _save_checkpoint(self, step, prefix="ckpt_epoch"):
        if self._save_checkpoints_dir:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_path = os.path.join(self._save_checkpoints_dir, prefix)
            checkpoint_saver.save(self._sess, checkpoint_path, global_step=step)

    def _restore_checkpoint(self):
        #print("restore model path",self._restore_checkpoint_path)
        #and  os.listdir(self._restore_checkpoint_path)
        ckpt_path=os.path.join(self._restore_checkpoint_path,"checkpoint")
        if os.path.exists(ckpt_path):
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            ckpt = tf.train.latest_checkpoint(self._restore_checkpoint_path)
            checkpoint_saver.restore(self._sess, ckpt)

    def _validate(self, epoch, step,feed_mask):
        if self._evaluator is not None:
            eval_results,cvr_predicts,cvr_labels = self._evaluator.run(sess=self._sess,feed_mask=feed_mask)
            for i in range(10):
                print("predict,label",cvr_predicts[i],cvr_labels['cvrlabel'][i])
            self._valid_logger.log_info(eval_results, epoch=epoch, step=step)

    def _test(self, epoch, step,feed_mask):
        if self._tester is not None:
            test_results,cvr_predicts,cvr_labels = self._tester.run(sess=self._sess,feed_mask=feed_mask)
            for i in range(10):
                print("predict,label",cvr_predicts[i],cvr_labels['cvrlabel'][i])
                
            #self._test_logger._log_to_console(labels, epoch=epoch, step=step)
            self._test_logger.log_info(test_results, epoch=epoch, step=step)

    def _join_pipeline(self, map_functions):

        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
