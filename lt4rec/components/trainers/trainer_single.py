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
from components.pruning.prune import Pruning

class TrainerSingle(object):

    def __init__(self,
                 dataset,
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
                 init_masks=None,
                 task_id=1,
                 ):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._train_fn = train_fn
        self._train_epochs = train_epochs
        self._save_checkpoints_dir = os.path.join(save_checkpoints_dir,str(task_id))
        self._save_mask_dir=os.path.join(self._save_checkpoints_dir,"mask")
        #self._save_checkpoints_dir = save_checkpoints_dir
        self._restore_checkpoint_path = os.path.join(restore_checkpoint_path,str(task_id))
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

        self.task_id=task_id
        self._loss, self._train_op = self._build_train_graph()
        self._tensorboard_logdir=os.path.join(tensorboard_logdir,str(task_id))
        self._valid_logger = ValidateLogger("Train", self._validate_hour , self._tensorboard_logdir)
        self._test_logger = ValidateLogger("Valid", self._test_hour , self._tensorboard_logdir)
        self._train_logger = TrainLogger(self._log_steps, self._train_hour, self._tensorboard_logdir)
        self.cut_name=cut_name
        self.final_rate=final_rate
        self.pruning_iter=pruning_iter
        self.init_masks=init_masks
        self.masks={}
        if not os.path.exists(self._save_checkpoints_dir):
            os.makedirs(self._save_checkpoints_dir)
    
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
        pruner = Pruning(
        self._sess,self.masks,self._save_checkpoints_dir, self.prune_names, final_rate=self.final_rate, pruning_iter=self.pruning_iter)
        self._save_checkpoint(0,"ckpt_init")
        #self._restore_checkpoint()#warmup
        if self.init_masks is not None and os.path.exists(self.init_masks):
            pruner.load(self.init_masks)
            pruner.apply_mask(pruner.remain_mask)   #apply mask to net's variable
        feed_mask=self._feed_init_mask
        for prune_step in range(self.pruning_iter+1):#first time without pruning
            
            self._train_loop(feed_mask)
            
            # prune and save
            #print("save model,start prune time:{}".format(pruner.prune_times))
            #self._save_checkpoint(prune_step,
            #prefix="ckpt_{}".format( pruner.cur_rate))
            feed_mask=pruner.pruning_model()
            pruner.save(
            os.path.join(
                self._save_mask_dir, "{}_{}%.pkl".format(pruner.prune_times, pruner.cur_rate)))
            
        if sess is None:
            self._sess.close()

    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _combine_loss(self,ctr_loss,cvr_loss):
        return ctr_loss+cvr_loss

    def _build_train_graph(self):
        trainable_params = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self.optim_variables=trainer.variables()
        transform_fn = self._join_pipeline(self._transform_functions)
        loss_ctr,loss_cvr = self._train_fn(transform_fn(self._dataset.next_batch))
        #loss = self._combine_loss(loss_ctr,loss_cvr)
        loss = [loss_ctr,loss_cvr]
        grads = tf.gradients(loss[self.task_id], trainable_params)
        train_op = trainer.apply_gradients(list(zip(grads, trainable_params)))
        return loss[self.task_id], train_op

    def _train_loop(self,feed_mask):
        #self._restore_checkpoint()
        #self._validate_at_start=False
        if self._validate_at_start:
            print("testing...")
            self._test(epoch=0, step=0,feed_mask=feed_mask)

        step = 0
        for epoch in range(self._train_epochs):
            self._dataset.init(self._sess)
            while True:
                success = self._train_step(epoch, step,feed_mask)
                if not success:
                    break
                step += 1
                if self._validate_duaring_training and step % self._validate_steps == 0:
                    self._validate(epoch=epoch, step=step,feed_mask=feed_mask)
            self._train_logger._cleanup()
            self._validate(epoch=epoch, step=step,feed_mask=feed_mask)
            #self._save_checkpoint(epoch + 1)

    def _train_step(self, epoch, step,feed_mask):
        try:
            t_start = time.time()
            loss,_ = self._sess.run([self._loss,self._train_op],feed_dict=feed_mask)
            t_end = time.time()
            loss_dic={"loss_{}".format(self.task_id):loss}
            self._train_logger.log_info(loss_dic=loss_dic,
                                        time=t_end - t_start,
                                        size=self._dataset.batch_size,
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
        #print("model path",self._restore_checkpoint_path)
        #and  os.listdir(self._restore_checkpoint_path)
        if os.path.exists(self._restore_checkpoint_path):
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
