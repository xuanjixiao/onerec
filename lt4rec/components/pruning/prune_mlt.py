# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from functools import partial
from copy import deepcopy
from components.utils.loggers import Logger
import tensorflow as tf
import pickle
import os


class MTL_Masker:
    def __init__(self,sess, mask_dic,masks_dir,pruning_param_names,save_checkpoints_dir):
        self.masks = masks_dir  
        self.weights = []
        self._sess=sess
        self._log = Logger(__name__)
        self.pruning_names = set(pruning_param_names)
        self.backup_values=[]
        self._save_checkpoints_dir=save_checkpoints_dir
        self.masks_dic=mask_dic
        self.masks_init()
        

    def masks_init(self):
        if self.masks is None:
            mask = {
            p.name: np.ones(p.shape).astype("float32")
            for  p in tf.trainable_variables()
            if p.name in self.pruning_names}
            self.masks = mask
        else:
            self.masks=self.load_masks(self.masks)
        self._log.info("has masks {},tpye {}".format(len(self.masks), type(self.masks)))

        self._feed_masks=[]

        for i,mask in enumerate(self.masks):
            self.get_cur_rate(mask,i)
            feed_mask=self.apply_mask(i)
            self._feed_masks.append(feed_mask)
            
        


    def get_cur_rate(self,mask,task_id):
        cur_m = sum(m.sum().item() for m in mask.values())
        total_m=0
        for name, p in mask.items():
            #self._log.info("Need pruning {}, params: {}".format(name, p.size))
            total_m += p.size
        
        cur_rate = round(100.0 * cur_m / total_m, 2)   #left rate
        self._log.info(
            "Task_id #{} , remain params {},total params {} remain percent {}%".format(
                task_id, cur_m,total_m, cur_rate))


    def load_masks(self,masks_dir):
        """mlt/mask/task_id_mask.pkl
        """
        masks_path = [
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if not f.startswith("init")
        ]
        masks_path = list(
            sorted(
                filter(lambda f: os.path.isfile(f), masks_path),
                key=lambda s: int(os.path.basename(s).split(".pkl")[0]),
            )
        )
        masks = []
        self._log.info("loading masks")

        for path in masks_path:
            with open(path, "rb") as f:
                dump = pickle.load(f)
            assert "mask" in dump and "pruning_time" in dump
            self._log.info(
                "loading pruning_time {}".format(dump["pruning_time"])
            )
            masks.append(dump["mask"])
        # sanity check
        assert len(masks) == len(masks_path)
        return masks



    def _restore_init_checkout(self,params):
        checkpoint_saver = tf.train.Saver(var_list=params,max_to_keep=None)
        #checkpoint_saver = tf.train.Saver(max_to_keep=None)
        checkpoint_path = os.path.join(self._save_checkpoints_dir, "ckpt_init-0")
        self._log.info("init_weight success form{}".format(checkpoint_path))
        checkpoint_saver.restore(self._sess, checkpoint_path)
  
    
    def _save_checkpoint(self,step,params, prefix="ckpt_epoch"):
        if self._save_checkpoints_dir:
            checkpoint_saver = tf.train.Saver(var_list=params,max_to_keep=None)
            #checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_path = os.path.join(self._save_checkpoints_dir, prefix)
            checkpoint_saver.save(self._sess, checkpoint_path, global_step=step)

    def before_forward(self, task_id):
        #self._log.info("before forward,save unmasked params and apply mask")
        # apply mask to param
        return self._feed_masks[task_id]
        #return self.apply_mask(task_id)


    def map_mask_name_to_w(self,mask_name):
        return mask_name.split("_m")[0]+"_w:0"

    def apply_mask(self, task_id):
        if isinstance(self.masks, dict):
            remain_mask = self.masks
        else:
            remain_mask = self.masks[task_id]
        
        feed_dict={}
        for name,p in self.masks_dic.items():
            w_name=self.map_mask_name_to_w(name)
            if w_name in remain_mask:
                feed_dict[p]=remain_mask[w_name]
                #self._log.info("feed mask of {}".format(w_name))
        return feed_dict

        
