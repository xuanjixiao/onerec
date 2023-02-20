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
def prune_by_percent_once(percent, mask, param):
    # Put the weights that aren't masked out in sorted order.
    sorted_weights = np.sort(np.abs(param[mask == 1]), axis=None)

    # Determine the cutoff for weights to be pruned.
    if sorted_weights.size <= 0:
        print("cutoff all of params, shape: {}".format(param.shape))
        print("last cutoff mask {}".format(np.sum(mask)))
        # print('cut all of params')
        return np.zeros(mask.shape)

    cutoff_index = np.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]
    print("cutoff index{}, cutoff weights {}".format(cutoff_index, cutoff))
    # Prune all weights below the cutoff.
    return np.where(np.abs(param) <= cutoff, np.zeros(mask.shape), mask)


def prune_by_percentile_once(percent, mask, param):

    cutoff = np.percentile(np.abs(param[mask]), percent * 100.0, axis=None)
    print("cutoff weights {}".format(cutoff))
    # Prune all weights below the cutoff.
    return np.where(np.abs(param) <= cutoff, np.zeros(mask.shape), mask)


class Pruning():
    def __init__(
        self,
        sess,
        mask_dic,
        save_checkpoints_dir,
        pruning_param_names,
        final_rate=0.1,
        pruning_iter=3,
        prune_once=None,
    ):
        self._sess=sess
        assert pruning_iter >= 0
        self.final_rate = final_rate
        self.pruning_iter = pruning_iter
        prune_once = prune_once or prune_by_percent_once
        self.pruning_names = set(pruning_param_names)
        self._log = Logger(__name__)
        self._log.info(self.pruning_names)
        self.prune_times = 0
        self.one_rate = (
            1 - (self.final_rate ** (1.0 / self.pruning_iter))
            if self.pruning_iter > 0
            else 1.0
        )
        self.prune_once = partial(prune_once, self.one_rate)
        self._log.info(
            "Pruning iter {}, pruning once persent {}, final remain rate {}".format(
                self.pruning_iter, self.one_rate, self.final_rate
            )
        )
        self._save_checkpoints_dir=save_checkpoints_dir
        # backup initial weights
        #self.backup_weights =self.get_backup_weights()
        self._log.info(
            "model params :{}".format(
                [v for v in tf.trainable_variables()]
            )
        )
        
        remain_mask = {
            p.name: np.ones(p.shape).astype("float32")
            for  p in tf.trainable_variables()
           if p.name in self.pruning_names
        }
        #remain_mask=mask_dic.copy()
        self.masks_dic=mask_dic
        self.remain_mask = remain_mask
        self.pruning_names = set(self.remain_mask.keys())

        self._log.info("Pruning params are in following ...")
        total_m = 0
        for name, p in self.remain_mask.items():
            self._log.info("Need pruning {}, params: {}".format(name, p.size))
            total_m += p.size
        self._log.info("Total need pruning params: {}".format(total_m))
        self._log.info("feed mask dic:{}".format(self.masks_dic))
        self.total_params = total_m
        self.cur_rate = 100.0
        self.last_cutoff = 0


    def on_batch_end(self):         
        if self.final_rate < 1.0 and self.prune_times > 0:
            self.apply_mask(self.remain_mask)

        
    def _restore_init_checkout(self):
        checkpoint_saver = tf.train.Saver(max_to_keep=None)
        checkpoint_path = os.path.join(self._save_checkpoints_dir, "ckpt_init-0")
        
        self._log.info("init_weight success form{}".format(checkpoint_path))
        checkpoint_saver.restore(self._sess, checkpoint_path)

    def map_mask_name_to_w(self,mask_name):
        return mask_name.split("_m")[0]+"_w:0"

    def apply_mask(self, remain_mask):
        """apply mask on net'variable
        """
        feed_dict={}
        for name,p in self.masks_dic.items():
            w_name=self.map_mask_name_to_w(name)
            if w_name in remain_mask:
                feed_dict[p]=remain_mask[w_name]
                self._log.info("feed mask of {}".format(w_name))
                #print(p.shape,remain_mask[w_name].shape)
        return feed_dict
        
    def update_cur_rate(self):
        total_m = sum(m.sum().item() for m in self.remain_mask.values())
        #total_m = self.total_params - total_m       #left parameter
        self.cur_rate = round(100.0 * total_m / self.total_params, 2)   #left rate
        self._log.info(
            "Init No #{} pruning, remain params {},total params {} remain percent {}%".format(
                self.prune_times, total_m,self.total_params, self.cur_rate
            )
        )

    def pruning_model(self):
        self.prune_times += 1
        if self.prune_times <= self.pruning_iter:
            # self.prune(self.one_rate, self.remain_mask, self._model)
            self.prune_global(self.one_rate, self.remain_mask)  #prune model and update reamin_mask
            total_m = sum(m.sum().item() for m in self.remain_mask.values())
            #total_m = self.total_params - total_m       #left parameter
            self.cur_rate = round(100.0 * total_m / self.total_params, 2)   #left rate
            self._log.info(
                "No #{} pruning, remain params {},total params {} remain percent {}%".format(
                    self.prune_times, total_m,self.total_params, self.cur_rate
                )
            )
            # reset optimizer & model weights when pruning
            self._restore_init_checkout()
        else:
            self._log.info(
                "No #{} pruning, exceed max pruning times".format(self.prune_times + 1)
            )
        return self.apply_mask(self.remain_mask)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            #"init_weights": self.backup_weights,
            "mask": self.remain_mask,
            "pruning_time": self.prune_times,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self._log.info("save pruning state {}".format(filepath))


    def load(self, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        #self.backup_weights = state['init_weights']
        self.remain_mask = state["mask"]
        self.prune_times = state["pruning_time"]
        
        self._log.info("load mask from {}".format(filepath))
        self._log.info("current pruning time {}".format(self.prune_times))
        self.update_cur_rate()


    def prune_global(self, rate, remain_mask):

        names = []
        masks = []
        weights = []        #masked params
        my_variable = [v for v in tf.trainable_variables()]
        params = self._sess.run(my_variable)
        remain_params=[]    #need cut params
        for k,p in zip(my_variable, params):
            if k.name in remain_mask:
                # mask = remain_mask[name] == 0
                names.append(k.name)
                mask = remain_mask[k.name]
                masks.append(mask)
                remain_params.append(p)
                weights.append(p[mask == 1])    

        all_w = np.concatenate([np.reshape(w, -1) for w in weights], axis=0)
        sorted_w = np.sort(np.abs(all_w))
        if sorted_w.size <= 0:
            self._log.info("cutoff all of params")
            return
        cutoff_index = np.round(rate * sorted_w.size).astype(int)
        cutoff = sorted_w[cutoff_index]     #cut off value

        new_masks = []
        for m, p in zip(masks, remain_params):
            new_m = np.where(np.abs(p) < cutoff, np.zeros(m.shape), m)#if less than cutoff,mask=0
            new_masks.append(new_m)
        #device = next(model.parameters()).device
        for name, new_m in zip(names, new_masks):   #update remain_mask
            remain_mask[name] = new_m.astype("float32")
        self._log.info(
            "No #{}, cutoff weights {}, weights mean {}, weights std {}".format(
                self.prune_times, cutoff, all_w.mean(), all_w.std()
            )
        )
        self.last_cutoff = cutoff
