# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.losses.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):

    def __init__(self, label_name):
        self._label_name = label_name

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        return self._cross_entropy_loss(logits, labels)

    def _cross_entropy_loss(self, logits, labels):
        sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        avg_loss = tf.reduce_mean(sample_loss)
        return avg_loss


class WeightedCrossEntropyLoss(BaseLoss):

    def __init__(self, label_name, weight_name):
        self._label_name = label_name
        self._weight_name = weight_name

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        weights = tf.to_float(examples[self._weight_name])
        return self._weighted_cross_entropy_loss(logits, labels, weights)

    def _weighted_cross_entropy_loss(self, logits, labels, weights):
        sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        avg_loss = tf.reduce_mean(tf.multiply(sample_loss, weights))
        return avg_loss


class FocalLoss(BaseLoss):

    def __init__(self, label_name,alpha=0.25, gamma=2):
        self._label_name = label_name
        self._alpha=alpha
        self._gamma=gamma

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        return self._focal_loss(logits, labels,self._alpha,self._gamma)

    def _focal_loss(self, logits, labels,alpha=0.25, gamma=2):
        predicts=tf.sigmoid(logits)
        zeros = tf.zeros_like(predicts, dtype=predicts.dtype)
        #is_negative_sample=tf.where(((duration>60) & (playrates<0.1)&(playrates>0))|((duration<=60)&(playrates<0.2)&(playrates>0)),
        #tf.ones_like(labels),tf.zeros_like(labels))
        #weights=tf.where(is_negative_sample>0,tf.ones_like(labels)*0.2,tf.ones_like(labels))
        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(labels > zeros, labels - predicts, zeros) # positive sample 

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(labels > zeros, zeros, predicts) # negative sample 
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(predicts, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - predicts, 1e-8, 1.0))
        
        return tf.reduce_mean(per_entry_cross_ent)
 
class MtlCrossEntropyLoss(BaseLoss):

    def __init__(self, label_name):
        self._label_name = label_name

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        playrates=tf.to_float(examples["playrate"])
        return self._cross_entropy_loss(logits, labels,playrates)

    def _cross_entropy_loss(self, logits, labels,playrates):
        weights=tf.where((playrates>0) & (playrates<0.3),tf.ones_like(labels)*0.2,tf.ones_like(labels))
        sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        avg_loss = tf.reduce_mean(sample_loss*weights)
        return avg_loss