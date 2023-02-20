# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.losses.base_loss import BaseLoss


class MSELoss(BaseLoss):

    def __init__(self, label_name):
        self._label_name = label_name

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        logits=tf.sigmoid(logits)
        return self._mse_loss(logits, labels)

    def _mse_loss(self, logits, labels):
        avg_loss=tf.losses.mean_squared_error(labels=labels,predictions=logits)
        return avg_loss

class WeightedMSELoss(BaseLoss):

    def __init__(self, label_name):
        self._label_name = label_name

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        logits=tf.sigmoid(logits)
        #duration=tf.string_to_number(examples["duration"])
        return self._mse_loss(logits, labels)

    def _mse_loss(self, logits, labels):
        labels=tf.clip_by_value(labels,0,1)#cut labels>1
        weights=tf.where(labels>0,tf.ones_like(labels),tf.zeros_like(labels))#only calculate positive sample
        #is_negative_sample=tf.where(((duration>60) & (labels<0.1)&(labels>0))|((duration<=60)&(labels<0.2)&(labels>0)),
        #tf.ones_like(labels),tf.zeros_like(labels))
        #weights=tf.where(is_negative_sample>0,5*weights,weights)
        avg_loss=tf.losses.mean_squared_error(labels=labels,predictions=logits, weights=weights)
        return avg_loss
