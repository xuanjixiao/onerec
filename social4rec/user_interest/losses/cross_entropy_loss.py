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


class ClipWeightedCrossEntropyLoss(WeightedCrossEntropyLoss):

    def __init__(self, label_name, weight_name, clip_min, clip_max):
        self._clip_min = clip_min
        self._clip_max = clip_max
        super().__init__(label_name, weight_name)

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        weights = tf.clip_by_value(
            tf.to_float(examples[self._weight_name]),
            self._clip_min,
            self._clip_max,
        )
        return self._weighted_cross_entropy_loss(logits, labels, weights)


class LogWeightedCrossEntropyLoss(WeightedCrossEntropyLoss):

    def __init__(self, label_name, weight_name, a, b):
        self._a = a
        self._b = b
        super().__init__(label_name, weight_name)

    def loss_fn(self, logits, examples):
        labels = tf.to_float(examples[self._label_name])
        weights = tf.log(
            tf.maximum(tf.to_float(examples[self._weight_name]), 0.0) / self._a + 1.0
        ) * self._b + 1.0
        return self._weighted_cross_entropy_loss(logits, labels, weights)
