# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from components.networks.base_network import BaseNetwork


class MlpNetwork(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 loss,
                 hidden_sizes=[1024, 512, 512],
                 scope_name="mlp_network"):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._loss = loss
        self._hidden_sizes = hidden_sizes
        self._scope_name = scope_name

    def _train_fn(self, example):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            logits = self._build_graph(example)
            loss = self._loss.loss_fn(logits, example)
            return loss

    def _eval_fn(self, example):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            logits = self._build_graph(example)
            outputs = tf.sigmoid(logits)
            return outputs

    def _serve_fn(self, example):
        return self._eval_fn(example)

    def _build_graph(self, inputs):
        categorical_part = tf.concat(
            [tf.squeeze(inputs[name], axis=1) for name in self._categorical_features],
            axis=1,
        )
        numerical_part = tf.concat(
            [inputs[name] for name in self._numerical_features], axis=1
        )
        hidden = tf.concat([categorical_part, numerical_part], axis=1)
        for i, size in enumerate(self._hidden_sizes):
            hidden = slim.fully_connected(hidden, size, scope="fc_" + str(i))
        outputs = slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")
        return outputs
