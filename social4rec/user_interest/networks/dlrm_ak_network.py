# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from components.networks.base_network import BaseNetwork


class DlrmAKNetwork(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 multivalue_features,
                 attention_features,
                 loss,
                 ptype="mean",
                 hidden_sizes=[512, 512, 512],
                 scope_name="dlrm2_network"):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._multivalue_features = multivalue_features
        self._attention_features = attention_features
        self._loss = loss
        self._hidden_sizes = hidden_sizes
        self._scope_name = scope_name
        self._ptype = ptype

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
        hiddens = self._build_lower_part_graph(inputs)
        hiddens = self._build_cross_interaction(hiddens)
        outputs = self._build_upper_part_graph(hiddens)
        return outputs

    def _build_lower_part_graph(self, inputs):
        categorical_part = self._build_categorical_part(inputs)
        numerical_part = self._build_numerical_part(inputs)
        multivalue_part = self._build_multivalue_part(inputs)
        attention_part = self._build_attention_part(inputs)
        return tf.stack(categorical_part + multivalue_part + attention_part, axis=1)

    def _build_upper_part_graph(self, inputs):
        hidden = inputs
        for i, size in enumerate(self._hidden_sizes):
            hidden = slim.fully_connected(hidden, size, scope="fc_" + str(i))
        return slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")

    def _build_cross_interaction(self, inputs):
        cross = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))
        cross = tf.reshape(cross, [-1, cross.get_shape()[1] * cross.get_shape()[2]])
        flatten = tf.reshape(inputs,
                             [-1, inputs.get_shape()[1] * inputs.get_shape()[2]])
        outputs = tf.concat([cross, flatten], axis=1)
        return outputs

    def _build_categorical_part(self, inputs):
        return [tf.squeeze(inputs[name], axis=1) for name in self._categorical_features]

    def _build_numerical_part(self, inputs):
        outputs = []
        if len(self._numerical_features) != 0:
            h0 = tf.concat([inputs[name] for name in self._numerical_features], axis=1)
            h1 = slim.fully_connected(h0, 512, scope="numerical_fc1")
            h2 = slim.fully_connected(h1, 256, scope="numerical_fc2")
            outputs = slim.fully_connected(h2, 128, scope="numerical_fc3")
        return [outputs]

    def _build_multivalue_part(self, inputs):
        def pooling(vals, ptype):
            if ptype == "mean":
                return tf.reduce_mean(vals, axis=1)
            else:
                return tf.reduce_sum(vals, axis=1)

        outputs = [pooling(inputs[name], self._ptype)
            for name in self._multivalue_features]
        return outputs

    def _build_attention_part(self, inputs):
        # key: [batch, dim]
        # vals: [batch, vlen, dim]
        # out: [batch, dim], weighted sum of vals
        def attention(key, vals):
            #key = tf.expand_dims(key, axis=1)
            weight = tf.reduce_sum(tf.multiply(key, vals), axis=2)
            sum_weight = tf.reduce_sum(weight, axis=1, keepdims=True)
            norm_weight = weight / sum_weight
            return tf.reduce_sum(tf.multiply(tf.expand_dims(norm_weight, axis=2), vals),
                    axis=1)
        outputs = [attention(inputs[names[0]], inputs[names[1]])
            for names in self._attention_features]
        return outputs
