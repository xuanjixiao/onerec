# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from components.networks.base_network import BaseNetwork


class Dlrm2Network(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 multivalue_features,
                 loss,
                 hidden_sizes=[512, 512, 512],
                 scope_name="dlrm2_network"):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._multivalue_features = multivalue_features
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
        hiddens = self._build_lower_part_graph(inputs)
        hiddens = self._build_cross_interaction(hiddens)
        outputs = self._build_upper_part_graph(hiddens)
        return outputs

    def _build_lower_part_graph(self, inputs):
        categorical_part = self._build_categorical_part(inputs)
        numerical_part = self._build_numerical_part(inputs)
        multivalue_part = self._build_multivalue_part(inputs)
        return tf.stack(categorical_part + numerical_part + multivalue_part, axis=1)

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
        outputs = [
            self._engagement_attention(inputs[names[0]],
                                       [inputs[name] for name in names[1:]])
            for names in self._multivalue_features
        ]
        return outputs

    def _engagement_attention(self, inputs, engagements):
        attention_weights = self._attention_weights_from_engagements(engagements)
        reduced = self._reduce_with_attention_weights(inputs, attention_weights)
        return reduced

    def _attention_weights_from_engagements(self, engagements):
        expanded_engagements = []
        for i, engagement in enumerate(engagements):
            engagement = tf.cast(tf.maximum(engagement, 0), tf.float32)
            if i == 0:
                binarized = tf.cast(engagement > 0, tf.float32)
                sum_of_binarized = tf.reduce_sum(binarized, axis=1, keepdims=True)
                normalized_binarized = binarized / (sum_of_binarized + 1e-20)
                expanded_engagements.extend([binarized, normalized_binarized])
            sum_of_engagement = tf.reduce_sum(engagement, axis=1, keepdims=True)
            normalized = engagement / (sum_of_engagement + 1e-20)
            logged = tf.log(engagement + 1.0)
            sum_of_logged = tf.reduce_sum(logged, axis=1, keepdims=True)
            normalized_logged = logged / (sum_of_logged + 1e-20)
            expanded_engagements.extend([normalized, normalized_logged])

        expanded_engagements = tf.stack(expanded_engagements, axis=2)
        h1 = slim.fully_connected(expanded_engagements, 128, scope="att_fc1")
        h2 = slim.fully_connected(h1, 64, scope="att_fc2")
        h3 = slim.fully_connected(h2, 1, activation_fn=None, scope="att_fc3")
        attention_weights = tf.multiply(h3, tf.expand_dims(binarized, axis=2))
        return attention_weights

    def _reduce_with_attention_weights(self, inputs, attention_weights):
        reduced = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        return reduced
