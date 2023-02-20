# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
from components.networks.base_network import BaseNetwork
import numpy as np
SOM_WIDTH = 300
SOM_HEIGHT = 300
NUM_INPUT = 32
SOM_LEN = SOM_WIDTH * SOM_HEIGHT


class ClusterNetwork(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 multivalue_features,
                 ptype,
                 loss,
                 hidden_sizes=[1024, 512, 512],
                 scope_name="mlp_network"):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._multivalue_features = multivalue_features
        self._loss = loss
        self._hidden_sizes = hidden_sizes
        self._scope_name = scope_name
        self._ptype = ptype
        self._feature_center = None
        self._feature = None
        self._result = 1
        self._xishu = 1

        init_ones = tf.ones_initializer()
        self.time = tf.get_variable("time", shape=[1],
                                    initializer=init_ones)
        self.weights = tf.get_variable("weights", shape=[SOM_LEN, NUM_INPUT],
                                       initializer=tf.contrib.layers.xavier_initializer())

        # self.radius = tf.placeholder(tf.float64, [1], name='radius')
        self.radius = tf.constant(
            50.0, dtype=tf.float64, shape=[1], name='radius')

        # learning rate
        self.alpha = tf.constant(
            0.001,  shape=[1], dtype=tf.float64, name='alpha')
        self.nodes = tf.reshape(tf.transpose(tf.constant(np.indices(
            (SOM_HEIGHT, SOM_WIDTH))), perm=[1, 2, 0]), [SOM_LEN, 2])

    def to2d(self, x):

        return [x // SOM_WIDTH, x % SOM_WIDTH]

    def to1d(self, h, w):
        return h * SOM_WIDTH + w

    def _train_fn(self, example, inputs):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            omgid = inputs['FEA_OMGID']
            omgid = tf.squeeze(omgid, 1)
            fea_center, feature, see, we = self._build_graph(example(inputs))
            loss = tf.losses.mean_squared_error(fea_center, feature)
            return self.weights, feature, self._xishu, see, we, loss

    def _build_multivalue_part(self, inputs):
        def pooling(vals, ptype):
            if ptype == "mean":
                return tf.reduce_mean(vals, axis=1)
            else:
                return tf.reduce_sum(vals, axis=1)

        outputs = tf.concat([pooling(inputs[name], self._ptype)
                             for name in self._multivalue_features], axis=1,)
        return outputs

    def _eval_fn(self, example):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            logits = self._build_graph(example)
            outputs = tf.sigmoid(logits)
            return outputs

    def _serve_fn(self, example):
        return self._eval_fn(example)

    # def _cosine(self, q,a):
    #     pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    #     pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    #     pooled_mul_12 = tf.reduce_sum(q * a, 1)
    #     score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    #     return score

    def _cosine(self, _matrixA, _matrixB):

        _matrixA_matrixB = tf.matmul(_matrixA, tf.transpose(_matrixB))
        # 按行求和，生成一个列向量
        # 即各行向量的模
        _matrixA_norm = tf.sqrt(tf.reduce_sum(
            tf.multiply(_matrixA, _matrixA), 1))
        _matrixB_norm = tf.sqrt(tf.reduce_sum(
            tf.multiply(_matrixB, _matrixB), 1))
        _matrixA_norm = tf.expand_dims(_matrixA_norm, axis=1)
        _matrixB_norm = tf.expand_dims(_matrixB_norm, axis=1)
        return tf.div(_matrixA_matrixB, tf.matmul(_matrixA_norm, tf.transpose(_matrixB_norm))+1e-8)

    # def _normalize_adj(self, adj):
    #     """Symmetrically normalize adjacency matrix."""
    #     adj = sp.coo_matrix(adj)
    #     rowsum = np.array(adj.sum(1))
    #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _find_neibor(self, hidden):
        inter = tf.matmul(hidden, tf.transpose(hidden))
        test_sum = tf.reduce_sum(tf.square(hidden), axis=1)  # num_test x 1
        train_sum = tf.reduce_sum(tf.square(hidden), axis=1)  # num_train x 1
        test_sum = tf.expand_dims(test_sum, 1)
        train_sum = tf.expand_dims(train_sum, 1)

        ones = tf.ones((tf.shape(train_sum)[0]))
        dia_ones = tf.matrix_diag(ones)
        dists = tf.sqrt(-2 * inter + tf.transpose(test_sum) +
                        train_sum + dia_ones)

        sim_thresold = tf.cast(((inter+dia_ones) > 0.01), tf.float32)
        top_k = tf.nn.top_k(-dists, k=200, sorted=True, name=None)
        kth = tf.reduce_min(top_k.values, 1, keepdims=True)  # 找出最小值
        top2 = tf.cast(tf.greater_equal(-dists, kth), tf.float32) + dia_ones
        top2 = tf.cast((tf.multiply(top2, sim_thresold) > 0), tf.float32)
        div = tf.reduce_sum(top2, 1)
        div = tf.expand_dims(div, 1)
        feature_agg = tf.divide(tf.matmul(tf.transpose(top2), hidden), div)
        inter1 = tf.matmul(feature_agg, tf.transpose(feature_agg))
        sim = tf.cast((inter1 > 0.01), tf.float32)
        final = tf.reduce_sum(sim, 1)

        self._xishu = inter
        self._feature_center = feature_agg
        self._result = final
        self._feature = hidden
        self.time = 0

    def clust(self, hidden):
        self.time = tf.assign(self.time, self.time+1)
        if self.time % 100000 == 0:
            self.radius = self.radius/2
            self.alpha = self.alpha * 0.1

        i = 0
        with tf.device('/gpu:%d' % i):
            hidden1 = hidden[i*256:(i+1)*256]
            x = tf.expand_dims(hidden1, 1)
            w = tf.expand_dims(self.weights, 0)
            dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                x, w), reduction_indices=[2]), 'dist')
            bmu1 = tf.argmin(dist, -1, name='bmu')

        i = 1
        with tf.device('/gpu:%d' % i):
            hidden1 = hidden[i*256:]
            x = tf.expand_dims(hidden1, 1)
            w = tf.expand_dims(self.weights, 0)
            dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                x, w), reduction_indices=[2]), 'dist')
            bmu2 = tf.argmin(dist, -1, name='bmu')

        bmu = tf.concat([bmu1, bmu2], -1)
        # map in 2D
        # weights2d = tf.reshape(
        #     self.weights, [SOM_HEIGHT, SOM_WIDTH, NUM_INPUT])
        # BMU coordinates in 2D
        bmuc = tf.convert_to_tensor(self.to2d(bmu))
        # distance from map nodes to BMU
        bmuc = tf.expand_dims(tf.transpose(bmuc), 1)
        nodess = tf.expand_dims(self.nodes, 0)
        distance_to_bmu = tf.sqrt(tf.reduce_sum(tf.cast(tf.squared_difference(
            bmuc, nodess), dtype=tf.float64), reduction_indices=[2]))
        # distance_to_bmu2d = tf.reshape(
        #     distance_to_bmu, [distance_to_bmu.get_shape()[0], SOM_HEIGHT, SOM_WIDTH])

        nf = tf.maximum(-(distance_to_bmu - self.radius) / self.radius, 0)
        # nf2d = tf.reshape(nf, [nf.get_shape()[0], SOM_HEIGHT, SOM_WIDTH])
        x = tf.expand_dims(hidden, 1)
        adjusted = tf.cast(tf.expand_dims(nf, 2) *
                           tf.expand_dims(self.alpha, 1), tf.float32) * (x - w)
        adjusted = tf.reduce_sum(adjusted, 0)
        adjusted = self.weights + adjusted
        # adj2d = tf.reshape(adjusted, [SOM_HEIGHT, SOM_WIDTH, NUM_INPUT])

        self.weights = tf.assign(self.weights, adjusted)
        self._feature_center = tf.gather(self.weights, bmu)
        self._feature = hidden
        self._xishu = bmu

        return self.time, self.weights

    def _build_graph(self, inputs):
        categorical_part = tf.concat(
            [tf.squeeze(inputs[name], axis=1)
             for name in self._categorical_features],
            axis=1,
        )
        # if len(self._numerical_features) != 0:
        #     numerical_part = tf.concat(
        #         [inputs[name] for name in self._numerical_features], axis=1
        #     )
        # else:
        #     numerical_part = []

        multivalue_part = self._build_multivalue_part(inputs)

        hidden = tf.concat([categorical_part, multivalue_part], axis=1)

        # hidden = tf.layers.dense(
        #         inputs=hidden,
        #         units=64,
        #         kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #         activation=tf.nn.leaky_relu,
        #         use_bias=True,
        #         # kernel_regularizer=regularizer,
        #     )

        distance_to_bmu, we = self.clust(hidden)

        return self._feature_center, self._feature, distance_to_bmu, we
