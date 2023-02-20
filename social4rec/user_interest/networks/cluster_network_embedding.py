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
NUM_INPUT = 64
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

        self.weights = tf.get_variable("weights", shape=[SOM_LEN, NUM_INPUT],
                initializer=tf.contrib.layers.xavier_initializer())
        # self.weights = tf.Variable(tf.random_uniform([SOM_LEN, NUM_INPUT], minval=-1.0, maxval=1.0, dtype=tf.float32), name='weights')
        # self.radius = tf.placeholder(tf.float64, [1], name='radius')
        self.radius = tf.constant(30.0, dtype=tf.float64, shape=[1], name='radius')
        self.weights1 = tf.placeholder(dtype=tf.float32, shape=[SOM_LEN, NUM_INPUT])
        # tf.placeholder()
        # learning rate
        self.alpha = tf.constant(0.001,  shape=[1], dtype=tf.float64,name='alpha')
        self.nodes = tf.reshape(tf.transpose(tf.constant(np.indices((SOM_HEIGHT, SOM_WIDTH))), perm=[1, 2, 0]), [SOM_LEN, 2])
    def to2d(self, x):

        return [x // SOM_WIDTH, x % SOM_WIDTH]
    def to1d(self, h, w):
        return h * SOM_WIDTH + w

    def _train_fn(self, example, inputs):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            omgid = inputs['FEA_OMGID']
            omgid = tf.squeeze(omgid, 1)
            feature, xishu, result = self._build_graph(example(inputs))
            # loss = self._loss.loss_fn(fea, index)
            # kk = [example(inputs)[name] for name in self._categorical_features]
            # tf.logging.info("%s %s",(omgid, str(kk)))
            # loss = tf.losses.mean_squared_error(fea_center, feature)
            return omgid, feature, xishu, result

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
    def _cosine(self,_matrixA, _matrixB):

        _matrixA_matrixB = tf.matmul(_matrixA,tf.transpose(_matrixB))
        ### 按行求和，生成一个列向量
        ### 即各行向量的模
        _matrixA_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_matrixA,_matrixA), 1))
        _matrixB_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_matrixB,_matrixB), 1))
        _matrixA_norm = tf.expand_dims(_matrixA_norm, axis=1)
        _matrixB_norm = tf.expand_dims(_matrixB_norm, axis=1)
        return tf.div(_matrixA_matrixB, tf.matmul(_matrixA_norm,tf.transpose(_matrixB_norm))+1e-8)

    # def _normalize_adj(self, adj):
    #     """Symmetrically normalize adjacency matrix."""
    #     adj = sp.coo_matrix(adj)
    #     rowsum = np.array(adj.sum(1))
    #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _find_neibor(self, hidden):
        # hidden = tf.expand_dims(hidden, axis=1)
        # hidden = tf.tile(input=hidden, multiples=[2,3])

        # self._feature_record.setdefault(feature_id, hidden)
        # hidden = self._feature_record[feature_id]

        # result = tf.where(tf.equal(feature_id, fea_index))
        # hidden[result[]]
        # print(fea_record.keys())
        inter = tf.matmul(hidden, tf.transpose(hidden))
        test_sum = tf.reduce_sum(tf.square(hidden), axis=1)  # num_test x 1
        train_sum = tf.reduce_sum(tf.square(hidden), axis=1)  # num_train x 1
        # print(test_sum.shape)
        test_sum = tf.expand_dims(test_sum, 1)
        train_sum = tf.expand_dims(train_sum, 1)
        ones = tf.ones((tf.shape(train_sum)[0]))
        dia_ones = tf.matrix_diag(ones)
        dists = tf.sqrt(-2 * inter + tf.transpose(test_sum) + train_sum + dia_ones)
        # print('sss')
        # sim = self._cosine(hidden, hidden)
        # print(sim.shape)

        sim_thresold = tf.cast(((inter+dia_ones) > 0.01), tf.float32)
        # sim_thresold = sim > 0.5
        # dists_sim = tf.multiply(dists, sim_thresold)
        # print(dists_sim.shape)
        # print(dists.shape)
        # tf.nn.top_k(input, k=1, sorted=True, name=None)
        # distance = tf.reduce_sum(tf.abs(tf.add(hidden, tf.negative(hidden[0]))), reduction_indices=1)
        top_k = tf.nn.top_k(-dists, k=200, sorted=True, name=None)

        # top_k_index = top_k.indices

        # flag_index = tf.zeros((tf.shape(hidden)[0], tf.shape(hidden)[0]))

        # print("#$"*30)
        # print(top_k_index)
        kth = tf.reduce_min(top_k.values,1,keepdims=True) # 找出最小值
        top21 = tf.cast(tf.greater_equal(-dists, kth), tf.float32) + dia_ones


        top2 = tf.cast((tf.multiply(top21, sim_thresold) > 0), tf.float32)


        div = tf.reduce_sum(top2, 1)
        div1 = tf.expand_dims(div, 1)
        # flag_index[top_k_index].assign(1)
        # print("#"*30)
        feature_agg = tf.divide(tf.matmul(tf.transpose(top2), hidden),div1)
        # print(top_k_index.shape)
        # self._neibor_index[feature_id] = top_k_index
        # sim = self._cosine(feature_agg, feature_agg)

        inter1 = tf.matmul(feature_agg, tf.transpose(feature_agg))
        sim = tf.cast((inter1 > 0.01), tf.float32)
        final = tf.reduce_sum(sim, 1)

        # hidden = hidden + 0.001 * (feature_agg - hidden)
        # self._feature_record[feature_id] = hidden
        # print(feature_id.shape)
        # print(hidden.shape)
        # record = tf.map_fn(lambda x:(x[0], x[1]), (feature_id,hidden))
        # record = tf.squeeze(feature_id, 1)
        self._xishu = inter
        # record = dict(tf.map_fn(lambda x:(x[0], x[1]), (record, hidden)))
        self._feature_center = feature_agg
        # self._feature_record[record[0][0]] = feature_id[1][0]
        # self._feature_record = zip(feature_id, hidden)
        # print(self._feature_record)
        # exit()
        # self._feature_record = fea_record
        self._result = final
        self._feature = hidden
        # self._feature_record[feature_id] = hidden


    def clust(self, hidden):
        # random_index = tf.random_uniform([1], minval=0, maxval=hidden, dtype=tf.int32)
        # data_vector = tf.gather(iris_input, random_index)
        x = tf.expand_dims(hidden, 1)
        w = tf.expand_dims(self.weights1, 0)

        # distance from map nodes to input vectors
        dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(x, w), reduction_indices=[2]), 'dist')
        # best matching unit (BMU) index

        # print(dist.shape)
        bmu = tf.argmin(dist, -1, name='bmu')
        center = tf.gather(self.weights1, bmu)
        # print(bmu)
        # # map in 2D
        # weights2d = tf.reshape(self.weights, [SOM_HEIGHT, SOM_WIDTH, NUM_INPUT])
        # # BMU coordinates in 2D
        # bmuc = tf.convert_to_tensor(self.to2d(bmu))
        # print(bmuc)
        # print(self.nodes.shape)
        # # exit()
        # # distance from map nodes to BMU
        # bmuc = tf.expand_dims(tf.transpose(bmuc), 1)
        # self.nodes = tf.expand_dims(self.nodes, 0)
        # distance_to_bmu = tf.sqrt(tf.reduce_sum(tf.cast(tf.squared_difference(bmuc, self.nodes), dtype=tf.float64), reduction_indices=[2]))
        # print(distance_to_bmu.shape)
        # # print(distance_to_bmu.get_shape[0])
        # distance_to_bmu2d = tf.reshape(distance_to_bmu, [distance_to_bmu.get_shape()[0], SOM_HEIGHT, SOM_WIDTH])

        # # neighbourhood function (how map nodes will be affected by weights adjustment)
        # # print(-(distance_to_bmu - self.radius) / self.radius)
        # nf = tf.maximum(-(distance_to_bmu - self.radius) / self.radius, 0)
        # # print(nf)
        # # exit()
        # nf2d = tf.reshape(nf, [nf.get_shape()[0], SOM_HEIGHT, SOM_WIDTH])
        # # print(nf)
        # # print(nf2d)
        # # exit()
        # adjusted = self.weights +  tf.cast(tf.expand_dims(nf, 2) * tf.expand_dims(self.alpha, 1), tf.float32) *(x - w)
        # # print(adjusted)
        # adjusted = tf.reduce_mean(adjusted, 0)
        # adj2d = tf.reshape(adjusted, [SOM_HEIGHT, SOM_WIDTH, NUM_INPUT])
        # adjust = tf.assign(self.weights, adjusted)
        # print(adjust)
        # self._feature_center = tf.gather(self.weights, bmu)
        self._feature = hidden
        self._xishu = bmu
        self._result = center

    def _build_graph(self, inputs):
        categorical_part = tf.concat(
            [tf.squeeze(inputs[name], axis=1) for name in self._categorical_features],
            axis=1,
        )
        if len(self._numerical_features) != 0:
            numerical_part = tf.concat(
                [inputs[name] for name in self._numerical_features], axis=1
            )
        else:
            numerical_part = []
        multivalue_part = self._build_multivalue_part(inputs)

        # print(categorical_part.shape)
        # print(multivalue_part.shape)
        # exit()
        hidden = tf.concat([categorical_part, multivalue_part], axis=1)



        self.clust(hidden)
        # print("*"*30)
        # print(hidden.shape)
        # exit()
        # for i, size in enumerate(self._hidden_sizes):
        #     hidden = slim.fully_connected(hidden, size, scope="fc_" + str(i))
        # outputs = slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")
        return self._feature, self._xishu, self._result
