# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from components.utils.loggers import TrainLogger, ValidateLogger
from tensorflow.python import pywrap_tensorflow
from numpy import *
from collections import defaultdict
import numpy as np
import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 dataset,
                 transform_functions,
                 train_fn,
                 validate_steps,
                 log_steps,
                 learning_rate,
                 train_epochs=1,
                 evaluator=None,
                 weights=None,
                 save_checkpoints_dir=None,
                 restore_checkpoint_path=None,
                 validate_at_start=False,
                 tensorboard_logdir=None):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._train_fn = train_fn
        self._train_epochs = train_epochs
        self._save_checkpoints_dir = save_checkpoints_dir
        self._restore_checkpoint_path = restore_checkpoint_path
        self._validate_steps = validate_steps
        self._log_steps = log_steps
        self._learning_rate = learning_rate
        self._validate_at_start = validate_at_start
        self._evaluator = evaluator
        self._fea_record = defaultdict(list)
        self.weights1 = weights
        self._result, self._omgid, self._feature, self._loss = self._build_train_graph()
        self._valid_logger = ValidateLogger(tensorboard_logdir)
        self._train_logger = TrainLogger(self._log_steps, tensorboard_logdir)
        self.total_num = []
        self.total_dict = {}
        self.k_weight = None
        self.k_feature = []
        self.k_omgid = []

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )

    def _generate_tfrecord(self, annotation_dict, record_path, resize=None):
        num_tf_example = 0
        writer = tf.python_io.TFRecordWriter(record_path)
        for id, feature in annotation_dict.items():

            id = "omgid="+id.decode()
            example = tf.train.Example(features=tf.train.Features(feature={
                "omgid": self._bytes_feature(id.encode()),
                "raw_feature": self._bytes_feature(feature.tobytes()),
            }))

            writer.write(example.SerializeToString())
            num_tf_example += 1
        writer.close()


    def run(self, sess=None):
        if sess is None:
            self._sess = self._create_and_init_session()
        else:
            self._sess = sess

        self._train_loop()

        for i in self.total_num:
            if i not in self.total_dict.keys():
                self.total_dict[i] = 1
            else:
                self.total_dict[i] += 1

        choose = []
        t = sorted(self.total_dict.items(), key=lambda kv: (kv[1], kv[0]))
        for i in range(300):
            choose.append(t[-(i+1)][0])

        _, clusterAssment, set_num = self.kMeans(
            self.k_weight, mean=choose, sort_list=choose)
        with tf.Session() as sess:
            record_path = '***/user_test.tfrecords'
            num_tf_example = 0
            writer = tf.python_io.TFRecordWriter(record_path)
            for cent in range(300):   # 重新计算中心点
                feature = []
                omg = []
                cents = set_num[nonzero(clusterAssment[:, 0].A == cent)[0]]
                for cent in cents:
                    index = np.array(nonzero(self.total_num == cent)[0])
                    if len(feature) == 0:
                        feature = self.k_feature[index]
                        omg = self.omgid[index]
                    else:
                        feature = np.concatenate(
                            (feature, self.k_feature[index]), axis=0)
                        omg = np.concatenate((omg, self.omgid[index]), axis=0)
                if feature.shape[0] >= 20:
                    k_value = 20
                else:
                    k_value = feature.shape[0]
                feature2 = tf.placeholder(tf.float32, shape=[None, 32])
                feature1 = tf.placeholder(tf.float32, shape=[None, 32])

                x = tf.expand_dims(feature2[:, :8], 1)
                w = tf.expand_dims(feature1[:, :8], 0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                    x, w), reduction_indices=[2]), 'dist')
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_1 = tf.reduce_mean(
                    tf.gather(feature1, top_k.indices), 1)

                x = tf.expand_dims(feature2[:, 8:16], 1)
                w = tf.expand_dims(feature1[:, 8:16], 0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                    x, w), reduction_indices=[2]), 'dist')
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_2 = tf.reduce_mean(
                    tf.gather(feature1, top_k.indices), 1)

                x = tf.expand_dims(feature2[:, 16:24], 1)
                w = tf.expand_dims(feature1[:, 16:24], 0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                    x, w), reduction_indices=[2]), 'dist')
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_3 = tf.reduce_mean(
                    tf.gather(feature1, top_k.indices), 1)

                x = tf.expand_dims(feature2[:, 24:], 1)
                w = tf.expand_dims(feature1[:, 24:], 0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(
                    x, w), reduction_indices=[2]), 'dist')
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_4 = tf.reduce_mean(
                    tf.gather(feature1, top_k.indices), 1)

                final_feature = tf.concat(
                    [feature_1, feature_2, feature_3, feature_4], -1)

                i = 0
                final = []
                flag = False
                while True:
                    if (i+1)*256 >= feature.shape[0]:
                        feature_final = sess.run([final_feature], feed_dict={
                                                 feature1: feature, feature2: feature[i*256:]})
                        flag = True
                    else:
                        feature_final = sess.run([final_feature], feed_dict={
                                                 feature1: feature, feature2: feature[i*256:(i+1)*256]})
                    if len(final) == 0:
                        final = feature_final[0]
                    else:
                        final = np.concatenate(
                            (final, feature_final[0]), axis=0)
                    i += 1
                    if flag:
                        break
                print(np.array(final).shape)
                print(omg.shape)

                for i in range(len(omg)):
                    id = omg[i]
                    feature = final[i]
                    id = "omgid="+id.decode()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "omgid": self._bytes_feature(id.encode()),
                        "raw_feature": self._bytes_feature(feature.tobytes()),
                    }))

                    writer.write(example.SerializeToString())
                    num_tf_example += 1

            writer.close()
            print("{} tf_examples has been created successfully, which are saved in {}".format(
                num_tf_example, record_path))

        print("finish")

        if sess is None:
            self._sess.close()

    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _build_train_graph(self):
        transform_fn = self._join_pipeline(self._transform_functions)
        result, omgid, feature, loss = self._train_fn(
            transform_fn, self._dataset.next_batch)
        # grads = tf.gradients(loss, trainable_params)
        # train_op = trainer.apply_gradients(list(zip(grads, trainable_params)))
        return result, omgid, feature, loss

    def _train_loop(self):

        model_reader = pywrap_tensorflow.NewCheckpointReader(
            '***/ckpt_epoch-1')

        # 使reader变换成类似于dict形式的数据
        # var_dict = model_reader.get_variable_to_shape_map()
        # for key in var_dict:
        #     print("variable name: ", key)
        #     print(model_reader.get_tensor(key))
        self.k_weight = model_reader.get_tensor('weights')
        self._restore_checkpoint()

        # 最后，循环打印输出
        # for key in var_dict:
        #     print("variable name: ", key)
        #     print(model_reader.get_tensor(key))
        if self._validate_at_start:
            self._validate(epoch=0, step=0)

        step = 0
        for epoch in range(self._train_epochs):
            self._dataset.init(self._sess)
            xishu = []
            while True:
                success = self._train_step(
                    epoch, step, xishu, self.k_feature, self.k_weight)
                print(success)
                if not success:
                    break
                step += 1

                print(step)
            # self._save_checkpoint(epoch + 1)

    # 计算欧几里得距离

    def distEclud(self, vecA, vecB):
        return np.sqrt(sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离

    # 构建聚簇中心，取k个(此例中为4)随机质心
    def randCent(self, dataSet, k, sort_list):
        n = shape(dataSet)[1]
        m = shape(dataSet)[0]
        centroids = mat(zeros((k, n)))   # 每个质心有n个坐标值，总共要k个质心
        centroids = dataSet[sort_list, :]
        return centroids, sort_list

    # k-means 聚类算法

    def kMeans(self, dataSet, k=300, mean=0, sort_list=None):
        set_num = np.array(list(set(self.total_num)))
        m = len(set_num)
        print(m, 'ssss')
        clusterAssment = mat(zeros((m, 2)))    # 用于存放该样本属于哪类及质心距离
        # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
        centroids, cen_choose = self.randCent(dataSet, k, sort_list)
        clusterChanged = True   # 用来判断聚类是否已经收敛

        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # 把每一个数据点划分到离它最近的中心点
                minDist = inf
                minIndex = -1
                for j in range(k):
                    distJI = self.distEclud(
                        centroids[j, :], dataSet[set_num[i], :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True  # 如果分配发生变化，则需要继续迭代
                # 并将第i个数据点的分配情况存入字典
                clusterAssment[i, :] = minIndex, cen_choose[minIndex]
            for cent in range(k):   # 重新计算中心点
                ptsInClust = dataSet[set_num[nonzero(clusterAssment[:, 0].A == cent)[
                    0]]]   # 去第一列等于cent的所有列
                centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 算出这些数据的中心点
            tf.logging.info("centroid %s", centroids)
        return centroids, clusterAssment, set_num

    def _train_step(self, epoch, step, xishu, feature, weights):

        try:
            omgid, feature1, _, _ = self._sess.run(
                [self._result, self._omgid, self._feature, self._loss], feed_dict={self.weights1: weights})

            # fea_record = dict(zip(xishu1, omgid))
            # for k,v in fea_record.items():
            #     self._fea_record[k].append(v)
            # self.total_num.extend(xishu1)

            if len(self.k_feature) == 0:

                self.omgid = omgid
                self.k_feature = feature1
            else:

                self.omgid = np.concatenate((self.omgid, omgid), axis=0)
                self.k_feature = np.concatenate(
                    (self.k_feature, feature1), axis=0)

            return True

        except tf.errors.OutOfRangeError:

            return False

    def _save_checkpoint(self, step, prefix="ckpt_epoch"):
        if self._save_checkpoints_dir:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_path = os.path.join(self._save_checkpoints_dir, prefix)
            checkpoint_saver.save(
                self._sess, checkpoint_path, global_step=step)

    def _restore_checkpoint(self):
        if self._restore_checkpoint_path:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_saver.restore(self._sess, self._restore_checkpoint_path)

    def _validate(self, epoch, step):
        if self._evaluator is not None:
            eval_results = self._evaluator.run(sess=self._sess)
            self._valid_logger.log_info(eval_results, epoch=epoch, step=step)

    def _join_pipeline(self, map_functions):

        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
