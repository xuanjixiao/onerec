# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import numpy as np
from components.utils.loggers import TrainLogger, ValidateLogger
from tensorflow.python import pywrap_tensorflow
from numpy import *
import random

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
                 weights = None,
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
        self._fea_record = {}
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
            float_list = tf.train.FloatList(value=[value])
        )



    def _generate_tfrecord(self, annotation_dict, record_path, resize=None):
        num_tf_example = 0
        writer = tf.python_io.TFRecordWriter(record_path)
        for id, feature in annotation_dict.items():

            id = "omgid="+id.decode()
            # print(id)
            # exit()
            example = tf.train.Example(features = tf.train.Features(feature = {
                                                                           "omgid": self._bytes_feature(id.encode()),
                                                                           "raw_feature": self._bytes_feature(feature.tobytes()),
                                                                           }))

            writer.write(example.SerializeToString())
            num_tf_example += 1
        writer.close()
        # print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))



    def run(self, sess=None):
        if sess is None:
            self._sess = self._create_and_init_session()
        else:
            self._sess = sess

        self._train_loop()



        for i in self.total_num:
            # print(i)
            # exit()
            if i not in self.total_dict.keys():
                self.total_dict[i] = 1
            else:
                self.total_dict[i] +=1

        choose=[]
        t = sorted(self.total_dict.items(), key = lambda kv:(kv[1], kv[0]))
        print(t)
        for i in range(5000):
            # print(t[-(i+1)])
            choose.append(t[-(i+1)][0])
        # top_k = arr[0:1000]
        print(choose)
        # print(top_k)
        # top_kk = arr[-100:]
        # print(top_kk)
        clusterAssment = self.kMeans(self.k_weight, sort_list=choose)

        print(clusterAssment)
        total_feature = []
        with tf.Session() as sess:
            record_path = '***/user2.tfrecords'
            num_tf_example = 0
            writer = tf.python_io.TFRecordWriter(record_path)
            for cent in range(5000):   # 重新计算中心点
                feature = []
                omg = []
                he = nonzero(np.array(clusterAssment) == cent)[0]
                print(he)
                print(len(he))
                for cent1 in he:
                    index = np.array(nonzero(self.total_num == cent1)[0])
                    # print(index)
                    # print(self.omgid[index])
                    if len(feature) == 0:
                        feature = self.k_feature[index]
                        omg = self.omgid[index]
                    else:
                        feature = np.concatenate((feature, self.k_feature[index]), axis=0)
                        omg = np.concatenate((omg, self.omgid[index]), axis=0)
                if feature.shape[0] >=20:
                    k_value = 20
                else:
                    k_value = feature.shape[0]
                feature2 = tf.placeholder(tf.float32, shape=[None, 64])
                feature1 = tf.placeholder(tf.float32, shape=[None, 64])
                x = tf.expand_dims(feature2[:,:16],1)
                w = tf.expand_dims(feature1[:,:16],0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(x, w), reduction_indices=[2]), 'dist')
                # best matching unit (BMU) index
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                # hist = sess.run([top_k.indices], feed_dict={feature1: feature})
                # tf.logging.info("%s",hist)
                feature_1 = tf.reduce_mean(tf.gather(feature1, top_k.indices),1)
                x = tf.expand_dims(feature2[:,16:32],1)
                w = tf.expand_dims(feature1[:,16:32],0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(x, w), reduction_indices=[2]), 'dist')
                # best matching unit (BMU) index
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_2 = tf.reduce_mean(tf.gather(feature1, top_k.indices),1)
                x = tf.expand_dims(feature2[:,32:48],1)
                w = tf.expand_dims(feature1[:,32:48],0)
                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(x, w), reduction_indices=[2]), 'dist')
                # best matching unit (BMU) index
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_3 = tf.reduce_mean(tf.gather(feature1, top_k.indices),1)
                x = tf.expand_dims(feature2[:,48:],1)
                w = tf.expand_dims(feature1[:,48:],0)

                dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(x, w), reduction_indices=[2]), 'dist')
                # best matching unit (BMU) index
                top_k = tf.nn.top_k(-dist, k=k_value, sorted=True, name=None)
                feature_4 = tf.reduce_mean(tf.gather(feature1, top_k.indices),1)
                # print(feature_4.shape)
                # tf.logging.info("%s", tf.gather(feature1, top_k.indices).shape)
                # exit()
                final_feature = tf.concat([feature_1, feature_2, feature_3, feature_4], -1)
                # print(dist.shape)
                # bmu = tf.argmin(dist, -1, name='bmu')
                i = 0
                final = []
                flag = False
                while True:
                    if (i+1)*64 >= feature.shape[0]:
                        feature_final = sess.run([final_feature], feed_dict={feature1: feature, feature2: feature[i*64:]})
                        flag = True
                    else:
                        feature_final = sess.run([final_feature], feed_dict={feature1: feature, feature2: feature[i*64:(i+1)*64]})
                    if len(final) == 0:
                        final = feature_final[0]
                    else:
                        final = np.concatenate((final, feature_final[0]), axis=0)
                    i+=1
                    if flag:
                        break
                # total_feature.append(feature)

                print(np.array(final).shape)
                print(omg.shape)

                # for id, feature in annotation_dict.items():
                for i in range(len(omg)):
                    id = omg[i]
                    feature = final[i]
                    # print(id)
                    # print(feature)
                    # exit()

                    id = "omgid="+id.decode()
                    # print(id)
                    # exit()
                    example = tf.train.Example(features = tf.train.Features(feature = {
                                                                                   "omgid": self._bytes_feature(id.encode()),
                                                                                   "raw_feature": self._bytes_feature(feature.tobytes()),
                                                                                   }))

                    writer.write(example.SerializeToString())
                    num_tf_example += 1

                # ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]

            writer.close()
            print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

        # print(clusterAssment)
        # exit()
        print("finish")



        #######
        if sess is None:
            self._sess.close()


    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _build_train_graph(self):
        trainable_params = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        transform_fn = self._join_pipeline(self._transform_functions)
        result, omgid, feature, loss = self._train_fn(transform_fn, self._dataset.next_batch)
        # grads = tf.gradients(loss, trainable_params)
        # train_op = trainer.apply_gradients(list(zip(grads, trainable_params)))
        return result, omgid, feature, loss

    def _train_loop(self):


        model_reader = pywrap_tensorflow.NewCheckpointReader('***/ckpt_epoch-1')

        #然后，使reader变换成类似于dict形式的数据
        var_dict = model_reader.get_variable_to_shape_map()
        # for key in var_dict:
        #     print("variable name: ", key)
        #     print(model_reader.get_tensor(key))
        self.k_weight = model_reader.get_tensor('weights')
        # print(weight)
        # print(weight.shape)
        self._restore_checkpoint()
        # exit()

        #最后，循环打印输出
        # for key in var_dict:
        #     print("variable name: ", key)
        #     print(model_reader.get_tensor(key))
        if self._validate_at_start:
            self._validate(epoch=0, step=0)

        step = 0
        for epoch in range(self._train_epochs):
            self._dataset.init(self._sess)
            xishu = []
            feature = []
            while True:
                success = self._train_step(epoch, step, xishu, self.k_feature, self.k_weight)
                print(success)
                if not success:
                    break
                step += 1

                print(step)
                # if step % 500 == 0:
                #     self._save_checkpoint(epoch + 1)
                #     model_reader = pywrap_tensorflow.NewCheckpointReader('./chk_points/ckpt_epoch-2')

                #     #然后，使reader变换成类似于dict形式的数据
                #     var_dict = model_reader.get_variable_to_shape_map()

                #     #最后，循环打印输出
                #     for key in var_dict:
                #         print("variable name: ", key)
                #         print(model_reader.get_tensor(key))
                #     self._validate(epoch=epoch, step=step)
            # self._save_checkpoint(epoch + 1)



    # 计算欧几里得距离
    def distEclud(self, vecA, vecB):
        return np.sqrt(sum(np.power(vecA - vecB, 2))) # 求两个向量之间的距离

    # 构建聚簇中心，取k个(此例中为4)随机质心
    def randCent(self, dataSet, k, sort_list):
        n = shape(dataSet)[1]
        m = shape(dataSet)[0]
        # choose_list = []
        centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
        new_dict = {v : k for k, v in self.total_dict.items()}
        # for i in sort_list:
            # if i not in self.total_dict.keys():
            #     continue
            # else:
            #     if self.total_dict[i] > 10000:
            # while new_dict
            # choose_list.append(new_dict[i])
                # else:
                #     continue
        # for j in range(n):
        #     minJ = min(dataSet[:,j])
        #     maxJ = max(dataSet[:,j])
        #     rangeJ = float(maxJ - minJ)
        # centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
        # print(len(choose_list))
        # cen_choose = random.sample(range(1,len(choose_list)),k)
        centroids = dataSet[sort_list,:]
        print(sort_list)
        print(centroids)
        print('sss')
        return centroids, sort_list














# # # coding=utf-8
# # import tensorflow as tf
# # import numpy as np

#     def creat_samples(n_clusters,n_samples_per_cluster,n_features,embiggen_factor,seed):
#         np.random.seed(seed)
#         slices = []
#         centroids = []
#         for i in range(n_clusters):
#             samples = tf.random_normal((n_samples_per_cluster,n_features),
#                                        mean=0,stddev=5,dtype=tf.float32,name='cluster_{}'.format(i))
#             current_centroid = (np.random.random((1,n_features))*embiggen_factor) - (embiggen_factor/2)
#             centroids.append(current_centroid)
#             samples += current_centroid
#             slices.append(samples)
#         samples = tf.concat(slices,0,name='samples')
#         centroids = tf.concat(centroids,0,name='centroids')
#         return centroids,samples

    # def plot_clusters(all_samples,centroids,n_sample_per_cluster):
    #     import matplotlib.pyplot as plt
    #     colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    #     for i,centroid in enumerate(centroids):
    #         samples = all_samples[i*n_sample_per_cluster:(i+1)*n_sample_per_cluster]
    #         plt.scatter(samples[:,0],samples[:,1],c=colour[i])
    #         plt.plot(centroid[0],centroid[1],markersize=35,marker='x',color='k',mew=10)
    #         plt.plot(centroid[0],centroid[1],markersize=35,marker='x',color='m',mew=5)
    #     plt.show()

    def choose_random_centroids(self,samples,n_clusters):
        n_samples = tf.shape(samples)[0]
        random_indices = tf.random_shuffle(tf.range(0,n_samples))
        begin = [0,]
        size = [n_clusters,]
        size[0] = n_clusters
        centroid_indices = tf.slice(random_indices,begin,size)
        init_centroids = tf.gather(samples,centroid_indices)# 通过索引将对应的向量取出来
        # print(n_samples,init_centroids,samples.get_shape().as_list())
        return init_centroids

    def assign_to_nearest(self,samples,centroids):
        vector = tf.expand_dims(samples,0)
        centroids = tf.expand_dims(centroids,1)
        error = tf.subtract(vector, centroids)
        distances = tf.reduce_sum(tf.square(error),2)
        mins = tf.argmin(distances,0)
        nearest_indices = mins
        return nearest_indices # shape as len(samples), index of cluster for per sample

    def update_centroids(self,samples,nearest_indices,n_clusters):
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(samples,nearest_indices,n_clusters)# 矩阵拆分,(data,partitions,number of partition)
        # Partitions data into num_partitions tensors using indices from partitions.
        # partitions Any shape: Indices in the range [0, num_partitions).
        # number of partition: An int that is >= 1. The number of partitions to output.
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition,0),0) for partition in partitions],0)
        return new_centroids

    def kMeans(self, dataset, sort_list):
        n_features = 64
        n_clusters = 5000
        # print(dataset.shape)
        n_samples_per_cluster = 500
        # seed = 888
        # embiggen_factor = 70
        # count = tf.constant(0)
        # np.random.seed(seed)
        cluster_center = tf.placeholder(dtype=tf.float32,shape=[n_clusters,n_features])
        samples = tf.placeholder(dtype=tf.float32,shape=[None,n_features])
        samples_batch = tf.placeholder(dtype=tf.float32,shape=[None,n_features])
        choose_list = tf.constant(sort_list)
        # centroids,samples = creat_samples(n_clusters,n_samples_per_cluster,n_features,embiggen_factor,seed)
        # init_centroids = choose_random_centroids(samples,n_clusters)
        init_centroids = tf.gather(samples, choose_list)


        nearest_indices = self.assign_to_nearest(samples_batch,cluster_center)
        # total_indices = tf.hstack([total_indices, nearest_indices])
        # count = count+1
        # print(count)
        # print(tf.shape(samples)[0])
        # print(total_indices)
        # print(total_indices.shape)
        # exit()
        # samples_batch = samples[count*256: (count+1)*256]
        # nearest_indices = self.assign_to_nearest(samples,cluster_center)
        # total_indices = tf.hstack([total_indices, nearest_indices])
        nearest_indices_ = tf.placeholder(dtype=tf.float32,shape=[len(dataset),])
        updated_centroids = self.update_centroids(samples,nearest_indices_,n_clusters)

        model = tf.global_variables_initializer()

        with tf.Session() as sess:
            # sample_values = sess.run(samples)
            centroid_value = sess.run(init_centroids, feed_dict={samples:dataset})
            # print(centroid_value)

            for i in range(100):
                count = 0
                total_indices = []
                while count*256 < len(dataset):
                    samples_batch_ = dataset[count*256: (count+1)*256]
                    near = sess.run(nearest_indices,feed_dict={cluster_center:centroid_value,samples:dataset, samples_batch:samples_batch_})
                    total_indices.extend(near)
                    count+=1
                    # print(len(near))
                    # print(len(total_indices))
                centroid_value = sess.run(updated_centroids,feed_dict={samples:dataset, nearest_indices_:total_indices})
            # print('cluster center:\n{}'.format(centroid_value))
                # tf.logging.info("%s", (total_indices))
                # tf.logging.info("%s", (len(total_indices)))
        # exit()
        # plot_clusters(sample_values, centroid_value, n_samples_per_cluster)
        return total_indices











    # k-means 聚类算法
    def kMeans11(self, dataSet, k=5000, mean = 0, sort_list=None):
        set_num = np.array(list(set(self.total_num)))
        m = len(set_num)
        print(m, 'ssss')
        clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
        # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离

        centroids, cen_choose = self.randCent(dataSet, k, sort_list)
        clusterChanged = True   # 用来判断聚类是否已经收敛

        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            # for k,i in enumerate(m):
                # if i not in self.total_dict.keys():
                #     continue
                # else:
                    # if self.total_dict[i] > mean:
                    #     continue
                minDist = inf; minIndex = -1
                for j in range(k):
                    distJI = self.distEclud(centroids[j,:], dataSet[set_num[i],:])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
                if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
                clusterAssment[i,:] = minIndex, cen_choose[minIndex]   # 并将第i个数据点的分配情况存入字典
            print(centroids)
            print(nonzero(clusterAssment[:,0].A == 0)[0])
            print(set_num[nonzero(clusterAssment[:,0].A == 0)[0]])
            for cent in range(k):   # 重新计算中心点
                ptsInClust = dataSet[set_num[nonzero(clusterAssment[:,0].A == cent)[0]]]   # 去第一列等于cent的所有列
                # print(ptsInClust)
                # print('fdfdfdfdfdfd')
                centroids[cent,:] = np.mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
            # print(centroids)
            # print('ssssss')
            tf.logging.info("centroid %s", centroids)
        return centroids, clusterAssment, set_num




    def _train_step(self, epoch, step, xishu, feature, weights):

        try:
            t_start = time.time()
            omgid, feature1, xishu1, result = self._sess.run([self._result, self._omgid, self._feature, self._loss], feed_dict={self.weights1:weights})
            # print(loss)
            # # print(result.sum())
            # print(feature1, xishu1)
            # # print(feature1.shape)
            # print(omgid, result)
            # print(xishu1)
            self.total_num.extend(xishu1)
            print(len(self.total_num))
            if len(self.k_feature) == 0:
                # xishu = xishu1
                self.omgid = omgid
                self.k_feature = feature1
            else:
                # xishu = np.concatenate((xishu1, xishu),axis=0)
                self.omgid =  np.concatenate((self.omgid, omgid), axis=0)
                self.k_feature = np.concatenate((self.k_feature, feature1), axis=0)
            # exit()
            # fea_record = dict(zip(omgid, result))
            # tf.logging.info("%s", xishu1)
            # tf.logging.info("%s", omgid)

            # if step == 50:
            #     return False
            # self._fea = (index , loss)
            # print(fea_record.keys())
            # self._fea_record = {**self._fea_record, **fea_record}
            # print(len(self._fea_record))

            # self._fea_index_cur = np.array(loss)
            # self._fea_cur = np.array(index)
            # exit()
            '''

            if step >= 90:
                if len(xishu) == 0:
                    xishu = xishu1
                    feature = feature1
                else:
                    xishu = np.concatenate((xishu1, xishu),axis=0)
                    feature = np.concatenate((feature1, feature), axis=0)
            t_end = time.time()
            # self._train_logger.log_info(loss=loss,
            #                             time=t_end - t_start,
            #                             size=self._dataset.batch_size,
            #                             epoch=epoch,
            #                             step=step + 1)
            # print(loss)
            if step > 100:
                print(feature.shape)
            if step == 110:
                # exit()
                from sklearn.manifold import TSNE
                import pandas as pd
                tsne=TSNE()
                print('fdfdfd')
                # tsne.fit_transform(feaure)  #进行数据降维,降成两维
                print('no here')
                a=tsne.fit_transform(feature) #a是一个array,a相当于下面的tsne_embedding_
                # tsne=pd.DataFrame(tsne.embedding_,index=feaure.index) #转换数据格式
                print(a)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                print(xishu)
                # plt.figure(figsize=(10000, 10000))
                def plot_embedding_2d(X, title=None):
                    #坐标缩放到[0,1]区间
                    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
                    X = (X - x_min) / (x_max - x_min)

                    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    for i in range(X.shape[0]):
                        ax.text(X[i, 0], X[i, 1],str(xishu[i]),
                                color=plt.cm.Set1(xishu[i] / 100000.),
                                fontdict={'weight': 'bold', 'size': 9})

                    if title is not None:
                        plt.title(title)

                # plt.plot(a,'r.', clip_on=False)
                plot_embedding_2d(a,"t-SNE 2D")
                plt.savefig('/cephfs/group/omg-qqv-video-mining/huaqiangdai/cluster_zhuiju/chk_points/2.jpg' ,dpi=1000)
                exit()
            #     return False
            '''
            return True

        except tf.errors.OutOfRangeError:

            return False

    def _save_checkpoint(self, step, prefix="ckpt_epoch"):
        if self._save_checkpoints_dir:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_path = os.path.join(self._save_checkpoints_dir, prefix)
            checkpoint_saver.save(self._sess, checkpoint_path, global_step=step)

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
