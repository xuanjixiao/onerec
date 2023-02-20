# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from components.utils.loggers import TrainLogger, ValidateLogger


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
        self._result, self._omgid, self._feature, self._time, self.we, self._train_op = self._build_train_graph()
        self._valid_logger = ValidateLogger(tensorboard_logdir)
        self._train_logger = TrainLogger(self._log_steps, tensorboard_logdir)

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
            example = tf.train.Example(features=tf.train.Features(feature={
                "omgid": self._bytes_feature(id),
                "raw_feature": self._bytes_feature(feature.tobytes()),
            }))

            writer.write(example.SerializeToString())
            num_tf_example += 1
        writer.close()
        print("{} tf_examples has been created successfully, which are saved in {}".format(
            num_tf_example, record_path))

    def run(self, sess=None):
        if sess is None:
            self._sess = self._create_and_init_session()
        else:
            self._sess = sess

        self._train_loop()

        print("finish")

        if sess is None:
            self._sess.close()

    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _build_train_graph(self):
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        print('ssss')
        trainer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        transform_fn = self._join_pipeline(self._transform_functions)
        result, omgid, feature, time1, we, loss = self._train_fn(
            transform_fn, self._dataset.next_batch)
        grads = tf.gradients(loss, trainable_params)
        train_op = trainer.apply_gradients(list(zip(grads, trainable_params)))
        return result, omgid, feature, time1, we, train_op

    def _train_loop(self):
        # self._restore_checkpoint()
        if self._validate_at_start:
            self._validate(epoch=0, step=0)

        step = 0
        for epoch in range(self._train_epochs):
            self._dataset.init(self._sess)
            xishu = []
            feature = []
            while True:
                success = self._train_step(epoch, step, xishu, feature)
                # print(success)
                if not success:
                    break
                step += 1

                print(step)
                if step % 10000 == 0:
                    self._save_checkpoint(epoch + 2)
                # if step % self._validate_steps == 0:
                #     self._validate(epoch=epoch, step=step)
            self._save_checkpoint(epoch + 2)

    def _train_step(self, epoch, step, xishu, feature):
        try:
            _ = self._sess.run(
                [self._train_op])

            # 聚类可视化
            '''
            if step == 10050 :
                from sklearn.manifold import TSNE
                import pandas as pd
                tsne=TSNE()
                # tsne.fit_transform(feaure)  #进行数据降维,降成两维
                # try:
                a=tsne.fit_transform(feature) #a是一个array,a相当于下面的tsne_embedding_
                # except expression as identifier:
                #     pass
                # tsne=pd.DataFrame(tsne.embedding_,index=feaure.index) #转换数据格式
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
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
                                fontdict={'weight': 'bold', 'size': 8})

                    if title is not None:
                        plt.title(title)

                # plt.plot(a,'r.', clip_on=False)
                plot_embedding_2d(a,"t-SNE 2D")
            '''
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
