# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import tensorflow as tf
import os


class Evaluator(object):

    def __init__(self,
                 dataset,
                 transform_functions,
                 eval_fn,
                 ctr_metrics,
                 cvr_metrics,
                 restore_checkpoint_path=None):
        self._dataset = dataset
        self._transform_functions = transform_functions
        self._eval_fn = eval_fn
        self._ctr_metrics = ctr_metrics
        self._cvr_metrics = cvr_metrics
        self._restore_checkpoint_path = restore_checkpoint_path
        self._ctr_predict,self._cvr_predict, self._ctr_labels,self._cvr_labels = self._build_eval_graph()
    def run(self, sess=None,feed_mask=None):
        if sess is None:
            self._sess = self._create_and_init_session()
        else:
            self._sess = sess
        if feed_mask is None:
            print("you need feed mask placeholder!")
            return 
        results,cvr_predict,cvr_labels = self._eval_loop(feed_mask)

        if sess is None:
            self._sess.close()
        return results,cvr_predict,cvr_labels

    def close(self):
        tf.reset_default_graph()

    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer().run(session=sess)
        return sess

    def _eval_loop(self,feed_mask):
        # self._restore_checkpoint()
        self._dataset.init(self._sess)
        ctr_predict, ctr_labels = [], defaultdict(list)
        cvr_predict, cvr_labels = [], defaultdict(list)
        while True:
            try:
                ctr_results = self._sess.run(
                    [self._ctr_predict]
                    + [self._ctr_labels[name] for name in sorted(self._ctr_labels.keys())],feed_dict=feed_mask
                )
                cvr_results = self._sess.run(
                    [self._cvr_predict]
                    + [self._cvr_labels[name] for name in sorted(self._cvr_labels.keys())],feed_dict=feed_mask
                )
            except tf.errors.OutOfRangeError:
                break
            ctr_predict.append(ctr_results[0])
            cvr_predict.append(cvr_results[0])
            for name, result in zip(sorted(self._ctr_labels.keys()), ctr_results[1:]):
                ctr_labels[name].append(result)
            for name, result in zip(sorted(self._cvr_labels.keys()), cvr_results[1:]):
                cvr_labels[name].append(result)
        ctr_predict = np.concatenate(ctr_predict, axis=0)
        cvr_predict = np.concatenate(cvr_predict, axis=0)

        for label_name in ctr_labels.keys():
            ctr_labels[label_name] = np.concatenate(ctr_labels[label_name], axis=0)
        for label_name in cvr_labels.keys():
            cvr_labels[label_name] = np.concatenate(cvr_labels[label_name], axis=0)

        metric_results = {}
        for metric_name, metric in self._ctr_metrics.items():
            ctr_result = metric.eval(ctr_predict, ctr_labels)
            metric_results[metric_name] = ctr_result
        for metric_name, metric in self._cvr_metrics.items():
            cvr_result = metric.eval(cvr_predict, cvr_labels)
            metric_results[metric_name] = cvr_result
        return metric_results,cvr_predict,cvr_labels

    # def _restore_checkpoint(self):
    #     if self._restore_checkpoint_path:
    #         checkpoint_saver = tf.train.Saver(max_to_keep=None)
    #         checkpoint_saver.restore(self._sess, self._restore_checkpoint_path)

    def _restore_checkpoint(self):
        if os.path.exists(self._restore_checkpoint_path):
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            ckpt = tf.train.latest_checkpoint(self._restore_checkpoint_path)
            checkpoint_saver.restore(self._sess, ckpt)

    def _build_eval_graph(self):
        next_batch = self._dataset.next_batch
        transform_fn = self._join_pipeline(self._transform_functions)
        ctr_output,cvr_output = self._eval_fn(transform_fn(next_batch))
        ctr_labels = {name: next_batch[name] for name in self._all_required_label_names()[0]}
        cvr_labels = {name: next_batch[name] for name in self._all_required_label_names()[1]}
        return ctr_output,cvr_output, ctr_labels,cvr_labels

    def _all_required_label_names(self):
        ctr_required = set()
        cvr_required = set()
        for metric in self._ctr_metrics.values():
            ctr_required.update(metric.required_label_names)
        for metric in self._cvr_metrics.values():
            cvr_required.update(metric.required_label_names)
        return list(ctr_required),list(cvr_required)

    def _join_pipeline(self, map_functions):
        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
