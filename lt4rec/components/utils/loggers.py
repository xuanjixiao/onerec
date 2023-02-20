# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import defaultdict

class TrainLogger(object):

    def __init__(self, log_steps, train_hour, tensorboard_logdir=None):
        self._log_steps = log_steps
        self._tensorboard_writer = None
        self._train_hour = train_hour
        if tensorboard_logdir:
            self._tensorboard_writer = tf.summary.FileWriter(tensorboard_logdir)
        self._cleanup()

    def log_info(self, loss_dic, time, size, epoch, step):
        for k,values in loss_dic.items():
            self._total_loss_dic[k] += values
        self._total_time += time
        self._total_size += size
        self._total_steps += 1

        if self._total_steps >= self._log_steps:
            fps = self._total_size / float(self._total_time)
            for name,values in self._total_loss_dic.items():
                avg_loss = values / self._total_steps
                self._log_to_console(avg_loss, self._total_time, fps, epoch, step,name)
                self._log_to_tensorboard(avg_loss, fps, step,name)
            self._cleanup()

    def _log_to_console(self, loss, time, fps, epoch, step,name):
        print(
            "[Train-%s] Epoch: %d\tStep: %d\t %s: %.5f\tTime: %.2f\tFPS: %d"
            % (self._train_hour, epoch, step,name,loss,time, fps)
        )

    def _log_to_tensorboard(self, loss, fps, step,name):
        if self._tensorboard_writer:
            summary = tf.Summary(
                value=[tf.Summary.Value(node_name=self._train_hour, tag="train_%s" %(name), simple_value=loss),
                       tf.Summary.Value(node_name=self._train_hour, tag="train_fps", simple_value=fps)]
            )
            self._tensorboard_writer.add_summary(summary, step)
            self._tensorboard_writer.flush()

    def _cleanup(self):
        self._total_loss_dic = defaultdict(float)
        self._total_time = 0
        self._total_instance = 0
        self._total_size = 0
        self._total_steps = 0


class ValidateLogger(object):
    def __init__(self, prefix, validate_hour=None, tensorboard_logdir=None):
        self._tensorboard_writer = None
        self._validate_hour = validate_hour
        self._prefix = prefix
        if tensorboard_logdir:
            self._tensorboard_writer = tf.summary.FileWriter(tensorboard_logdir)

    def log_info(self, metric_results, epoch, step):
        self._log_to_console(metric_results, epoch, step)
        self._log_to_tensorboard(metric_results, step)

    def _log_to_console(self, metric_results, epoch, step):
        results_str = "\t".join(
            ["%s: %s" % (metric_name, metric_result)
             for metric_name, metric_result in metric_results.items()]
        )
        print("[%s-%s] Epoch: %d\tStep: %d\t%s" % (self._prefix, self._validate_hour, epoch, step, results_str))

    def _log_to_tensorboard(self, metric_results, step):
        if self._tensorboard_writer:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=("%s_%s" % (self._prefix, metric_name)),
                                        simple_value=metric_result.result)
                       for metric_name, metric_result in metric_results.items()]
            )
            self._tensorboard_writer.add_summary(summary, step)
            self._tensorboard_writer.flush()

class Logger(object):
    def __init__(self,name,flag=True):
        self._file_name=name
        self.flag=flag
    def info(self,logs):
        if self.flag:
            print("--{}--info:{}".format(self._file_name,logs))