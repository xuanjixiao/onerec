# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TrainLogger(object):

    def __init__(self, log_steps, tensorboard_logdir=None):
        self._log_steps = log_steps
        self._tensorboard_writer = None
        if tensorboard_logdir:
            self._tensorboard_writer = tf.summary.FileWriter(tensorboard_logdir)
        self._cleanup()

    def log_info(self, loss, time, size, epoch, step):
        self._total_loss = loss
        self._total_time += time
        self._total_size += size
        self._total_steps += 1

        # if self._total_steps >= self._log_steps:
        #     avg_loss = self._total_loss / self._total_steps
        #     fps = self._total_size / float(self._total_time)
        #     self._log_to_console(avg_loss, self._total_time, fps, epoch, step)
        #     self._log_to_tensorboard(loss, fps, step)
        #     self._cleanup()

    def _log_to_console(self, loss, time, fps, epoch, step):
        print(
            "[Train] Epoch: %d\tStep: %d\tLoss: %.5f\tTime: %.2f\tFPS: %d"
            % (epoch, step, loss, time, fps)
        )

    def _log_to_tensorboard(self, loss, fps, step):
        if self._tensorboard_writer:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag="train_loss", simple_value=loss),
                       tf.Summary.Value(tag="train_fps", simple_value=fps)]
            )
            self._tensorboard_writer.add_summary(summary, step)
            self._tensorboard_writer.flush()

    def _cleanup(self):
        self._total_loss = 0
        self._total_time = 0
        self._total_instance = 0
        self._total_size = 0
        self._total_steps = 0


class ValidateLogger(object):
    def __init__(self, tensorboard_logdir=None):
        self._tensorboard_writer = None
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
        print("[Validate] Epoch: %d\tStep: %d\t%s" % (epoch, step, results_str))

    def _log_to_tensorboard(self, metric_results, step):
        if self._tensorboard_writer:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=("valid_%s" % metric_name),
                                        simple_value=metric_result.result)
                       for metric_name, metric_result in metric_results.items()]
            )
            self._tensorboard_writer.add_summary(summary, step)
            self._tensorboard_writer.flush()
