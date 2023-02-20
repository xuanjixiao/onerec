# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.statistics_gens.base_statistics_gen import BaseStatisticsGen
from components.statistics_gens.statistics import Statistics


class DatasetStatisticsGen(BaseStatisticsGen):

    def __init__(self, dataset, num_batches=None):
        self._dataset = dataset
        self._num_batches = num_batches

    def run(self) -> Statistics:
        sess = tf.Session()
        self._dataset.init(sess)
        statistics = Statistics()
        n = 0
        while True:
            try:
                batch = sess.run(self._dataset.next_batch)
            except tf.errors.OutOfRangeError:
                break
            for name, values in batch.items():
                statistics.update(name, values)
            n += 1
            if n % 1000 == 0:
                print("Statistics collected for %d batches ..." % n)
            if self._num_batches is not None and n >= self._num_batches:
                break
        sess.close()
        print("Statistics collected for %d batches in total." % n)
        return statistics
