# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.datasets.base_dataset import BaseDataset


class TFRecordDataset(BaseDataset):

    def __init__(self,
                 filepath,
                 batch_size,
                 file_repeat=False,
                 file_shuffle=False,
                 num_parallels=16,
                 shuffle_buffer_size=10000,
                 read_buffer_size=int(1e8),
                 prefetch_buffer_size=1,
                 map_functions=[],
                 drop_remainder=True,
                 sloppy=True):
        self._read_buffer_size = read_buffer_size
        self._sloppy = sloppy
        super(TFRecordDataset, self).__init__(
            filepath=filepath,
            batch_size=batch_size,
            file_repeat=file_repeat,
            file_shuffle=file_shuffle,
            num_parallels=num_parallels,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            map_functions=map_functions,
            drop_remainder=drop_remainder,
        )

    def _build_raw_dataset(self, files):
        dataset = files.apply(
            tf.data.experimental.parallel_interleave(
                lambda f: tf.data.TFRecordDataset(
                    f, "GZIP", buffer_size=self._read_buffer_size
                ),
                cycle_length=self._num_parallels,
                sloppy=self._sloppy,
            )
        )
        return dataset
