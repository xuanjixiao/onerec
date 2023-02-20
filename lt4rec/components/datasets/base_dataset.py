# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Callable, List

import tensorflow as tf


class BaseDataset(metaclass=abc.ABCMeta):
    """Base class for a dataset component.

    All subclasses of BaseDataset must override the _build_raw_dataset private method
    to build the inner-most raw tf dataset. The rest of the framework will
    automatically do dataset post-processing: e.g. file shuffling, instance shuffling,
    batching, mapping or transformation, and efficient prefetching.
    """

    def __init__(self,
                 filepath: List[str],
                 batch_size: int,
                 file_repeat: bool = False,
                 file_shuffle: bool = False,
                 num_parallels: int = 16,
                 shuffle_buffer_size: int = 10000,
                 prefetch_buffer_size: int = 1,
                 map_functions: List[Callable[[tf.train.Example],
                                              tf.train.Example]] = [],
                 drop_remainder: bool = True):
        """Construct a dataset component.

        :param filepath: The filepath of the input data to load.
        :param batch_size: The number of instances to group into batches.
        :param file_repeat: Whether to repeat the files to make an endless output.
        :param file_shuffle: Whether to shuffle the input file orders.
        :param num_parallels: The number of elements to process asynchronously
            in parallel.
        :param shuffle_buffer_size: The minimum number of elements that will be cached
            before shuffling.
        :prefetch_buffer_size: The maximum number of elements that will be buffered
            when prefetching.
        :map_functions: The ordered list of mapping functions applied on the dataset.
        :drop_remainder: Whether the last batch should be dropped in the case it has
            fewer than `batch_size` elements.
        """
        self._filepath = filepath
        self._batch_size = batch_size
        self._file_repeat = file_repeat
        self._file_shuffle = file_shuffle
        self._num_parallels = num_parallels
        self._shuffle_buffer_size = shuffle_buffer_size
        self._prefetch_buffer_size = prefetch_buffer_size
        self._map_functions = map_functions
        self._drop_remainder = drop_remainder

        self._dataset = self._build_dataset()
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    def init(self, sess):
        """Initialize the dataset.

        :param sess: The tensorflow graph session used to initiliaze the dataset.
        """
        self._iterator.initializer.run(session=sess)

    @property
    def next_batch(self):
        """Returns a next batch.

        :return: A nested structure of `tf.Tensor`s containing the next batch.
        """
        return self._next_batch

    @property
    def batch_size(self):
        """Returns the batch size.

        :return: The batch size.
        """
        return self._batch_size

    @abc.abstractmethod
    def _build_raw_dataset(self, files: tf.data.Dataset) -> tf.data.Dataset:
        """Build the raw dataset.

        :param files: A `tf.Dataset` of strings corresponding to file names.
        :return: A `tf.Dataset` of instances.
        """
        pass

    def _build_dataset(self):
        files = self._read_files(self._filepath)
        dataset = self._build_raw_dataset(files)
        dataset = self._shuffle_and_batch(dataset)
        dataset = self._dataset_map(dataset)
        dataset = self._apply_prefetch(dataset)
        return dataset

    def _read_files(self, filepath):
        files = tf.data.Dataset.list_files(self._regex_expand(filepath),
                                           self._file_shuffle)
        if self._file_repeat:
            files = files.repeat()
        return files

    def _shuffle_and_batch(self, dataset):
        dataset = dataset.shuffle(self._shuffle_buffer_size)
        dataset = dataset.batch(self._batch_size, self._drop_remainder)
        return dataset

    def _dataset_map(self, dataset):
        if len(self._map_functions) > 0:
            map_fn = self._join_pipeline(self._map_functions)
            dataset = dataset.map(map_fn, num_parallel_calls=self._num_parallels)
        return dataset

    def _apply_prefetch(self, dataset):
        dataset = dataset.prefetch(self._prefetch_buffer_size)
        return dataset

    def _join_pipeline(self, map_functions):

        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn

    def _regex_expand(self, filepaths):
        results = []
        for filepath in filepaths:
            assert filepath.count('{') == filepath.count('}') and filepath.count('{') <= 1
            left_pos = filepath.find('{')
            right_pos = filepath.find('}')
            if right_pos > -1 and left_pos > -1 and right_pos > left_pos:
                expanded = filepath[left_pos + 1: right_pos].split(',')
                results = results + [filepath[:left_pos] + content + filepath[right_pos + 1:] for content in expanded]
            else:
                results.append(filepath)
        filepath = ','.join(results)
        return tf.string_split([filepath], ",").values
