# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf


class BaseNetwork(metaclass=abc.ABCMeta):
    """Base class for a neural network component.

    All subclasses of BaseNetwork must override _train_fn, _eval_fn and _serve_fn
    methods. _train_fn builds the training graph and returns a loss `tf.Tensor`.
    _eval_fn builds the evaluation graph and outputs a inference `tf.Tensor`.
    _serve_fn might be the same to _eval_fn or do additional graph surgery for
    efficient online serving.
    """

    @property
    def train_fn(self):
        """Returns a function to build training graph.

        :return: A function to build training graph (loss as output)."""
        return self._train_fn

    @property
    def eval_fn(self):
        """Returns a function to build inference graph.

        :return: A function to build inference graph (inference result as output)."""
        return self._eval_fn

    @property
    def serve_fn(self):
        """Returns a function to build serving graph.

        :return: A function to build serving graph (inference result as output)."""
        return self._serve_fn

    @abc.abstractmethod
    def _train_fn(self, example: tf.train.Example) -> tf.Tensor:
        """Build training graph.

        :param example: The `tf.Example` used as graph input.
        :return: A loss `tf.Tensor`."""
        pass

    @abc.abstractmethod
    def _eval_fn(self, example: tf.train.Example) -> tf.Tensor:
        """Build inference graph.

        :param example: The `tf.Example` used as graph input.
        :return: An inference result `tf.Tensor`."""
        pass

    @abc.abstractmethod
    def _serve_fn(self, example: tf.train.Example) -> tf.Tensor:
        """Build serving graph.

        Additional graph surgery can be performed to accelerate online serving.

        :param example: The `tf.Example` used as graph input.
        :return: An inference result `tf.Tensor`."""
        pass

    @property
    def serve_inputs(self):
        """Returns a function to offer inputs for serving.

        :return: A function for offering inputs.
        """
        return self._get_serve_inputs()

    @abc.abstractmethod
    def _get_serve_inputs(self):
        """Return inputs for serving input.

        :return: List of input names.
        """
        pass

    def _tile_tensor_with_batch_size(self, tensor, batch_size):
        """Tile tensor for optimized model serving.

        Tile the tensor to [batch_size, 1] if its first dimension is 1.
        This is used for optimized model serving,
        which has squeezed user/context feature inputs (batch_size = 1)
        and unsqueezed item feature inputs (batch_size > 1).

        :param tensor: Input tensor.
        :param batch_size: Target batch size.
        :return: Tiled tensor with first dimension equal to `batch_size`.
        """
        dims = len(tensor.get_shape().as_list())
        shape = [1] * dims
        shape[0] = batch_size
        output = tf.cond(tf.equal(tf.shape(tensor)[0], 1),
                         lambda: tf.tile(tensor, shape),
                         lambda: tensor)
        return output

    def _tile_tensors_with_batch_size(self, tensors, batch_size):
        """Tile tensors for optimized model serving.

        For every tensor of tensors,
        do _tile_tensor_with_batch_size(tensor, batch_size).

        :param tensors: Input tensors list.
        :param batch_size: Target batch size.
        :return: Tiled tensors with first dimension of
        every tensor equal to `batch_size`.
        """
        return [
            self._tile_tensor_with_batch_size(tensor, batch_size)
            for tensor in tensors
        ]

    def _get_batch_size(self, example):
        """Get the batch size.

        Get the batch size by traversing all features' shape in example.

        :param example: Dict of string->tensor.
        :return: A tensor of batch_size.
        """
        batch_size_tensors = [tf.shape(tensor)[0] for key, tensor in example.items()
                              if key in self.serve_inputs]
        batch_size = tf.reduce_max(batch_size_tensors)
        return batch_size