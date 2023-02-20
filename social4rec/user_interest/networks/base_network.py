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
