# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from components.networks.base_network import BaseNetwork


class NullNetwork(BaseNetwork):

    def __init__(self, input_as_output):
        self._input_as_output = input_as_output

    def _train_fn(self, example):
        raise NotImplementedError

    def _eval_fn(self, example):
        return example[self._input_as_output]

    def _serve_fn(self, example):
        return self._eval_fn(example)
