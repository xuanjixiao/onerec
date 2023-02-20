# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from components.transforms.base_transform import BaseTransform


class CrossTransform(BaseTransform):

    def __init__(self, cross_schema):
        self._cross_schema = cross_schema

    def _transform_fn(self, example):
        pass
