# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeatureConfig(object):

    def __init__(self, name, dtype, size, default_value=None):
        assert dtype in ("int64", "float32", "string")
        self.name = name
        self.dtype = {"int64": tf.int64,
                      "float32": tf.float32,
                      "string": tf.string}[dtype]
        self.size = size
        if default_value is None:
            if dtype == "string":
                self.default_value = "-1"
            else:
                self.default_value = 0
        else:
            self.default_value = default_value
