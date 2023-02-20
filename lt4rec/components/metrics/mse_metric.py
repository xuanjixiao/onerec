# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error

from components.metrics.base_metric import BaseMetric
from components.metrics.base_metric import MetricResult


class MSE(BaseMetric):

    def __init__(self, label_name):
        self._label_name = label_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            mse = mean_squared_error(y_true=label, y_pred=predict)
            return MetricResult(result=mse, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._label_name]

class WeightedMSE(BaseMetric):

    def __init__(self, label_name):
        self._label_name = label_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            weight=np.where(label>0,np.ones_like(label),np.zeros_like(label))
            mse = mean_squared_error(y_true=label, y_pred=predict,sample_weight=weight)
            return MetricResult(result=mse, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._label_name]

class WeightedRMSE(BaseMetric):

    def __init__(self, label_name):
        self._label_name = label_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            weight=np.where(label>0,np.ones_like(label),np.zeros_like(label))
            mse = mean_squared_error(y_true=label, y_pred=predict,sample_weight=weight)
            rmse =np.sqrt(mse)
            return MetricResult(result=rmse, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._label_name]
