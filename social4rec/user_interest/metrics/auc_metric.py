# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from components.metrics.base_metric import BaseMetric
from components.metrics.base_metric import MetricResult


class AUC(BaseMetric):

    def __init__(self, label_name):
        self._label_name = label_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            auc = roc_auc_score(y_true=label, y_score=predict)
            return MetricResult(result=auc, meta={'#': predict.size})

    @property
    def required_label_names(self):
        return [self._label_name]


class GroupAUC(BaseMetric):

    def __init__(self, label_name, group_key_name):
        self._label_name = label_name
        self._group_key_name = group_key_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        group_key = labels_map[self._group_key_name]
        predict_groups, label_groups = defaultdict(list), defaultdict(list)
        for l, p, key in zip(label.flatten(), predict.flatten(), group_key.flatten()):
            predict_groups[key].append(p)
            label_groups[key].append(l)

        weight_sum, auc_sum = 0, 0
        for key in label_groups.keys():
            if len(label_groups) >= 2:
                n_pos = sum(label_groups[key])
                n_neg = len(label_groups[key]) - n_pos
                if n_pos > 0 and n_neg > 0:
                    weight_sum += n_pos * n_neg
                    cur_auc = roc_auc_score(y_true=label_groups[key],
                                            y_score=predict_groups[key])
                    auc_sum += n_pos * n_neg * cur_auc

        if weight_sum == 0:
            return MetricResult(result=float('nan'))
        else:
            gauc = auc_sum / weight_sum
            return MetricResult(
                result=gauc,
                meta={'#': predict.size, '#pairs': weight_sum}
            )

    @property
    def required_label_names(self):
        return [self._label_name, self._group_key_name]


class WeightedAUC(BaseMetric):

    def __init__(self, label_name, weight_name):
        self._label_name = label_name
        self._weight_name = weight_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        weight = np.maximum(labels_map[self._weight_name], 1)
        if np.sum(label) == 0 or np.sum(label) == label.size:
            return MetricResult(result=float('nan'))
        else:
            wauc = roc_auc_score(y_true=label, y_score=predict, sample_weight=weight)
            return MetricResult(result=wauc)

    @property
    def required_label_names(self):
        return [self._label_name, self._weight_name]


class WeightedGroupAUC(BaseMetric):

    def __init__(self, label_name, group_key_name, weight_name):
        self._label_name = label_name
        self._group_key_name = group_key_name
        self._weight_name = weight_name

    def eval(self, predict, labels_map):
        label = labels_map[self._label_name]
        group_key = labels_map[self._group_key_name]
        weight = np.maximum(labels_map[self._weight_name], 1)
        predict_groups, label_groups = defaultdict(list), defaultdict(list)
        weight_groups = defaultdict(list)
        for l, p, key, w in zip(label.flatten(),
                                predict.flatten(),
                                group_key.flatten(),
                                weight.flatten()):
            predict_groups[key].append(p)
            label_groups[key].append(l)
            weight_groups[key].append(w)

        weight_sum, auc_sum = 0, 0
        for key in label_groups.keys():
            if len(label_groups) >= 2:
                n_pos = sum(label_groups[key])
                n_neg = len(label_groups[key]) - n_pos
                if n_pos > 0 and n_neg > 0:
                    weight_sum += n_pos * n_neg
                    cur_auc = roc_auc_score(y_true=label_groups[key],
                                            y_score=predict_groups[key],
                                            sample_weight=weight_groups[key])
                    auc_sum += n_pos * n_neg * cur_auc

        if weight_sum == 0:
            return MetricResult(result=float('nan'))
        else:
            wgauc = auc_sum / weight_sum
            return MetricResult(result=wgauc)

    @property
    def required_label_names(self):
        return [self._label_name, self._group_key_name]
