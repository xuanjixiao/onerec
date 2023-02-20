# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Dict, List, Any

import numpy as np


class MetricResult(object):
    """Class for metric result."""

    def __init__(self, result: float, meta: Dict[str, Any] = {}):
        """Construct a metric result.

        :param result: The metric's result.
        :param meta: A map of other meta metarmation.
        """
        self._result = result
        self._meta = meta

    @property
    def result(self):
        return self._result

    @property
    def meta(self):
        return self._meta

    def __repr__(self):
        if self._meta:
            meta_str = ','.join(['%s:%s' % (k, v) for k, v in self._meta.items()])
            return "%f (%s)" % (self._result, meta_str)
        else:
            return "%f" % self._result


class BaseMetric(metaclass=abc.ABCMeta):
    """Base class for a evaluative metric.

    All subclasses of BaseMetric must override `eval` and `required_label_names` methods
    to conduct the metric evaluation and provide names of required data fields.
    """

    @abc.abstractmethod
    def eval(self,
             predict: np.ndarray,
             labels_map: Dict[str, np.ndarray]) -> MetricResult:
        """Evaluate the metric.

        :param predict: The prediction/inference `np.ndarray` results.
        :param labels_map: A map of other `np.ndarray` data fields,
            e.g. ground-truth labels, required for the metric evaluation.
        :return: Metric results. """
        pass

    @property
    @abc.abstractmethod
    def required_label_names(self) -> List[str]:
        """Returns the names of all required data fields.

        :return: A list of required data fields' name."""
        pass
