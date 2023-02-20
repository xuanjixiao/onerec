# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from components.statistics_gens.statistics import Statistics


class BaseStatisticsGen(metaclass=abc.ABCMeta):
    """Base class for a statistics generator component.

    All subclasses of BaseStatisticsGen must override `run` method to generate
    feauture statistics.
    """

    @abc.abstractmethod
    def run(self) -> Statistics:
        """Generate feature statistics.

        :return: Feature statistics results."""
        pass
