# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from components.statistics_gens.base_statistics_gen import BaseStatisticsGen
from components.statistics_gens.statistics import Statistics


class ExternalStatisticsGen(BaseStatisticsGen):

    def __init__(self, filepath: str, datatype = 'pickle', threshold = 0):
        self._filepath = filepath
        self._datatype = datatype
        self._threshold = threshold

    def run(self) -> Statistics:
        statistics = Statistics()
        if self._datatype == 'pickle':
            statistics.load_from_file(self._filepath)
        elif self._datatype == 'text':
            statistics.load_from_textfile(self._filepath, self._threshold)
        return statistics
