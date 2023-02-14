import numpy as np
import os
from root_path import abspath_root
from dataclasses import dataclass
from collections import namedtuple


class LagTimeCalc:
    @property
    def STARTING_THT_VALUE(self):
        return 4.0

    @property
    def SQUARE_OF_STARTING_VALUE(self):
        return np.square(self.STARTING_THT_VALUE)  # e.g. 16.0

    @property
    def DOUBLE_STARTING_VALUE(self):
        return self.STARTING_THT_VALUE * 2  # e.g. 8.0

    @property
    def _1_5_STARTING_VALUE(self):
        return self.STARTING_THT_VALUE * 1.5  # e.g. 6.0

    @property
    def MIN_NUM_OF_LAGTIMES_NEEDED(self):
        return 3

    @property
    def STR_TIME_H(self):
        return 'Time (h)'

    @property
    def ALL_THT_DATA_CSV_PATH(self):
        return os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.csv')

    @property
    def LAGTIMES_PATH(self):
        return os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes')

    @property
    def TEST_LAGTIMES_PATH(self):
        return os.path.join(abspath_root, 'test', 'lag_times', 'data', 'tht_data')

    @property
    def LAGTIME_MEANS_PATH(self):
        return os.path.join(self.LAGTIMES_PATH, 'lagtime_means')

    @property
    def THT_PATH(self):
        return os.path.join(abspath_root, 'data', 'tht_data')

    @property
    def ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES(self):
        return 1 / 8

    @property
    def STANDARD_ASYN_END_THT_VALUE(self):
        return 250


# Alternative format for constants using slots:
class SlotsLagTimeCalc:
    __slots__ = ()
    STARTING_THT_VALUE = 4.0
    SQUARE_OF_STARTING_VALUE = np.square(STARTING_THT_VALUE)  # e.g. 16.0
    DOUBLE_STARTING_VALUE = STARTING_THT_VALUE * 2  # e.g. 8.0
    _1_5_STARTING_VALUE = STARTING_THT_VALUE * 1.5  # e.g. 6.0
    MIN_NUM_OF_LAGTIMES_NEEDED = 3
    STR_TIME_H = 'Time (h)'
    ALL_THT_DATA_CSV_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.csv')
    LAGTIMES_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes')
    LAGTIME_MEANS_PATH = os.path.join(LAGTIMES_PATH, 'lagtime_means')
    THT_PATH = os.path.join(abspath_root, 'data', 'tht_data')
    ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES = 1 / 8
    STANDARD_ASYN_END_THT_VALUE = 250


# Alternative format for constants using dataclass:
@dataclass(frozen=True)
class DataClassLagTimeCalc:
    STARTING_THT_VALUE = 4.0
    SQUARE_OF_STARTING_VALUE = np.square(STARTING_THT_VALUE)  # e.g. 16.0
    DOUBLE_STARTING_VALUE = STARTING_THT_VALUE * 2  # e.g. 8.0
    _1_5_STARTING_VALUE = STARTING_THT_VALUE * 1.5  # e.g. 6.0
    MIN_NUM_OF_LAGTIMES_NEEDED = 3
    STR_TIME_H = 'Time (h)'
    ALL_THT_DATA_CSV_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.csv')
    LAGTIMES_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes')
    LAGTIME_MEANS_PATH = os.path.join(LAGTIMES_PATH, 'lagtime_means')
    THT_PATH = os.path.join(abspath_root, 'data', 'tht_data')
    ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES = 1 / 8
    STANDARD_ASYN_END_THT_VALUE = 250
