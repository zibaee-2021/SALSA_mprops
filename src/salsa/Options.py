from enum import Enum, unique


@unique
class Props(Enum):

    bSC = 'beta_strand_contiguity'
    mHA = 'mean_helical_amphipathicity'
    LSC = 'low-sequence-complexity'


@unique
class DefaultBSC(Enum):
    window_len_min = 4
    window_len_max = 20
    top_scoring_windows_num = 400
    threshold = 1.2


@unique
class DefaultMHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8


@unique
class DefaultLSC(Enum):
    window_len_min = 4
    window_len_max = 20
    top_scoring_windows_num = 10000
    threshold = 1