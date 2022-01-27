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
class DefaultMaHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    periodicity = 100
# alpha-helix has 3.6 residues per turn, giving a periodicity of 100°.


@unique
class DefaultMpp2HA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    periodicity = 120
# 3-10-helix, Polyproline II helix and Polyglycine II helix all have 3 residues per turn, giving periodicity of 120°.


@unique
class DefaultMpp1HA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    periodicity = 108.1
# Polyproline I helix has 3.33 residues per turn, giving periodicity of 108.1°.


@unique
class DefaultMpiHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    periodicity = 81.8
# pi-helix has 4.4 residues per turn, giving a periodicity of 81.8°.


@unique
class DefaultMbHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    periodicity = 180
# beta-sheet has 2 residues per turn, giving a periodicity of 180°.

@unique
class DefaultLSC(Enum):
    window_len_min = 4
    window_len_max = 20
    top_scoring_windows_num = 1000
    threshold = 0.8