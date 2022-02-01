from enum import Enum, unique


@unique
class Props(Enum):

    bSC = 'beta-strand contiguity'
    mHA = 'mean helical amphipathicity'
    LSC = 'low-sequence-complexity'
    mEDM = 'mean electric dipole moment'


@unique
class DefaultBSC(Enum):
    window_len_min = 4
    window_len_max = 20
    top_scoring_windows_num = 400
    threshold = 1.15
    abs_threshold = False
    periodicity = ''
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}


@unique
class DefaultMaHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    abs_threshold = True
    periodicity = 100
# alpha-helix has 3.6 residues per turn, giving a periodicity of 100°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}


@unique
class DefaultMpp2HA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.5
    abs_threshold = True
    periodicity = 120
# 3-10-helix, Polyproline II helix and Polyglycine II helix all have 3 residues per turn, giving periodicity of 120°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}


@unique
class DefaultMpp1HA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    abs_threshold = False
    periodicity = 108.1
# Polyproline I helix has 3.33 residues per turn, giving periodicity of 108.1°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}

@unique
class DefaultMpiHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    abs_threshold = False
    periodicity = 81.8
# pi-helix has 4.4 residues per turn, giving a periodicity of 81.8°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}

@unique
class DefaultMbHA(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.8
    abs_threshold = False
    periodicity = 180
# beta-sheet has 2 residues per turn, giving a periodicity of 180°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}


@unique
class DefaultMEDM(Enum):
    window_len_min = 11
    window_len_max = 33
    top_scoring_windows_num = 10000
    threshold = 0.05
    abs_threshold = True
    # periodicity = 100
    periodicity = 120
    # periodicity = 108.1
    # periodicity = 81.8
    # periodicity = 180
# alpha-helix has 3.6 residues per turn, giving a periodicity of 100°.
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}


@unique
class DefaultLSC(Enum):
    window_len_min = 2
    window_len_max = 40
    top_scoring_windows_num = 10000
    threshold = 0.8
    abs_threshold = False
    periodicity = None
    all_params = {'window_len_min': window_len_min,
                  'window_len_max': window_len_max,
                  'top_scoring_windows_num': top_scoring_windows_num,
                  'threshold': threshold,
                  'abs_threshold': abs_threshold,
                  'periodicity': periodicity}
