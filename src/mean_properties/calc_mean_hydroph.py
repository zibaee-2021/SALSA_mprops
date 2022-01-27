from enum import Enum
import numpy as np


def compute_mean_hydrophilicity(sequence: str):
    # hydrophilicity_sum = 0
    # for aa in sequence:
    #     hydrophilicity_sum += RWS.HYDROPHILICTY.value[aa]
    # return hydrophilicity_sum / len(sequence)
    return np.mean([RWS.HYDROPHILICTY.value[aa] for aa in sequence])


# Radzicka_Wolfenden_Scale
class RWS(Enum):

    HYDROPHILICTY = {'A': -0.45, 'C': -3.63, 'D': -13.34, 'E': -12.63, 'F': -3.15,
                     'G': 0, 'H': -12.66, 'I': -0.24, 'K': -11.91, 'L': -0.11,
                     'M': -3.87, 'N': -12.07, 'P': 0.1, 'Q': -11.77, 'R': -22.31,
                     'S': -7.45, 'T': -7.27, 'V': -0.40, 'W': -8.27, 'Y': -8.50}