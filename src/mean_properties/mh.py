from enum import Enum
import numpy as np


def compute_mean_hydrophilicity(sequence: str):
    """
    Calculate the mean hydrophilicity of the given protein sequence, using Radzicka & Wolfenden hydrophilicity scale.
    (The R&W scale lacked a value for Proline. It was estimated at -3.8 based on comparison with Roseman hydrophobicity
    numbers.)
    :param sequence: Protein sequence in 1-letter notation.
    :return: Mean hydrophilicity of given protein.
    """
    return np.mean([RWS.HYDROPHILICTY.value[aa] for aa in sequence])


# Radzicka_Wolfenden_Scale
class RWS(Enum):

    HYDROPHILICTY = {'A': -0.45, 'C': -3.63, 'D': -13.34, 'E': -12.63, 'F': -3.15,
                     'G': 0, 'H': -12.66, 'I': -0.24, 'K': -11.91, 'L': -0.11,
                     'M': -3.87, 'N': -12.07, 'P': -3.8, 'Q': -11.77, 'R': -22.31,
                     'S': -7.45, 'T': -7.27, 'V': -0.40, 'W': -8.27, 'Y': -8.50}


if __name__ == '__main__':
    print(compute_mean_hydrophilicity('VVDF'))
