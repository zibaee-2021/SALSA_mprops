from enum import Enum
import numpy as np


def compute_mean_total_charge(sequence: str):
    """
    Calculate the mean total charge of the given protein sequence.
    :param sequence: Protein sequence in 1-letter notation.
    :return: Mean total charge of the given protein sequence.
    """
    return np.mean(np.abs([Electrostatic.charge.value[aa] for aa in sequence]))


def compute_mean_net_charge(sequence: str):
    """
    Calculate the absolute value of the mean net charge of the given protein sequence.
    :param sequence: Protein sequence in 1-letter notation.
    :return: Mean of the absolute net charge of the given protein sequence.
    """
    return abs(np.sum([Electrostatic.charge.value[aa] for aa in sequence]) / len(sequence))


# Electrostatic charges
class Electrostatic(Enum):

    charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
              'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
              'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
              'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}


if __name__ == '__main__':
    print(compute_mean_total_charge('ACKR'))