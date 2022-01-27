from enum import Enum
import numpy as np


def compute_mean_total_charge(sequence: str):
    return np.mean(np.abs([Electrostatic.charge.value[aa] for aa in sequence]))


def compute_mean_net_charge(sequence: str):
    return np.sum([Electrostatic.charge.value[aa] for aa in sequence]) / len(sequence)


# Electrostatic charges
class Electrostatic(Enum):

    charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
              'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
              'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
              'S': 0, 'T': 1.17, 'V': 1.87, 'W': 0, 'Y': 0}


if __name__ == '__main__':
    print(compute_mean_total_charge('ACKR'))