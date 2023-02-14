from enum import Enum
import numpy as np

"""
There are 3 differences in value of Chou-Fasman numbers I use below, compared to those used in Sumner's original Java 
implementation of mean beta-sheet propensity. They are:
1. Asparagine alpha-helical preference number of 0.76 was used. Here I use 0.99. 
2. Asparagine reverse turn preference number of 1.24. Here I use 1.34.
3. Glutamic acid beta-strand preference number of 0.59 was used. Here I use 0.52.


Note the Chou-Fasman Preferences numbers are stored in this file in CFP Enum below. 
They are also stored in data/aa_properties/aa_props.csv. 
This duplication is temporary until I decide how to handle this. 
(The reason for including CFPs in this file is that the code is slow when reading from the csv).
"""


def compute_mean_beta_sheet_prop(sequence: str) -> float:
    """
    Calculate the mean beta-sheet propensity as described in Zibaee et al. 2007 Protein Science, Vol. 16,
    Issue 5: 906–918
    :param sequence: Protein sequence in 1-letter notation.
    :return: Mean beta-sheet propensity for the given protein.
    """
    beta_sum, alpha_sum, turn_sum = 0, 0, 0
    for c in sequence:
        if c == ' ':
            raise KeyError('There is white-space in this sequence. Check/correct sequence and try again.')
        else:
            beta_sum += CFP.BETASTRAND.value[c]
            alpha_sum += CFP.ALPHAHELIX.value[c]
            turn_sum += CFP.TURN.value[c]
    return beta_sum / (0.5 * (alpha_sum + turn_sum))


def massive_compute_mean_beat_sheet_prop(sequences: np.ndarray) -> float:
    """
    Perform `compute_mean_beta_sheet_prop` on large numbers of protein sequences.
    Consider constructing numpy matrices for potential performance enhancements.
    Consider potential for using asyncio/concurrency here..
    :param sequences: Protein sequences in 1-letter notation.
    :return: Mean beta-sheet propensities for the given proteins
    """


# Chou-Fasman Preferences
class CFP(Enum):
    BETASTRAND = {'A': 0.72, 'C': 1.40, 'D': 0.39, 'E': 0.52, 'F': 1.33, 'G': 0.58, 'H': 0.80, 'I': 1.67, 'K': 0.69,
                  'L': 1.22, 'M': 1.14, 'N': 0.48, 'P': 0.31, 'Q': 0.98, 'R': 0.84, 'S': 0.96, 'T': 1.17, 'V': 1.87,
                  'W': 1.35, 'Y': 1.45}
    ALPHAHELIX = {'A': 1.41, 'C': 0.66, 'D': 0.99, 'E': 1.59, 'F': 1.16, 'G': 0.43, 'H': 1.05, 'I': 1.09, 'K': 1.23,
                  'L': 1.34, 'M': 1.30, 'N': 0.99, 'P': 0.34, 'Q': 1.27, 'R': 1.21, 'S': 0.57, 'T': 0.76, 'V': 0.90,
                  'W': 1.02, 'Y': 0.74}
    TURN = {'A': 0.82, 'C': 0.54, 'D': 1.24, 'E': 1.01, 'F': 0.59, 'G': 1.77, 'H': 0.81, 'I': 0.47, 'K': 1.07,
            'L': 0.57, 'M': 0.52, 'N': 1.34, 'P': 1.32, 'Q': 0.84, 'R': 0.90, 'S': 1.22, 'T': 0.90, 'V': 0.41,
            'W': 0.65, 'Y': 0.76}


if __name__ == '__main__':
    # pass
    import pandas as pd
    from matplotlib import pyplot as plt
    all_20_AAs = list(CFP.BETASTRAND.value)
    aa_mbp, mbps, AAs = {}, [], []
    for aa in all_20_AAs:
        AAs.append(aa)
        mbps.append(compute_mean_beta_sheet_prop(sequence=aa))
    aa_mbp['aa'] = AAs
    aa_mbp['mbp'] = mbps

    pdf = pd.DataFrame.from_dict(data=aa_mbp)
    print(pdf.head())

    pdf.plot.bar(x='aa', y=['mbp'], rot=0, figsize=(12, 6),
                 ylabel='MEAN BETA-SHEET PROPENSITY',
                 title='Chou-Fasman MBP for individual amino acids'
                 )
    plt.show()

