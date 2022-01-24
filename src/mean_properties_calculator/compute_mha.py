import numpy as np
from enum import Enum
from data.aa_properties import read_props


def compute_mean_helical_amphipathicity(sequence: str, periodicity=100):
    """
    Compute mean helical amphipathicity by calculating the mean of the hydrophobic moment for the given protein
    sequence according to the given periodicity, as described by:

          Ma-HA = 1/N x Sigma[h x cosine(n x theta)]

    where:
        - N is the length of the given protein sequence,
        - Sigma (is the sum) of the enclosed expression for each residue along the sequence.
        - n is the position of the residue from 0 to the end of the sequence (i.e. N - 1),
        - h is the hydrophobicity of the residue (at position n).
        - theta is the angle (in radians) to subtend each amino acid along the sequence from N-terminus to C-terminus,

    Other periodicities that correspond to possible secondary structures:
    polyproline II, 3-10-helix, polyglycine II: 120°; polyproline I: 108.1°; beta-sheet: 180°; pi-helix: 81.8°
    :param sequence: Protein sequence in 1-character format.
    :param periodicity: The angle to subtend each amino acid along the sequence, 100° by default (i.e. alpha-helix).
    :return:
    """
    h_kd = read_props.read_hydrophobicity_scale(HydrophobScalesColNames.KD.value)
    theta = np.radians(periodicity)
    return np.mean([h_kd[aa] * np.cos(i * theta) for i, aa in enumerate(sequence)])


class HydrophobScalesColNames(Enum):
    AA = 'aa'
    KD = 'Kyte-Doolittle'
    HW = 'Hopp-Woods'
    CORN = 'Cornette'
    EBERG = 'Eisenberg'
    ROSE = 'Rose'
    JANIN = 'Janin'
    ENGEL = 'Engelman_(GES)'
    RW = 'Radzicka-Wolfenden'


if __name__ == '__main__':

        bla = compute_mean_helical_amphipathicity('MDVFMKGLSKA', periodicity=98)
