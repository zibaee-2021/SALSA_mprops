import numpy as np
from enum import Enum
from data.aa_properties import read_props


def compute_mean_electric_dipole_moment(sequence: str, periodicity: float):
    """
    Similar to the computation used for calculating the mean hydrophobic moment.
    Compute mean electric dipole moment by calculating the mean of the electrostatic moment for the given protein
    sequence according to the given periodicity, as described by:

          MEDM = 1/N x Sigma[q x cosine(n x theta)]

    where:
        - N is the length of the given protein sequence,
        - Sigma (is the sum) of the enclosed expression for each residue along the sequence.
        - n is the position of the residue from 0 to the end of the sequence (i.e. N - 1),
        - q is the electrostatic charge of the residue (at position n).
        - theta is the angle (in radians) to subtend each amino acid along the sequence from N-terminus to C-terminus,

    Other periodicities that correspond to possible secondary structures:
    polyproline II, 3-10-helix, polyglycine II: 120°; polyproline I: 108.1°; beta-sheet: 180°; pi-helix: 81.8°
    :param sequence: Protein sequence in 1-character format.
    :param periodicity: The angle to subtend each amino acid along the sequence, 100° by default (i.e. alpha-helix).
    :return:
    """
    all_props = read_props.read_aa_props_csv()
    aa_charge = read_props.convert_electrostatics_pdf_to_dicts(all_props)
    # print(aa_charge[sequence])
    theta = np.radians(periodicity)
    return np.mean([aa_charge[aa] * np.cos(i * theta) for i, aa in enumerate(sequence)])


if __name__ == '__main__':

        # bla = compute_mean_electric_dipole_moment('MDVFMKGLSKA', periodicity=98)
        bla = compute_mean_electric_dipole_moment('', periodicity=180)
        print(bla)
