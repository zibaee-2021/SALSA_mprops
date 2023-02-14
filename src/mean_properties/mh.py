import numpy as np
from data.aa_properties import read_props
from data.aa_properties.read_props import Scales


def compute_mean_hydrophilicity(sequence: str):
    """
    Calculate the mean hydrophilicity of the given protein sequence, using Radzicka & Wolfenden hydrophilicity scale.
    (The R&W scale lacked a value for Proline. It was estimated at -3.8 based on comparison with Roseman hydrophobicity
    numbers.)
    :param sequence: Protein sequence in 1-letter notation.
    :return: Mean hydrophilicity of given protein.
    """
    hl_scale = read_props.read_hydrophobicity_scale(Scales.RW.value)
    return np.mean([hl_scale[aa] for aa in sequence])


def compute_mean_hydrophobicity(sequence: str, scale: str = None):
    """
    Calculate the mean hydrophobicity of the given protein sequence, using the given hydrophobicity scale,
    `Eisenberg` by default.
    Other available scales are `Kyte-Doolittle, Hopp-Woods, Cornette, Rose, Janin, and Engelman_(GES)`.
    :param sequence: Protein sequence in 1-letter notation.
    :param scale: Hydrophobicity scale to use for computing mean hydrophobicity.
    :return: Mean hydrophobicity of given protein.
    """
    if scale is None: scale = Scales.EBERG.value
    hb_scale = read_props.read_hydrophobicity_scale(scale)
    return np.mean([hb_scale[aa] for aa in sequence])


if __name__ == '__main__':
    print(compute_mean_hydrophilicity('VVDF'))
    from data.protein_sequences import read_seqs
    print(compute_mean_hydrophilicity(read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')))
