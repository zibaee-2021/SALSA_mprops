import os
from typing import Tuple
from root_path import abspath_root
import pandas as pd
from pandas import DataFrame as pDF

rel_path_aa_props = os.path.join('data', 'aa_properties')
abs_path_aa_props = os.path.join(abspath_root, rel_path_aa_props)


def read_all_hydrophobicity_scales_csv() -> pDF:
    return pd.read_csv(os.path.join(abs_path_aa_props, 'hydrophobicity_scales.csv'))


def read_hydrophobicity_scale(hydrophobicity_scale='Kyte-Doolittle') -> dict:
    """
    Read and map given hydrophobicity scale to amino acid as key and hydrophobicity number as value. (Note the scales
    includes the Radzicka & Woldenden hydrophilicity scale).
    :param hydrophobicity_scale: Hydrophobicity scale name, specified by column name.
    'Kyte-Doolittle' by default.
    :return:
    """
    all_scales = read_all_hydrophobicity_scales_csv()
    return {k: v for k, v in zip(all_scales['aa'], all_scales[hydrophobicity_scale])}


def read_aa_props_csv() -> pDF:
    return pd.read_csv(os.path.join(abs_path_aa_props, 'aa_props.csv'))


def convert_alpha_beta_turn_pdf_to_dicts(aa_props: pDF) -> Tuple:
    p_alpha = dict(zip(aa_props['aa'], aa_props['Palpha']))
    p_beta = dict(zip(aa_props['aa'], aa_props['Pbeta']))
    p_turn = dict(zip(aa_props['aa'], aa_props['Pturn']))
    return p_alpha, p_beta, p_turn


if __name__ == '__main__':
    print(read_hydrophobicity_scale())