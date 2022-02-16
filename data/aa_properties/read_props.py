import os
from typing import Tuple
from enum import Enum
from root_path import abspath_root
import pandas as pd
from pandas import DataFrame as pDF

rel_path_aa_props = os.path.join('data', 'aa_properties')
abs_path_aa_props = os.path.join(abspath_root, rel_path_aa_props)
hydroph_scales_csv = 'hydrophobicity_scales.csv'

class Scales(Enum):
    KD = 'Kyte-Doolittle'
    HW = 'Hopp-Woods'
    CORN = 'Cornette'
    EBERG = 'Eisenberg'
    ROSE = 'Rose'
    JANIN = 'Janin'
    ENGEL = 'Engelman_(GES)'
    RW = 'Radzicka-Wolfenden'


def read_all_hydrophobicity_scales_csv() -> pDF:
    return pd.read_csv(os.path.join(abs_path_aa_props, hydroph_scales_csv))


def read_hydrophobicity_scale(hydrophobicity_scale: str = None) -> dict:
    """
    Read and map given hydrophobicity scale to amino acid as key and hydrophobicity number as value. (Note the scales
    includes the Radzicka & Woldenden hydrophilicity scale).
    :param hydrophobicity_scale: Hydrophobicity scale name, specified by column name.
    'Kyte-Doolittle' by default if no argument is passed.
    :return:
    """
    if hydrophobicity_scale is None:
        hydrophobicity_scale = Scales.KD.value
    df_col = pd.read_csv(hydroph_scales_csv, skipinitialspace=True, usecols=[hydrophobicity_scale, 'aa'])
    return dict(zip(df_col['aa'], df_col[hydrophobicity_scale]))


def read_aa_props_csv() -> pDF:
    return pd.read_csv(os.path.join(abs_path_aa_props, 'aa_props.csv'))


def convert_electrostatics_pdf_to_dicts(aa_props: pDF) -> dict[str: int]:
    charge = dict(zip(aa_props['aa'], aa_props['charge']))
    return charge


def convert_alpha_beta_turn_pdf_to_dicts(aa_props: pDF) -> Tuple[dict, dict, dict]:
    p_alpha = dict(zip(aa_props['aa'], aa_props['Palpha']))
    p_beta = dict(zip(aa_props['aa'], aa_props['Pbeta']))
    p_turn = dict(zip(aa_props['aa'], aa_props['Pturn']))
    return p_alpha, p_beta, p_turn


if __name__ == '__main__':

    print(read_all_hydrophobicity_scales_csv())
    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # #
    # # In order to estimate a value for Proline in the Radzicka-Wolfenden scale, I am
    # #     making a visualisation of the other scales, after separating and sorting them
    # # Move this to a notebook in sandbox
    # hyd_sc = read_all_hydrophobicity_scales_csv()
    # hyd_sc_kd = hyd_sc[['aa', 'Kyte-Doolittle']].sort_values(by='Kyte-Doolittle')
    # # ax = sns.barplot(x='aa', y='Kyte-Doolittle', data=hyd_sc_kd)
    # sns.catplot(data=hyd_sc_kd, x='aa', y='Kyte-Doolittle', kind='point')
    # plt.show()
    # hyd_sc_hw = hyd_sc[['aa', 'Hopp-Woods']].sort_values(by='Hopp-Woods')
    # hyd_sc_co = hyd_sc[['aa', 'Cornette']].sort_values(by='Cornette')
    # hyd_sc_ei = hyd_sc[['aa', 'Eisenberg']].sort_values(by='Eisenberg')
    # hyd_sc_ro = hyd_sc[['aa', 'Rose']].sort_values(by='Rose')
    # hyd_sc_ja = hyd_sc[['aa', 'Janin']].sort_values(by='Janin')
    # hyd_sc_en = hyd_sc[['aa', 'Engelman_(GES)']].sort_values(by='Engelman_(GES)')
    # hyd_sc_rw = hyd_sc[['aa', 'Radzicka-Wolfenden']].sort_values(by='Radzicka-Wolfenden')
