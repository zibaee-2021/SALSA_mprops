import os
from enum import Enum, unique
from typing import Tuple
from root_path import abspath_root
# from data.aa_properties import aa_props
import numpy as np
import pandas as pd
from pandas import DataFrame as pDF
from matplotlib import pyplot as plt

rel_path_aa_props = os.path.join('data', 'aa_properties')
abs_path_aa_props = os.path.join(abspath_root, rel_path_aa_props)


def read_aa_props_csv() -> pDF:
    return pd.read_csv(os.path.join(abs_path_aa_props, 'aa_props.csv'))


def convert_alpha_beta_turn_pdf_to_dicts(aa_props: pDF) -> Tuple:
    p_alpha = dict(zip(aa_props['aa'], aa_props['Palpha']))
    p_beta = dict(zip(aa_props['aa'], aa_props['Pbeta']))
    p_turn = dict(zip(aa_props['aa'], aa_props['Pturn']))
    return p_alpha, p_beta, p_turn