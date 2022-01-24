import os

import numpy as np
import pandas as pd
from pandas import DataFrame as pDF
from root_path import abspath_root

# abs_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
rel_path_prot_seq = os.path.join('data', 'protein_sequences')
abs_path_prot_seq = os.path.join(abspath_root, rel_path_prot_seq)


def read_protein_sequence_txt(txtfile: str) -> str:
    """
    Read protein sequence from given text file.
    :param txtfile: Name of text file, including txt extension or with no extension
    :return:
    """
    if txtfile[-4:] != '.txt':
        txtfile += '.txt'
    with open(os.path.join(abs_path_prot_seq, txtfile)) as f:
        aa_sequence = f.read()
    return aa_sequence


def read_protein_sequences_csv(csv=None) -> pDF:
    """
    Read protein sequences from given csv file or read `protein_sequences.csv` if no file name is given.
    Expecting format to be:
    'AC', 'fragment', 'name', 'amino_acids'
    'P37840', 'nan', 'SYUA_HUMAN', 'MDVFMKGL...'
    Note: th1e arguments `skipinitialspace=True` and `quotechar="'"` are required because the csv file contains
    single quotes, which otherwise would be included as part of the string.
    :param csv: csv filename with extension, none by default.
    :return: Tabulated list of sequences with corresponding column names.
    """
    if csv is None:
        csv = 'protein_sequences.csv'
    return pd.read_csv(os.path.join(abs_path_prot_seq, csv), skipinitialspace=True, quotechar="'")


def is_invalid_protein_sequence(aa_props: pDF, sequence: str) -> bool:
    for aa in sequence:
        if aa.upper() not in list(aa_props['aa']):
            print(f'invalid character is {aa}')
            return True
        else:
            continue
    return False


if __name__ == '__main__':
    seq = read_protein_sequence_txt('SYUA_HUMAN.txt')
    print(f'type(sequence): {type(seq)}')
