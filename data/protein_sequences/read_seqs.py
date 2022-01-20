import os
import pandas as pd
from pandas import DataFrame as pDF
from root_path import abspath_root

# abs_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
rel_path_prot_seq = os.path.join('data', 'protein_sequences')
abs_path_prot_seq = os.path.join(abspath_root, rel_path_prot_seq)


def read_protein_sequence_txt(txtfile: str) -> str:
    with open(os.path.join(abs_path_prot_seq, txtfile)) as f:
        aa_sequence = f.read()
    return aa_sequence


def read_protein_sequences_csv(csv: str) -> pDF:
    """
    Read protein sequences from SYUA_HUMAN.txt csv file.
    Expecting format to be:
    'AC', 'fragment', 'name', 'amino_acids'
    'P37840', 'nan', 'SYUA_HUMAN', 'MDVFMKGL...'
    etc
    :param csv: csv filename with extension
    :return: tabulated list of sequences with AC and protein names
    """
    return pd.read_csv(os.path.join(abs_path_prot_seq, csv))


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
