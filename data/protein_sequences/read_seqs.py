import os
import pandas as pd
from pandas import DataFrame as pDF
from root_path import abspath_root
from typing import List

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
    return pd.read_csv(os.path.join(abs_path_prot_seq, csv))


def is_invalid_protein_sequence(aa_props: pDF, sequence: str) -> bool:
    for aa in sequence:
        if aa.upper() not in list(aa_props['aa']):
            print(f'invalid character is {aa}')
            return True
        else:
            continue
    return False


def get_sequences_by_uniprot_accession_nums_or_names(accs=None, names=None,
                                                     frags_ids: dict[str:str] = None) -> dict[str:str]:
    """
    Retrieve protein sequence(s) of interest corresponding to the given identifier(s) and/or name(s).
    Uniprot "accession number". NOTE: It is not a number. It has an alphanumeric format, such as 'Q16143'.
    Uniprot protein name is a mnemonic that incorporates species information, e.g. 'SYUA_HUMAN'.
    :param accs: Uniprot accession number(s), as a string or list of strings. None by default.
    :param names: Uniprot protein name(s), as a string or list of strings. None by default.
    :param frags_ids: The fragment(s) of the protein(s) mapped to it.
    For example {'672-713':'P05067', '672-711': 'P05067'}.
    :return: Protein sequences mapped to the given accession number or name.
    e.g. {'SYUA_HUMAN': 'MDVFMKGLS...', 'P10636-7': 'MAEPRQEF...', etc}
    """
    protein_ids_sequences = dict()
    all_prot_recs = read_protein_sequences_csv()
    if accs is not None and accs != ['']:
        if isinstance(accs, str): accs = [accs]
        protein_ids_sequences = {acc: all_prot_recs.loc[(all_prot_recs.AC == acc)].iloc[0]['sequence'] for acc in accs}
    if names is not None and names != ['']:
        if isinstance(names, str): names = [names]
        for name in names:
            protein_ids_sequences[name] = all_prot_recs.loc[(all_prot_recs.name == name)].iloc[0]['sequence']
    if frags_ids is not None and frags_ids != {'': ''}:
        for fragment, id_name in frags_ids.items():
            prot_record = all_prot_recs.loc[(all_prot_recs.name == id_name) & (all_prot_recs.fragment == fragment)]
            if prot_record is not None and not prot_record.empty:
                protein_ids_sequences[id_name + '(' + fragment + ')'] = prot_record.iloc[0]['sequence']
            else:
                prot_record = all_prot_recs.loc[(all_prot_recs.AC == id_name) & (all_prot_recs.fragment == fragment)]
                if prot_record is not None and not prot_record.empty:
                    protein_ids_sequences[id_name + '(' + fragment + ')'] = prot_record.iloc[0]['sequence']
    return protein_ids_sequences


if __name__ == '__main__':
    seq = read_protein_sequence_txt('SYUA_HUMAN.txt')
    print(f'type(sequence): {type(seq)}')
