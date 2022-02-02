import os
from enum import Enum
from typing import List, Tuple
import pandas as pd
import re
from pandas import DataFrame as pDF
from root_path import abspath_root

# abs_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
rel_path_prot_seq = os.path.join('data', 'protein_sequences')
abs_path_prot_seq = os.path.join(abspath_root, rel_path_prot_seq)


class TxtFiles(Enum):
    ASYN = 'SYUA_HUMAN.txt'
    TDP43 = 'TADBP_HUMAN_TDP43.txt'
    TAU2N4R = 'TAU_HUMAN_2N4R.txt'
    PRP = 'PRIO_HUMAN.txt'


def read_protein_sequence_txt(txtfile: str) -> str:
    """
    Read protein sequence from given text file.
    :param txtfile: Name of text file, with or without txt extension. E.g. `SYUA_HUMAN.txt`
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


def _has_expected_fragment_id_format(prot_id: str) -> bool:
    """
    For some proteins the Uniprot accession number and name do not uniquely identify the sequences of interest
    because they are fragments of a larger protein. A typical example is for Abeta peptides which do not have their
    own unique identifier. To address this I construct a unique identifier combining the id/name with the fragment
    position using the same format as can be found in the 2nd column (called `fragment`) of protein_sequences.csv.
    This is inside parentheses and the numbering is based on the main protein. e.g. The id for Abeta1-42 is the acc
    id or the name of APP and its fragment position like so: P05067(672-713) or A4_HUMAN(672-713).

    Detailed explanation of the regular expression: r'^\w+[(](\d+)[-](\d+)[)]$'

    The order in the expression must be strictly observed for a pattern match to be made.

        r indicates that the characters within the following section inside single quote marks should be treated as
        raw string. This means you don't need to escape special chars, e.g. you can use \w+ instead of \\w+.
        ^ and $ frame the expression. It means that a match should have nothing before ^ and nothing after $.
        \w matches alphanumeric or underscore. \w+ matches any size. Hence the input should start with any sized
        alphanumeric or underscore.
        [(] Literal match to open parenthesis ( and this does not need to be escaped by backslash because it is inside
        square brackets.
        (\d+) matches any numeric characters of any size and does not match to decimal points.
        [-] Literal match to -
        (\d+) As described above
        [)] Literal match to closed parenthesis )

    re.match(re.compile, input_string) returns None if the input_string does not match the regex pattern.

    :return: False if the given protein id is not that of a protein fragment id.
    """
    return re.match(re.compile(r'^\w+[(](\d+)[-](\d+)[)]$'), prot_id) is not None


def _separate_id_and_fragment(prot_frag_id: str) -> Tuple[str, str]:
    """
    Splits the protein id fragment string to protein id and the fragment string to match the column name format in
    the protein_sequence csv.
    Given 'P05067(672-713)', the function should return 'P05067', '672-713'.
    :return: Protein id (acc or name) and fragment position, in this order.
    """
    prot_id = re.search(r'^\w+', prot_frag_id).group()
    frag = re.search(r'[(](\d+)[-](\d+)[)]$', prot_frag_id).group()
    assert(frag.startswith('(') and frag.endswith(')'))
    return prot_id, frag[1:-1]


def get_sequences_by_uniprot_accession_nums_or_names(prot_ids) -> dict[str:str]:
    """
    Retrieve protein sequence(s) of interest corresponding to the given identifier(s) and/or name(s).
    Uniprot "accession number". NOTE: It is not a number. It has an alphanumeric format, such as 'Q16143'.
    Uniprot protein name is a mnemonic that incorporates species information, e.g. 'SYUA_HUMAN'.
    :param prot_ids: Uniprot accession number(s) and/or protein name(s). If it is a fragment of the protein,
    the expected format for this is `P05067(672-713)`. As a string or list of strings.
    :return: Protein sequences mapped to the given accession number or name.
    e.g. {'SYUA_HUMAN': 'MDVFMKGLS...', 'P10636-7': 'MAEPRQEF...', etc}
    """
    protein_ids_seqs = dict()
    all_prot_recs = read_protein_sequences_csv()
    if isinstance(prot_ids, str): prot_ids = [prot_ids]
    for prot_id in prot_ids:
        if _has_expected_fragment_id_format(prot_id):
            prot_id_sep, frag = _separate_id_and_fragment(prot_id)
            prot_record = all_prot_recs.loc[(all_prot_recs.name == prot_id_sep) & (all_prot_recs.fragment == frag)]
            if prot_record is not None and not prot_record.empty:
                protein_ids_seqs[prot_id] = prot_record.iloc[0]['sequence']
            else:
                prot_record = all_prot_recs.loc[(all_prot_recs.AC == prot_id_sep) & (all_prot_recs.fragment == frag)]
                if prot_record is not None and not prot_record.empty:
                    protein_ids_seqs[prot_id] = prot_record.iloc[0]['sequence']
        else:
            prot_record = all_prot_recs.loc[(all_prot_recs.name == prot_id) & (all_prot_recs.fragment.isna())]
            if prot_record is not None and not prot_record.empty:
                protein_ids_seqs[prot_id] = prot_record.iloc[0]['sequence']
            else:
                prot_record = all_prot_recs.loc[(all_prot_recs.AC == prot_id) & (all_prot_recs.fragment.isna())]
                if prot_record is not None and not prot_record.empty:
                    prot_record = prot_record[pd.isna(prot_record['fragment'])]
                    protein_ids_seqs[prot_id] = prot_record.iloc[0]['sequence']
    return protein_ids_seqs


if __name__ == '__main__':
    # seq = read_protein_sequence_txt('SYUA_HUMAN.txt')
    # print(f'type(sequence): {type(seq)}')

    prot_ids_not_in_csv_file = ['P05067']
    # actual = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_ids)
    actual = get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_ids_not_in_csv_file)
    print(actual)
