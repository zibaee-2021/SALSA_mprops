import os
import pandas as pd
from pandas import DataFrame as pDF
from data.protein_sequences import read_seqs
from src.mutation import mutator
from root_path import abspath_root
import math
from src.utils import util_data

LAGTIME_MEANS_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes', 'lagtime_means')


def _get_ln_lags(csv_filename: str) -> pDF:
    """
    Read all 'lag-times' from 4-degree polynomial linear regressions for Synucleins.
    Generate the amino acid sequences.
    Compute normalised values of the 4 mean properties.
    :param csv_filename: Name of csv filename (including csv extension).
    :return: Table of 4 columns including Synucleins as index, 'lag-time' means and log of 'lag-times':
    [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    """
    pdf = pd.read_csv(os.path.join(LAGTIME_MEANS_PATH, csv_filename), index_col=[0])
    pdf['ln_lags'] = pdf.apply(lambda row: math.log(row['lagtime_means']), axis=1)
    return pdf


def _build_syn_sequences(pdf: pDF) -> dict:
    """
    Generate the amino acid sequences of the given Synucleins.
    (Note: Although the input is expected to include a column containing the 'lag-times', only the names are used in
    this function).
    Expected formats of Synuclein names are as follows:
    1. All human Synuclein names begin with the identity of Synuclein (alpha, beta, gamma) represented by a single
    lowercase letter (`a`, `b`, `g`).
    2. All hyphens replaced by underscores (e.g. `a1-80` replaced by `a1_80`).
    3. Deletion mutants have `del` suffixed to the stretch of residues that were deleted, e.g. `a68_71del`,
    with lowercase `d`.
    4. The `b5V` mutant series indicate the total number of mutations made (to Valine or Glycine or Glutamine), but
    not the positions, e.g. b5V4Q represents 5 Alanine to Valine substitutions, 4 Glutamate to Glutamine substitutions.
    5. All other substitutions are use the wild-type residue (uppercase), followed by its position, followed by the
    mutant residue (uppercase), e.g. `aK45VE46VV71ET72E`.
    6. Fugu sequences have `fr_` prefix. Mouse beta is `mus_bsyn`, chicken beta is `gallus_bsyn`.
    :param pdf: Table of 3 columns including Synucleins as index, 'lag-time' means and log of 'lag-times':
    [(index), 'lagtime_means', 'ln_lags']
    :return: Synucleins mapped to their sequences.
    """
    syn_names = list(pdf.index)
    util_data.check_syn_names(syn_names)
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    gsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_HUMAN')['SYUG_HUMAN']
    fr_asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_FUGU')['SYUA_FUGU']
    fr_bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_FUGU')['SYUB_FUGU']
    fr_gsyn1 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG1_FUGU')['SYUG1_FUGU']
    fr_gsyn2 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG2_FUGU')['SYUG2_FUGU']
    mus_bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_MOUSE')['SYUB_MOUSE']
    gallus_bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('Q9I9G9_CHICK')['Q9I9G9_CHICK']
    b5v = mutator.mutate(prot_seq=bsyn, pos_aa={11: 'V', 19: 'V', 63: 'V', 78: 'V', 102: 'V'})
    assert (len(b5v) == 134)
    syn_seqs = {'asyn': asyn, 'gsyn': gsyn, 'fr_asyn': fr_asyn, 'fr_gsyn1': fr_gsyn1,
                'fr_gsyn2': fr_gsyn2, 'mus_bsyn': mus_bsyn, 'gallus_bsyn': gallus_bsyn, 'b5V': b5v}
    for syn_name in syn_names:
        syn_seqs_keys = list(syn_seqs)
        if syn_name not in syn_seqs_keys:
            if syn_name in ['a11_140', 'a21_140', 'a31_140', 'a41_140', 'a51_140', 'a61_140', 'a71_140',
                            'a1_45', 'a1_50', 'a1_55', 'a1_60', 'a1_70', 'a1_75', 'a1_80', 'g1_80', 'b1_73',
                            'a68_71del', 'a71_72del', 'a71_74del', 'a71_76del', 'a71_78del', 'a71_81del', 'a71_82del',
                            'a73_83del', 'a74_84del', 'a73_82del']:
                syn_seqs[syn_name] = mutator.make_fragment(syn_name)
            elif syn_name == 'bsyn':
                syn_seqs[syn_name] = bsyn
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'ba1':  # bsyn1-72, asyn73-83, bsyn73-134. Hence this should be 145 aa long
                syn_seqs[syn_name] = bsyn[:72] + asyn[72:83] + bsyn[72:]
                assert (len(syn_seqs[syn_name]) == 145)
            elif syn_name == 'ba12':  # asyn1-96, bsyn86-134. Hence this should be 145 aa long.
                syn_seqs[syn_name] = asyn[:96] + bsyn[85:]
                assert (len(syn_seqs[syn_name]) == 145)
            elif syn_name == 'ga':  # gsyn1-72, asyn73-83, gsyn84-127. Hence this should be 127 aa long
                syn_seqs[syn_name] = gsyn[:72] + asyn[72:83] + gsyn[83:]
                assert(len(syn_seqs[syn_name]) == 127)
            elif syn_name == 'b5V':
                syn_seqs[syn_name] = b5v
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'b5V2Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5v, pos_aa={125: 'Q', 126: 'Q'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'b5V4Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5v, pos_aa={104: 'Q', 105: 'Q', 125: 'Q', 126: 'Q'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'b5V6Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5v, pos_aa={96: 'Q', 97: 'Q', 104: 'Q', 105: 'Q',
                                                                          125: 'Q', 126: 'Q'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'b5V8Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5v, pos_aa={87: 'Q', 88: 'Q', 96: 'Q', 97: 'Q',
                                                                          104: 'Q', 105: 'Q', 125: 'Q', 126: 'Q'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'aA30P':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={30: 'P'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aE46K':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={46: 'K'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aA53T':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={53: 'T'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aK45V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={46: 'V'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aK45VE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aK45VE46VV71ET72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V', 71: 'E', 72: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aV71ET72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={71: 'E', 72: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'bR45V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={45: 'V'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'bR45VE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={45: 'V', 46: 'V'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'bE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={46: 'V'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'aT72V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={72: 'V'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aS87E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={87: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'aS129E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={129: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'fr_asyn':
                syn_seqs[syn_name] = fr_asyn
                assert (len(syn_seqs[syn_name]) == 127)
            elif syn_name == 'fr_bsyn':
                syn_seqs[syn_name] = fr_bsyn
                assert (len(syn_seqs[syn_name]) == 117)
            elif syn_name == 'fr_gsyn1':
                syn_seqs[syn_name] = fr_gsyn1
                assert (len(syn_seqs[syn_name]) == 113)
            elif syn_name == 'fr_gsyn2':
                syn_seqs[syn_name] = fr_gsyn2
                assert (len(syn_seqs[syn_name]) == 124)
            elif syn_name == 'mus_bsyn':
                syn_seqs[syn_name] = mus_bsyn
                assert (len(syn_seqs[syn_name]) == 133)
            elif syn_name == 'gallus_bsyn':
                syn_seqs[syn_name] = gallus_bsyn
                assert (len(syn_seqs[syn_name]) == 133)
            else:
                print(f'{syn_name} is not recognised.. there must be a typo in one of the conditions or I need to add '
                      f'this one')
    return syn_seqs


def get_loglags_and_build_seqs(csv_filename: str) -> pDF:
    """
    Read file containing mean 'lag-times' of Synucleins, calculate natural log of the 'lag-times' and generate amino
    acid sequences from names of Synuclein constructs.
    :param csv_filename: Name of 'lag-time' means csv filename (including csv extension).
    :return: Table of 4 columns including Synucleins as index, 'lag-time' means, natural log of 'lag-times' and the
    amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    """
    pdf = _get_ln_lags(csv_filename)
    syn_seqs_dict = _build_syn_sequences(pdf)
    pdf_ = pDF.from_dict(syn_seqs_dict, orient='index', columns=['seqs'])
    pdf__ = pdf.join(pdf_)
    syns = list(pdf__.index)
    print(f"The following {len(syns)} Synucleins with 'lag-times' read in from csv file: {syns}")
    return pdf__


if __name__ == '__main__':
    get_loglags_and_build_seqs(csv_filename='lagtime_means_polynDegree_4_lagtimeEndvalue_8.csv')