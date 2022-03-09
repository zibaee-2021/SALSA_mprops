import os
import pandas as pd
from pandas import DataFrame as pDF
from data.protein_sequences import read_seqs
from src.mutation import mutator
from root_path import abspath_root
import math


def _get_ln_lags() -> pDF:
    """
    Read all lag times from 4-degree polynomial linear regressions for synucleins.
    Generate the amino acid sequences.
    Compute normalised values of the 4 mean properties.
    :return: Synucleins (index) mapped to lag time means and natural log of lag times.
    ['lag_time_means', 'ln_lags']
    """
    syns_lags = pd.read_csv(os.path.join(abspath_root, 'data', 'tht_data', 'lag_time_degree_4.csv'), index_col=[0])
    syns_lags['ln_lags'] = syns_lags.apply(lambda row: math.log(row['lag_time_means']), axis=1)
    return syns_lags


def _build_syn_sequences(syns_lags: pDF) -> dict:
    """
    Generate the amino acid sequences of the given synucleins. (the input is expected ton include columns of lag
    time data but this is not used in this function).
    :param syns_lags: Synucleins (inclydes lag time data).
    :return: Synucleins (index) mapped to their sequences.
    ['lag_time_means', 'ln_lags', 'seqs']
    """
    syn_names = list(syns_lags.index)
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    gsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_HUMAN')['SYUG_HUMAN']
    fr_asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_FUGU')['SYUA_FUGU']
    fr_bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_FUGU')['SYUB_FUGU']
    fr_gsyn1 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG1_FUGU')['SYUG1_FUGU']
    fr_gsyn2 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG2_FUGU')['SYUG2_FUGU']
    b5v = mutator.mutate(prot_seq=bsyn, pos_aa={11: 'V', 19: 'V', 63: 'V', 78: 'V', 102: 'V'})
    assert (len(b5v) == 134)
    syn_seqs = {'asyn': asyn, 'gsyn': gsyn, 'fr_asyn': fr_asyn, 'fr_gsyn1': fr_gsyn1,
                'fr_gsyn2': fr_gsyn2, 'b5V': b5v}
    for syn_name in syn_names:
        syn_seqs_keys = list(syn_seqs)
        if syn_name not in syn_seqs_keys:
            if syn_name in ['a11_140', 'a21_140', 'a31_140', 'a41_140', 'a51_140', 'a61_140', 'a71_140', 'a1_75',
                            'a1_80', 'b1_73', 'a68_71Del', 'a71_72Del', 'a71_72Del', 'a71_74Del', 'a71_78Del',
                            'a71_81Del', 'a73_83Del', 'a74_84Del']:
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
            elif syn_name == 'a45V46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'a45V46V71E72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V', 71: 'E', 72: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'a71E72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={71: 'E', 72: 'E'})
                assert (len(syn_seqs[syn_name]) == 140)
            elif syn_name == 'bR45V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={45: 'V'})
                assert (len(syn_seqs[syn_name]) == 134)
            elif syn_name == 'b45V46V':
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
            else:
                print(f'{syn_name} is not recognised.. there must be a typo in one of the conditions or I need to add '
                      f'this one')
    return syn_seqs


def get_ln_lags_and_build_seqs() -> pDF:
    """
    Read all lag times of synucleins via polynomial linear regressions. Generate the amino acid sequences.
    :return: Synucleins, (index) mapped to lag time means, natural log of lag times and the amino acid sequences.
    ['lag_time_means', 'ln_lags', 'seqs']
    """
    syns_lnlags = _get_ln_lags()
    syn_seqs_dict = _build_syn_sequences(syns_lnlags)
    syn_seqs = pDF.from_dict(syn_seqs_dict, orient='index', columns=['seqs'])
    syns_lnlags_seqs = syns_lnlags.join(syn_seqs)
    return syns_lnlags_seqs


if __name__ == '__main__':
    get_ln_lags_and_build_seqs()