from typing import Tuple, NamedTuple
from src.mean_properties import mbp, mh, mnc_mtc
from collections import namedtuple
from enum import Enum
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from src.salsa import execute
from src.salsa.Options import DefaultBSC, Props
from data.protein_sequences import read_seqs
from src.mutation import mutate
"""
The derivation of the coefficients for each of the 4 mean properties (mean beta-sheet propensity, mean hydrophilicity, 
mean absolute net charge and mean total charge) is based on the kinetics of fibrillogenesis for 32 (although .. it 
may be more.. I don't remember why I didn't include the 45v46v71e72e mutants)
recombinant 
synuclein constructs measured in vitro. The calculations are as described in Zibaee et al 2010 JBC (and in the 
docstrings of the relevant functions below). 
The combination of the 4 weighted properties into one is referred to as `mprops`. 
The combination of normalised mprops and normalised SALSA b-strand contiguity integrals into one calculation is just 
referred to as the `combined algorithm`. The use of the combined algorithm produces a fit to the lag times with an 
Rsquared ("coefficient of determination") of ... (Rsquared of 1.0 is a perfect fit).

It is noted here too though that another ... synuclein constructs were assayed and found to not assemble within the 
duration of the experiment of 96 hours. The predicted lag times for these is also calculated here and for all of the 
constructs the values range from 100 hours to .. hours.  

Given the nature of these calculations and the stochasticity of the process of fibrillogenesis, values are likely to be 
specific, not only for natively unfolded proteins, or even for synucleins only but for the specific conditions of the 
experiments as well. Hence I specify them here alongside the code and mathematical derivations: 

Lag times of fibrillogensis of synuclein constructs was based on experiments under the following conditions: 
- Protein concentration at 400 microMolars in 30 mM MOPs buffer pH 7.2 and 10 microMolar ThT. 
- Incubation at 37degC, shaken at 450 rpm. 
- 10 microlitre samples collect at 5 subsequent time points 4h, 8h, 24h, 48h, 96h. All measurements were performed on 
each sample immediately after collection.

Lag times were measured from each experiment, as time taken for ThT values to reach the square of their zero
time emission readings at 480nm. The final value is the mean of the lag times for each constructs, rounded to the 
nearest 0.25 hours.
"""

# Note: although namedtuple takes up twice the number of lines of code as a dict to create, this one uses about 300
# bytes, while the dict equivalent uses about 1200 bytes. Hence namedtuple uses about 1/4 the memory of dict.
# while the dict equivalent. An enum cannot be used as it does not allow duplicate values (i.e. it will not store
# two syns that have the same lag times).
lagtimes = namedtuple('lagtimes', 'asyn asyn_11_140, asyn_21_140, asyn_31_140, asyn_41_140, asyn_51_140, asyn_61_140, '
                                  'asyn_1_80, asyn_del68_71, asyn_del71_72, asyn_del71_74, asyn_del71_78, '
                                  'asyn_del71_81, asyn_del73_83, bsyn_asyn_1, bsyn_asyn_12, bsyn_5V, bsyn_5V2Q, '
                                  'bsyn_5V4Q, bsyn_5V6Q, bsyn_5V8Q, asyn_T72V, asyn_V71E_T72E, asyn_A30P, asyn_E46K, '
                                  'asyn_A53T, asyn_S87E, asyn_S129E, bsyn_1_73, fr_asyn, fr_gsyn1, fr_gsyn2')

lagtimes = lagtimes(asyn=5.5, asyn_11_140=27.5, asyn_21_140=10.75, asyn_31_140=27.5, asyn_41_140=54.0,
                    asyn_51_140=86.5, asyn_61_140=96.5, asyn_1_80=4.75, asyn_del68_71=41.5, asyn_del71_72=8.25,
                    asyn_del71_74=9.5, asyn_del71_78=52.75, asyn_del71_81=44.25, asyn_del73_83=19.75,
                    bsyn_asyn_1=81.25, bsyn_asyn_12=25.0, bsyn_5V=81.5, bsyn_5V2Q=20.25, bsyn_5V4Q=20.0,
                    bsyn_5V6Q=13.5, bsyn_5V8Q=6.75, asyn_T72V=1.0, asyn_V71E_T72E=84.0, asyn_A30P=6.25,
                    asyn_E46K=4.5, asyn_A53T=4.25, asyn_S87E=17.0, asyn_S129E=5.75, bsyn_1_73=8.0, fr_asyn=0.75,
                    fr_gsyn1=0.75, fr_gsyn2=0.25)


# These are the original coefficients from Zibaee et al 2010 JBC.
class CoeffOrig(Enum):
    NMBP = -0.0741
    NMH = 0.0662
    NMNC = 0.0629
    NMTC = 0.0601


# These are the coefficients calculated here, using data from a few extra constructs and using scikit-learn to
# perform linear regression and determine coefficients for the 4 mean properties.
class Coefficient(Enum):
    NMBP = 0
    NMH = 0
    NMNC = 0
    NMTC = 0


def _compute_mprops(_4norm_props: NamedTuple) -> float:
    return (Coefficient.NMBP.value * _4norm_props.nmbp) + \
             (Coefficient.NMH * _4norm_props.mh) + \
             (Coefficient.NMNC * _4norm_props.nmnc) + \
             (Coefficient.NMTC * _4norm_props.nmtc)


def _build_syn_sequences(syn_names: list) -> dict:
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    # fr_asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_FUGU')['SYUA_FUGU']
    # fr_gsyn1 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_FUGU')['SYUG_FUGU']
    # fr_gsyn2 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_FUGU')['SYUG_FUGU']
    fr_asyn = asyn
    fr_gsyn1 = asyn
    fr_gsyn2 = asyn
    bsyn_5V = mutate.mutate_protein(protein_seq=bsyn, pos_aa={80: 'V', 81: 'V', 82: 'V',  83: 'V', 84: 'V'})
    syn_seqs = {'asyn': asyn, 'fr_asyn': fr_asyn, 'fr_gsyn1': fr_gsyn1, 'fr_gsyn2': fr_gsyn2, 'bsyn_5V': bsyn_5V}
    for syn_name in syn_names:
        syn_seqs_keys = list(syn_seqs)
        if syn_name not in syn_seqs_keys:
            if syn_name == 'asyn_11_140':
                syn_seqs[syn_name] = asyn[10:]
            elif syn_name == 'asyn_21_140':
                syn_seqs[syn_name] = asyn[20:]
            elif syn_name == 'asyn_31_140':
                syn_seqs[syn_name] = asyn[30:]
            elif syn_name == 'asyn_41_140':
                syn_seqs[syn_name] = asyn[40:]
            elif syn_name == 'asyn_51_140':
                syn_seqs[syn_name] = asyn[50:]
            elif syn_name == 'asyn_61_140':
                syn_seqs[syn_name] = asyn[60:]
            elif syn_name == 'asyn_1_80':
                syn_seqs[syn_name] = asyn[:80]
            elif syn_name == 'asyn_del68_71':
                syn_seqs[syn_name] = asyn[:68] + asyn[70:]
            elif syn_name == 'asyn_del71_72':
                syn_seqs[syn_name] = asyn[:71] + asyn[71:]
            elif syn_name == 'asyn_del71_74':
                syn_seqs[syn_name] = asyn[:71] + asyn[73:]
            elif syn_name == 'asyn_del71_78':
                syn_seqs[syn_name] = asyn[:71] + asyn[77:]
            elif syn_name == 'asyn_del71_81':
                syn_seqs[syn_name] = asyn[:71] + asyn[80:]
            elif syn_name == 'asyn_del73_83':
                syn_seqs[syn_name] = asyn[:73] + asyn[82:]
            elif syn_name == 'bsyn_asyn_1': # bsyn1-72, asyn73-83, bsyn73-134??
                syn_seqs[syn_name] = bsyn[:72] + asyn[72:83] + bsyn[72:]
            elif syn_name == 'bsyn_asyn_12':
                syn_seqs[syn_name] = asyn[:96] + bsyn[85:]
            elif syn_name == 'bsyn_5V':
                syn_seqs[syn_name] = bsyn_5V
            elif syn_name == 'bsyn_5V2Q':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=bsyn_5V, pos_aa={80: 'Q', 81: 'Q'})
            elif syn_name == 'bsyn_5V4Q':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=bsyn_5V, pos_aa={80: 'Q', 81: 'Q', 82: 'Q', 84: 'Q'})
            elif syn_name == 'bsyn_5V6Q':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=bsyn_5V, pos_aa={80: 'Q', 81: 'Q', 82: 'Q',
                                                                                       84: 'Q', 85: 'Q', 86: 'Q'})
            elif syn_name == 'bsyn_5V8Q':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=bsyn_5V, pos_aa={80: 'Q', 81: 'Q', 82: 'Q',
                                                                                       84: 'Q', 85: 'Q', 86: 'Q',
                                                                                       87: 'Q', 88: 'Q'})
            elif syn_name == 'asyn_45V46V':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={45: 'V', 46: 'V'})
            elif syn_name == 'asyn_45V46V71E72E':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={45: 'V', 46: 'V', 71: 'E', 72: 'E'})
            # elif syn_name == 'asyn71E72E': ?? Did it assemble within 96 hours??
            #     _32_syns_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={71: 'E', 72: 'E'})
            elif syn_name == 'asyn_T72V':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={72: 'V'})
            elif syn_name == 'asyn_V71E_T72E':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={71: 'E', 72: 'E'})
            elif syn_name == 'asyn_A30P':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={30: 'P'})
            elif syn_name == 'asyn_E46K':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={46: 'K'})
            elif syn_name == 'asyn_A53T':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={53: 'T'})
            elif syn_name == 'asyn_S87E':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={87: 'E'})
            elif syn_name == 'asyn_S129E':
                syn_seqs[syn_name] = mutate.mutate_protein(protein_seq=asyn, pos_aa={129: 'E'})
            elif syn_name == 'bsyn_1_73':
                syn_seqs[syn_name] = bsyn[:73]
            elif syn_name == 'fr_asyn':
                syn_seqs[syn_name] = fr_asyn
            elif syn_name == 'fr_gsyn1':
                syn_seqs[syn_name] = fr_gsyn1
            elif syn_name == 'fr_gsyn2':
                syn_seqs[syn_name] = fr_gsyn2
            else:
                print(f'{syn_name} is not recognised.. there must be a typo in one of the conditions or I need to add '
                      f'this one')
    return syn_seqs


def _normalise_mbp_mh_mnc_mtc(mbp: float, mh: float, mnc: float, mtc: float) -> NamedTuple:
    """
    Based on calculations from Zibaee et al 2007 JBC, each of the 4 values were normalised to spread to an approx
    range of 0-100. The calculations were based on the lowest and highest of each of these scores of the sequences of
    the 32 Synuclein constructs included. They were included if they began to assemble into filaments in vitro
    within 96 hours if the
    experiments had been repeated 3 or more times (from different recombinant protein preparations).
    :param mbp: Non-normalised mean beta-sheet propensity.
    :param mh: Non-normalised mean hydrophobicity.
    :param mnc: Non-normalised mean absolute net charge.
    :param mtc: Non-normalised mean total charge.
    :return: Normalised mean beta-sheet propensity, mean hydrophobicity, mean absolute net charge and mean total charge.
    """
    nmbp = ((mbp * 1000) - 800) / 3
    nmh = ((mh * -100) - 500) / 1.6
    nmnc = abs(mnc) * 500
    nmtc = ((mtc * 1000) - 200) / 1.2
    _4norm_props = namedtuple('_4norm_props', 'nmbp, nmh, nmnc, nmtc')
    return _4norm_props(nmbp=nmbp, nmh=nmh, nmnc=nmnc, nmtc=nmtc)


def calculate_weights_for_each_4_props():
    """
    The coefficients of each of following 4 plots were used for combining the 4 properties into one equation.
    The 4 plots were of nmbp, nmh, nmnc, nmtc against natural log of the lag-times data of 32
    recombinant synuclein constructs.

    The lag-times were the time taken for the ThT value to become the squared of its value at time = 0. The 32
    synuclein constructs that were included satisfied the following two requirements:
    1. They assembled with a detectable lag-time within 96 hours;
    2. They had been assayed ≥ 3 times (each from a different protein preparation batch).
    :return:
    """
    syns_ln_lagtimes = {}
    for syn, lagtime in zip(lagtimes._fields, lagtimes):
        syns_ln_lagtimes[syn] = math.log(lagtime)

    syn_seqs = _build_syn_sequences(list(lagtimes._fields))
    x_nmbp, x_nmh, x_nmnc, x_nmtc, y = np.zeros(len(syn_seqs)), np.zeros(len(syn_seqs)), np.zeros(len(syn_seqs)),\
                                   np.zeros(len(syn_seqs)), np.zeros(len(syn_seqs))

    for i, (syn_name, seq), in enumerate(syn_seqs.items()):
        x_nmbp[i], x_nmh[i], x_nmnc[i], x_nmtc[i] = _normalise_mbp_mh_mnc_mtc(mbp=mbp.compute_mean_beta_sheet_prop(seq),
                                                                              mh=mh.compute_mean_hydrophilicity(seq),
                                                                              mnc=mnc_mtc.compute_mean_net_charge(seq),
                                                                              mtc=mnc_mtc.compute_mean_total_charge(seq))
        y[i] = syns_ln_lagtimes[syn_name]

    model = LinearRegression()

    x_nmbp, x_nmh, x_nmnc, x_nmtc = x_nmbp.reshape((-1, 1)), x_nmh.reshape((-1, 1)), x_nmnc.reshape((-1, 1)), \
                                    x_nmtc.reshape((-1, 1))

    model.fit(x_nmbp, y)
    mbp_rsq = model.score(x_nmbp, y)
    mbp_cf = float(model.coef_)

    model.fit(x_nmh, y)
    mh_rsq = model.score(x_nmh, y)
    mh_cf = float(model.coef_)

    model.fit(x_nmnc, y)
    mnc_rsq = model.score(x_nmnc, y)
    mnc_cf = float(model.coef_)

    model.fit(x_nmtc, y)
    mtc_rsq = model.score(x_nmtc, y)
    mtc_cf = float(model.coef_)

    _4coefs = namedtuple('_4coefs', 'mbp_cf mh_cf mnc_cf mtc_cf')
    _4rsq = namedtuple('_4rsq', 'mbp_rsq mh_rsq mnc_rsq mtc_rsq')
    _4coefs = _4coefs(mbp_cf=mbp_cf, mh_cf=mh_cf, mnc_cf=mnc_cf, mtc_cf=mtc_cf)
    _4rsq = _4rsq(mbp_rsq=mbp_rsq, mh_rsq=mh_rsq, mnc_rsq=mnc_rsq, mtc_rsq=mtc_rsq)
    return _4coefs, _4rsq


def _normalise_mprops(mprops):
    return 0


def compute_normalised_mprops(syn_seqs: NamedTuple):
    """

    :param syn_seqs: Synuclein name and its amino acid sequence
    :return:
    """
    _4coefs, _4rsq = calculate_weights_for_each_4_props() # only needs to be done once
    syn_nmprops = {}

    for syn_name, seq in zip(syn_seqs, syn_seqs._fields):
        _4norm_props = _normalise_mbp_mh_mnc_mtc(mbp=mbp.compute_mean_beta_sheet_prop(seq),
                                                 mh=mh.compute_mean_hydrophilicity(seq),
                                                 mnc=mnc_mtc.compute_mean_net_charge(seq),
                                                 mtc=mnc_mtc.compute_mean_total_charge(seq))
        mprops = _compute_mprops(_4norm_props)
        nmprops = _normalise_mprops(mprops)
        syn_nmprops[syn_name] = nmprops
    return syn_nmprops


def _normalise_bsc_integral(bsc_integral: float):
    return 0


def _compute_salsa_bsc_integrals(seq):
    scored_windows_all = execute.compute(sequence=seq, _property=Props.bSC.value, params=DefaultBSC.all_params.value)
    summed_scores = execute.sum_scores_for_plot(scored_windows_all)
    return execute.integrate_salsa_plot({'seq': summed_scores})['seq']


def compute_normalised_salsa_bsc_integrals(seq: str):
    bsc_integral = _compute_salsa_bsc_integrals(seq)
    return _normalise_bsc_integral(bsc_integral)


def predict_lag_time_with_combined_algo(syn_seqs):
    nmprops = compute_normalised_mprops(syn_seqs)
    nbsc_integral = compute_normalised_salsa_bsc_integrals(seq)
    # calculate combined_algo TODO



from test.profiler_ import basics
if __name__ == '__main__':
    calculate_weights_for_each_4_props()

    # print(f' dict {basics.get_size_of(lagtimes)}')
    # print(f'namedtuple {basics.get_size_of(lagtimes)}')


# SynsProps = namedtuple():
#     asyn = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_11_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_21_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_31_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_41_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_51_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_61_140 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_1_80 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del68_71 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del71_72 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del71_74 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del71_78 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del71_81 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_del73_83 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_asyn_1 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_asyn_12 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_5V = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_5V2Q = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_5V4Q = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_5V6Q = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_5V8Q = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_T72V = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_V71E_T72E = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_A30P = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_E46K = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_A53T = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_S87E = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     asyn_S129E = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     bsyn_1_73 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     fr_asyn = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     fr_gsyn1 = {'mbp': , 'mh': , 'mnc': , 'mtc': }
#     fr_gsyn2 = {'mbp': , 'mh': , 'mnc': , 'mtc': }