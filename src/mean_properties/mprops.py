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
from src.mutation import mutator
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

Another very important caveat is that for certain synuclein constructs for which lag time data was included, 
which had longer lag times, those proteins did not begin to assemble within the 96 hours limit. Hence, 
the data from these experiments was not included. As such their included data indicates these proteins to be more 
fibrillogenic than there were observed to be. 

I will see what result I get from replacing the absent data for those in which assembly was not observed with a dummy 
value of 200 hours.

I may also do this for the large number of other constructs for which assembly was not observed at all within the
100 hour limit.
"""

# Note: although namedtuple takes up twice the number of lines of code as a dict to create, this one uses about 300
# bytes, while the dict equivalent uses about 1200 bytes. Hence namedtuple uses about 1/4 the memory of dict.
# while the dict equivalent. An enum cannot be used as it does not allow duplicate values (i.e. it will not store
# two syns that have the same lag times).
# lagtimes = namedtuple('lagtimes', 'asyn asyn_11_140, asyn_21_140, asyn_31_140, asyn_41_140, asyn_51_140, asyn_61_140, '
#                                   'asyn_1_80, asyn_del68_71, asyn_del71_72, asyn_del71_74, asyn_del71_78, '
#                                   'asyn_del71_81, asyn_del73_83, bsyn_asyn_1, bsyn_asyn_12, bsyn_5V, bsyn_5V2Q, '
#                                   'bsyn_5V4Q, bsyn_5V6Q, bsyn_5V8Q, asyn_T72V, asyn_V71E_T72E, asyn_A30P, asyn_E46K, '
#                                   'asyn_A53T, asyn_S87E, asyn_S129E, bsyn_1_73, fr_asyn, fr_gsyn1, fr_gsyn2')

# orig_lagtimes = lagtimes(asyn=5.5, asyn_11_140=27.5, asyn_21_140=10.75, asyn_31_140=27.5, asyn_41_140=54.0,
#                     asyn_51_140=86.5, asyn_61_140=96.5, asyn_1_80=4.75, asyn_del68_71=41.5, asyn_del71_72=8.25,
#                     asyn_del71_74=9.5, asyn_del71_78=52.75, asyn_del71_81=44.25, asyn_del73_83=19.75,
#                     bsyn_asyn_1=81.25, bsyn_asyn_12=25.0, bsyn_5V=81.5, bsyn_5V2Q=20.25, bsyn_5V4Q=20.0,
#                     bsyn_5V6Q=13.5, bsyn_5V8Q=6.75, asyn_T72V=1.0, asyn_V71E_T72E=84.0, asyn_A30P=6.25,
#                     asyn_E46K=4.5, asyn_A53T=4.25, asyn_S87E=17.0, asyn_S129E=5.75, bsyn_1_73=8.0, fr_asyn=0.75,
#                     fr_gsyn1=0.75, fr_gsyn2=0.25)

# Note: You cannot use hyphens in namedtuples, so I replaced all hyphens with underscores.
lagtimes = namedtuple('lagtimes', 'asyn, aS87E, aS129E, a1_80, b1_73, gsyn, ga, a11_140, a21_140, ba1, a73_83Del, '
                                  'a31_140, a41_140, a51_140, a68_71Del, a74_84Del, a61_140, fr_asyn, b5V, fr_gsyn2, '
                                  'fr_gsyn1, b5V4Q, b5V6Q, b5V8Q, aE46K, aA30P, aA53T, ba12, a1_75, a71_72Del, aT72V, '
                                  'a71_74Del, a71E72E, a71_81Del, a45V46V71E72E, fr_bsyn, a45V46V, bE46V, b45V46V, '
                                  'aK45V, aE46V')

lagtimes_deg2 = lagtimes(asyn=11.3, aS87E=19.8, aS129E=8.1, a1_80=51.8, b1_73=14.6, gsyn=14.6, ga=14.4, a11_140=23.5,
                         a21_140=12.0, ba1=98.7, a73_83Del=20.4, a31_140=29.1, a41_140=37.7,  a51_140=58.4,
                         a68_71Del=43.7, a74_84Del=23.1, a61_140=37.8, fr_asyn=1.4, b5V=88.9, fr_gsyn2=1.4,
                         fr_gsyn1=1.6, b5V4Q=30.5, b5V6Q=17.2, b5V8Q=20.6, aE46K=4.2, aA30P=9.0, aA53T=7.8,
                         ba12=18.8, a1_75=7.4, a71_72Del=4.1, aT72V=2.4, a71_74Del=7.4, a71E72E=77.2, a71_81Del=43.0,
                         a45V46V71E72E=22.4, fr_bsyn=63.6, a45V46V=5.4, bE46V=39.3, b45V46V=37.5, aK45V=4.8, aE46V=3.0)

lagtimes_deg3 = lagtimes(asyn=7.2, aS87E=12.2, aS129E=5.3, a1_80=23.9, b1_73=9.2, gsyn=10.8, ga=10.2, a11_140=16.7,
                         a21_140=8.5, ba1=56.0, a73_83Del=12.1, a31_140=17.3, a41_140=24.0,  a51_140=33.9,
                         a68_71Del=28.1, a74_84Del=13.9, a61_140=18.3, fr_asyn=1.0, b5V=46.6, fr_gsyn2=1.0,
                         fr_gsyn1=1.0, b5V4Q=16.6, b5V6Q=11.2, b5V8Q=13.3, aE46K=3.3, aA30P=5.7, aA53T=4.7,
                         ba12=15.0, a1_75=4.0, a71_72Del=3.7, aT72V=1.5, a71_74Del=5.6, a71E72E=41.8, a71_81Del=27.0,
                         a45V46V71E72E=14.4, fr_bsyn=30.3, a45V46V=2.8, bE46V=21.2, b45V46V=21.1, aK45V=3.4, aE46V=1.8)

lagtimes_deg4 = lagtimes(asyn=16.1, aS87E=26.4, aS129E=12.0, a1_80=45.1, b1_73=20.3, gsyn=26.0, ga=24.0, a11_140=39.4,
                         a21_140=19.9, ba1=117.9, a73_83Del=26.0, a31_140=37.1, a41_140=53.9, a51_140=72.3,
                         a68_71Del=62.6, a74_84Del=30.0, a61_140=35.4, fr_asyn=2.2, b5V=94.5, fr_gsyn2=2.1,
                         fr_gsyn1=2.2, b5V4Q=34.0, b5V6Q=25.3, b5V8Q=29.8, aE46K=8.1, aA30P=12.7, aA53T=10.4,
                         ba12=37.1, a1_75=8.2, a71_72Del=9.8, aT72V=3.2, a71_74Del=13.8, a71E72E=85.0, a71_81Del=59.4,
                         a45V46V71E72E=32.2, fr_bsyn=58.2, a45V46V=5.7, bE46V=43.3, b45V46V=44.1, aK45V=7.8, aE46V=4.0)

lagtimes_deg5 = lagtimes(asyn=13.9, aS87E=22.3, aS129E=10.4, a1_80=35.1, b1_73=17.4, gsyn=23.4, ga=21.4, a11_140=35.0,
                         a21_140=17.7, ba1=98.0, a73_83Del=21.9, a31_140=31.1, a41_140=46.8, a51_140=60.6,
                         a68_71Del=53.6, a74_84Del=25.4, a61_140=28.0, fr_asyn=1.9, b5V=77.0, fr_gsyn2=1.9,
                         fr_gsyn1=1.9, b5V4Q=27.8, b5V6Q=22.0, b5V8Q=25.7, aE46K=7.4, aA30P=10.9, aA53T=8.9,
                         ba12=34.0, a1_75=6.7, a71_72Del=9.4, aT72V=2.7, a71_74Del=12.5, a71E72E=69.0, a71_81Del=50.6,
                         a45V46V71E72E=27.7, fr_bsyn=45.8, a45V46V=4.5, bE46V=35.3, b45V46V=36.4, aK45V=6.9, aE46V=3.3)

lagtimes_deg6 = lagtimes(asyn=17.9, aS87E=28.4, aS129E=13.4, a1_80=42.2, b1_73=22.2, gsyn=31.1, ga=28.3,
                         a11_140=45.9, a21_140=23.2, ba1=123.2, a73_83Del=27.8, a31_140=39.3, a41_140=60.5,
                         a51_140=76.8, a68_71Del=68.7, a74_84Del=32.3, a61_140=34.0, fr_asyn=2.5, b5V=95.5,
                         fr_gsyn2=2.4, fr_gsyn1=2.5, b5V4Q=34.6, b5V6Q=28.5, b5V8Q=33.2, aE46K=10.0, aA30P=14.1,
                         aA53T=11.4, ba12=45.7, a1_75=8.3, a71_72Del=13.0, aT72V=3.4, a71_74Del=16.7, a71E72E=85.3,
                         a71_81Del=64.6, a45V46V71E72E=35.6, fr_bsyn=55.4, a45V46V=5.7, bE46V=43.7, b45V46V=45.6,
                         aK45V=9.1, aE46V=4.2)

coefs_4props = namedtuple('coefficients_4props', 'nmbp_cf, nmh_cf, nmnc_cf, nmtc_cf')
# These are the original coefficients from Zibaee et al 2010 JBC.
# original_cfs_4props = coefs_4props(nmbp_cf=-0.0741, nmh_cf=0.0662, nmnc_cf=0.0629, nmtc_cf=0.0601)

# New coefficients calculated here, using same datasets generated for Zibaee et al 2010 JBC, plus data from Zibaee et
# al. 2007 JMB.
# Here coefficients are determined from best fit line using scikit-learn linear regression model.
cfs_4props = coefs_4props(nmbp_cf=-0.0, nmh_cf=0.0, nmnc_cf=0.0, nmtc_cf=0.0)
intcpts_4props = namedtuple('intercepts_4props', 'nmbp_ic nmh_ic nmnc_ic nmtc_ic', )
intcpts = intcpts_4props(nmbp_ic=-0.0, nmh_ic=0.0, nmnc_ic=0.0, nmtc_ic=0.0)


def _compute_mprops(_4norm_props: NamedTuple) -> float:
    return ((cfs_4props.nmbp_cf * _4norm_props.nmbp) + intcpts_4props.nmbp_ic) +\
           ((cfs_4props.nmh_cf * _4norm_props.mh) + intcpts_4props.nmh_ic) +\
           ((cfs_4props.nmnc_cf * _4norm_props.nmnc) + intcpts_4props.nmnc_ic) + \
           ((cfs_4props.nmtc_cf * _4norm_props.nmtc) + intcpts_4props.nmtc_ic)


def _make_fragment(syn_name: str) -> str:
    prot = ''
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    gsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_HUMAN')['SYUG_HUMAN']
    if syn_name[0] == 'a':
        prot = asyn
    elif syn_name[0] == 'b':
        prot = bsyn
    elif syn_name[0] == 'g':
        prot = gsyn
    else:
        print(f'Character should be a, b or g. Character passed was {syn_name[0]}')
    if 'Del' in syn_name:
        start_end = syn_name[1:-3].split('_')
        fragment = prot[: int(start_end[0]) - 1] + prot[int(start_end[1]):]
    else:
        start_end = syn_name[1:].split('_')
        fragment = prot[int(start_end[0]) - 1: int(start_end[1])]
    return fragment


def _build_syn_sequences(syn_names: list) -> dict:
    asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_HUMAN')['SYUA_HUMAN']
    bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_HUMAN')['SYUB_HUMAN']
    gsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG_HUMAN')['SYUG_HUMAN']
    fr_asyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUA_FUGU')['SYUA_FUGU']
    fr_bsyn = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUB_FUGU')['SYUB_FUGU']
    fr_gsyn1 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG1_FUGU')['SYUG1_FUGU']
    fr_gsyn2 = read_seqs.get_sequences_by_uniprot_accession_nums_or_names('SYUG2_FUGU')['SYUG2_FUGU']
    b5V = mutator.mutate(prot_seq=bsyn, pos_aa={11: 'V', 19: 'V', 63: 'V', 78: 'V', 102: 'V'})
    syn_seqs = {'asyn': asyn, 'bsyn': bsyn, 'gsyn': gsyn, 'fr_asyn': fr_asyn, 'fr_gsyn1': fr_gsyn1,
                'fr_gsyn2': fr_gsyn2, 'b5V': b5V}
    for syn_name in syn_names:
        syn_seqs_keys = list(syn_seqs)
        if syn_name not in syn_seqs_keys:
            if syn_name in ['a11_140', 'a21_140', 'a31_140', 'a41_140', 'a51_140', 'a61_140', 'a71_140', 'a1_75',
                            'a1_80', 'b1_73', 'a68_71Del', 'a71_72Del', 'a71_72Del', 'a71_74Del', 'a71_78Del',
                            'a71_81Del', 'a73_83Del', 'a74_84Del']
                syn_seqs[syn_name] = _make_fragment(syn_name)
            elif syn_name == 'ba1': # bsyn1-72, asyn73-83, bsyn73-134 ..??
                syn_seqs[syn_name] = bsyn[:72] + asyn[72:83] + bsyn[72:]
            elif syn_name == 'ba12': # asyn1-96, bsyn49 c-terminal residues...??
                syn_seqs[syn_name] = asyn[:96] + bsyn[85:]

            elif syn_name == 'ga':  # THIS ONE NEEDS TO BE CHECKED - IN MPROPS FILE
                syn_seqs[syn_name] = gsyn[:72] + asyn[72:83] + gsyn[72:]

            elif syn_name == 'b5V':
                syn_seqs[syn_name] = b5V
            elif syn_name == 'b5V2Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5V, pos_aa={125: 'Q', 126: 'Q'})
            elif syn_name == 'b5V4Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5V, pos_aa={104: 'Q', 105: 'Q', 125: 'Q', 126: 'Q'})
            elif syn_name == 'b5V6Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5V, pos_aa={96: 'Q', 97: 'Q', 104: 'Q', 105: 'Q',
                                                                          125: 'Q', 126: 'Q'})
            elif syn_name == 'b5V8Q':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=b5V, pos_aa={87: 'Q', 88: 'Q', 96: 'Q', 97: 'Q',
                                                                          104: 'Q', 105: 'Q', 125: 'Q', 126: 'Q'})

            elif syn_name == 'aA30P':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={30: 'P'})
            elif syn_name == 'aE46K':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={46: 'K'})
            elif syn_name == 'aA53T':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={53: 'T'})

            elif syn_name == 'aK45V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V'})
            elif syn_name == 'aE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={46: 'V'})
            elif syn_name == 'a45V46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V'})
            elif syn_name == 'a45V46V71E72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={45: 'V', 46: 'V', 71: 'E', 72: 'E'})
            elif syn_name == 'a71E72E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={71: 'E', 72: 'E'})

            elif syn_name == 'bR45V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={45: 'V'})
            elif syn_name == 'b45V46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={45: 'V', 46: 'V'})
            elif syn_name == 'bE46V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=bsyn, pos_aa={46: 'V'})

            elif syn_name == 'aT72V':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={72: 'V'})
            elif syn_name == 'aS87E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={87: 'E'})
            elif syn_name == 'aS129E':
                syn_seqs[syn_name] = mutator.mutate(prot_seq=asyn, pos_aa={129: 'E'})

            elif syn_name == 'fr_asyn':
                syn_seqs[syn_name] = fr_asyn
            elif syn_name == 'fr_bsyn':
                syn_seqs[syn_name] = fr_bsyn
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


def calculate_relative_weights_for_each_4_props() -> Tuple[NamedTuple, NamedTuple, NamedTuple]:
    """
    The function should only need to be run once to generate the necessary models for predictions.

    The coefficients of each of following 4 plots were used for combining the 4 properties into one equation.
    The 4 plots were of nmbp, nmh, nmnc, nmtc against natural log of the lag-times data of 32 **TODO more than 32**
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

    x_nmbp, x_nmh, x_nmnc, x_nmtc = x_nmbp.reshape((-1, 1)), x_nmh.reshape((-1, 1)),\
                                    x_nmnc.reshape((-1, 1)), x_nmtc.reshape((-1, 1))

    model = LinearRegression()
    model.fit(x_nmbp, y)
    mbp_rsq = model.score(x_nmbp, y)
    mbp_cf = float(model.coef_)
    mbp_ic = float(model.intercept_)

    model.fit(x_nmh, y)
    mh_rsq = model.score(x_nmh, y)
    mh_cf = float(model.coef_)
    mh_ic = float(model.intercept_)

    model.fit(x_nmnc, y)
    mnc_rsq = model.score(x_nmnc, y)
    mnc_cf = float(model.coef_)
    mnc_ic = float(model.intercept_)

    model.fit(x_nmtc, y)
    mtc_rsq = model.score(x_nmtc, y)
    mtc_cf = float(model.coef_)
    mtc_ic = float(model.intercept_)

    _4coefs = namedtuple('_4coefs', 'mbp_cf mh_cf mnc_cf mtc_cf')
    _4coefs = _4coefs(mbp_cf=mbp_cf, mh_cf=mh_cf, mnc_cf=mnc_cf, mtc_cf=mtc_cf)
    _4intcpts = namedtuple('_4intcpts', 'mbp_ic mh_ic mnc_ic mtc_ic')
    _4intcpts = _4intcpts(mbp_ic=mbp_ic, mh_ic=mh_ic, mnc_ic=mnc_ic, mtc_ic=mtc_ic)
    _4rsq = namedtuple('_4rsq', 'mbp_rsq mh_rsq mnc_rsq mtc_rsq')
    _4rsq = _4rsq(mbp_rsq=mbp_rsq, mh_rsq=mh_rsq, mnc_rsq=mnc_rsq, mtc_rsq=mtc_rsq)
    return _4coefs, _4intcpts, _4rsq


def _normalise_mprops(mprops):
    return 0


def compute_normalised_mprops(syn_seqs: NamedTuple):
    """

    :param syn_seqs: Synuclein name and its amino acid sequence
    :return:
    """
    _4coefs, _4intcpts, _4rsq = calculate_relative_weights_for_each_4_props() # only needs to be done once
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


def calculate_relative_weights_for_nmprops_and_nbsc_intgrl(syn_seqs):
    nmprops = compute_normalised_mprops(syn_seqs)
    nbsc_integral = compute_normalised_salsa_bsc_integrals(syn_seqs)
    # calculate combined_algo TODO


def predict_lag_time_with_combined_algo(syn_seqs):
    # TODO
    pass


if __name__ == '__main__':
    calculate_relative_weights_for_each_4_props()

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