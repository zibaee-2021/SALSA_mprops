import os
from typing import Tuple, NamedTuple
import pandas as pd
from pandas import DataFrame as pDF
from src.mean_properties import mbp, mh, mnc_mtc
from collections import namedtuple
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from src.salsa import execute
from src.salsa.Options import DefaultBSC, Props
from data.protein_sequences import read_seqs
from src.mutation import mutator
from root_path import abspath_root
import matplotlib.pyplot as plt
"""
The derivation of the coefficients for each of the 4 mean properties (mean beta-sheet propensity, mean hydrophilicity, 
mean absolute net charge and mean total charge) is based on the kinetics of fibrillogenesis for recombinant 
synuclein constructs measured in vitro. The calculations are similar to those described in Zibaee et al 2010 JBC
but differ in some important details. 

The combination of the 4 weighted properties into one is referred to as `mprops`. 
The combination of normalised mprops and normalised SALSA b-strand contiguity integrals into one calculation is just 
referred to as the `combined algorithm`. The use of the combined algorithm produces a fit to the lag times with an 
Rsquared ("coefficient of determination") of ... (Rsquared of 1.0 is a perfect fit).

Given the stochastic nature of fibrillogenesis, values are likely to be specific, not only to proteins that are 
natively unfolded, and not only to synuclein-sized proteins, but also for the specific experimental conditions of the 
assays as well. Hence I specify these conditions alongside the code and mathematical derivations: 
Lag times of fibrillogensis of synuclein constructs was based on experiments under the following conditions: 
- Protein concentration at 400 microMolars in 30 mM MOPs buffer pH 7.2 and 10 microMolar ThT. 
- Incubation at 37degC, shaken at 450 rpm. 
- 10 microlitre samples collect at 5 subsequent time points 4h, 8h, 24h, 48h, 96h. All measurements were performed on 
each sample immediately after collection.
Lag times were measured from each experiment, as time taken for ThT values to reach the square of their zero
time emission readings at 480nm. The final value is the mean of the lag times for each constructs.

A very important caveat is that for certain synuclein constructs for which lag time data was included, 
which had longer lag times, those proteins did not begin to assemble within the 100 hours limit. Hence, 
the data from these experiments was not included. As such their included data indicates these proteins to be more 
fibrillogenic than they actually were taking in to account the non-inclusion of the aforementioned experiments. 
"""


def _plot(model, x: np.array, y: np.array, data_labels: list, title: str, x_label: str, y_label: str = 'ln(lag-time)'):
    plt.scatter(x, y, color='g')
    plt.plot(x, model.predict(x), color='k')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for (dl, x, y) in zip(data_labels, x, y):
        plt.annotate(dl, (x, y + 0.1), size=6)
    plt.show()
    # ax = df.plot('x', 'y', kind='scatter', s=50)


def _train_combo_model(syns_lags_seqs_props, make_plot: bool) -> Tuple[LinearRegression, float]:
    """

    :param syns_lags_seqs_props:
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties
    against the natural log of 'lag times'.
    :return:
    """
    x_combo = np.array(syns_lags_seqs_props.combo)
    x_combo = x_combo.reshape((-1, 1))
    y = np.array(syns_lags_seqs_props.ln_lags)
    model = LinearRegression()
    model.fit(x_combo, y)
    rsq = round(float(model.score(x_combo, y)), 3)
    if make_plot: _plot(model, x=x_combo, y=y, data_labels=list(syns_lags_seqs_props.index), title='combo',
                        x_label='combined properties')
    return model, rsq


def _calc_relative_weights_for_nmprops_and_nbsc(syns_lags_seqs_props: pDF, make_plot: bool) -> Tuple[dict, dict, dict]:
    x_nbsc, x_nmprops = np.array(syns_lags_seqs_props.nbsc), np.array(syns_lags_seqs_props.nmprops)
    x_nbsc, x_nmprops = x_nbsc.reshape((-1, 1)), x_nmprops.reshape((-1, 1))
    y = np.array(syns_lags_seqs_props.ln_lags)
    _2coefs, _2intcpts, _2rsq = {}, {}, {}
    # NOTE: Pycharm bug `TypeError: 'NoneType' object is not callable` in debugger mode.
    for prop_name, prop_values in zip(['nbsc', 'nmprops'], [x_nbsc, x_nmprops]):
        model = LinearRegression()
        model.fit(prop_values, y)
        if make_plot: _plot(model, x=prop_values, y=y, data_labels=list(syns_lags_seqs_props.index),
                            x_label=prop_name, title=prop_name)
        _2coefs[prop_name] = round(float(model.coef_), 3)
        _2intcpts[prop_name] = round(float(model.intercept_), 3)
        _2rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
    print(f'_2rsq {_2rsq}')
    return _2coefs, _2intcpts, _2rsq


def _compute_salsa_bsc_integrals(seq: str) -> float:
    scored_windows_all = execute.compute(sequence=seq, _property=Props.bSC.value, params=DefaultBSC.all_params.value)
    summed_scores = execute.sum_scores_for_plot(scored_windows_all)
    return execute.integrate_salsa_plot({'seq': summed_scores})['seq']


def _calc_relative_weights_per_prop(syns_lags_seqs_props: pDF) -> Tuple[dict, dict, dict]:
    """
    The function should only need to be run once to generate the necessary models for predictions.

    The coefficients of each of following 4 plots were used for combining the 4 properties into one equation.
    The 4 plots were of nmbp, nmh, nmnc, nmtc against natural log of the lag-times data of 41 recombinant synuclein
    constructs.

    The lag-times were the time taken for the ThT value to become the squared of its value at time = 0. The 41
    synuclein constructs that were included satisfied the following two requirements:
    1. They assembled with a detectable lag-time within 96 hours;
    2. They had been assayed ≥ 3 times (each from a different protein preparation batch).
    :return:
    """
    x_nmbp, x_nmh, x_nmnc, x_nmtc = np.array(syns_lags_seqs_props.nmbp), np.array(syns_lags_seqs_props.nmh), \
                                    np.array(syns_lags_seqs_props.nmnc), np.array(syns_lags_seqs_props.nmtc)
    x_nmbp, x_nmh, x_nmnc, x_nmtc = x_nmbp.reshape((-1, 1)), x_nmh.reshape((-1, 1)),\
                                    x_nmnc.reshape((-1, 1)), x_nmtc.reshape((-1, 1))
    y = np.array(syns_lags_seqs_props.ln_lags)
    _4coefs, _4intcpts, _4rsq = {}, {}, {}
    # NOTE: Pycharm bug `TypeError: 'NoneType' object is not callable` in debugger mode.
    for prop_name, prop_values in zip(['nmbp', 'nmh', 'nmnc', 'nmtc'], [x_nmbp, x_nmh, x_nmnc, x_nmtc]):
        model = LinearRegression()
        model.fit(prop_values, y)
        if make_plot: _plot(model, x=prop_values, y=y, data_labels=list(syns_lags_seqs_props.index), title=prop_name,
                            x_label=prop_name)
        _4coefs[prop_name] = round(float(model.coef_), 3)
        _4intcpts[prop_name] = round(float(model.intercept_), 3)
        _4rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
    return _4coefs, _4intcpts, _4rsq


def _compute_4_normalised_props(syns_lags_seqs: pDF) -> pDF:
    syns_lags_seqs['mbp'] = syns_lags_seqs['seqs'].apply(mbp.compute_mean_beta_sheet_prop)
    syns_lags_seqs['mh'] = syns_lags_seqs['seqs'].apply(mh.compute_mean_hydrophilicity)
    syns_lags_seqs['mnc'] = syns_lags_seqs['seqs'].apply(mnc_mtc.compute_mean_net_charge)
    syns_lags_seqs['mtc'] = syns_lags_seqs['seqs'].apply(mnc_mtc.compute_mean_total_charge)
    for prop_name in ['mbp', 'mh', 'mnc', 'mtc']:
        max_ = np.max(list(syns_lags_seqs[prop_name]))
        min_ = np.min(list(syns_lags_seqs[prop_name]))
        syns_lags_seqs[f'n{prop_name}'] = syns_lags_seqs[prop_name].apply(lambda row: (row - min_) / (max_ - min_))
    return syns_lags_seqs


def _build_syn_sequences(syns_lags: list) -> dict:
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


def _get_ln_lags_and_4norm_props() -> pDF:
    syns_lags = pd.read_csv(os.path.join(abspath_root, 'data', 'tht_data', 'lag_time_degree_4.csv'), index_col=[0])
    syns_lags['ln_lags'] = syns_lags.apply(lambda row: math.log(row['lag_time_means']), axis=1)
    syn_seqs_dict = _build_syn_sequences(syns_lags)
    syn_seqs = pDF.from_dict(syn_seqs_dict, orient='index', columns=['seqs'])
    syns_lags_seqs = syns_lags.join(syn_seqs)
    return _compute_4_normalised_props(syns_lags_seqs)


def _compute_norm_mprops() -> pDF:
    syns_lags_seqs_props = _get_ln_lags_and_4norm_props()
    _4coefs, _4intcpts, _4rsq = _calc_relative_weights_per_prop(syns_lags_seqs_props)
    syns_lags_seqs_props['mprops'] = syns_lags_seqs_props.apply(lambda row: row.nmbp * _4coefs['nmbp'] +
                                                                            row.nmh * _4coefs['nmh'] +
                                                                            row.nmnc * _4coefs['nmnc'] +
                                                                            row.nmtc * _4coefs['nmtc'], axis=1)
    max_ = np.max(list(syns_lags_seqs_props['mprops']))
    min_ = np.min(list(syns_lags_seqs_props['mprops']))
    syns_lags_seqs_props['nmprops'] = syns_lags_seqs_props['mprops'].apply(lambda row: (row - min_) / (max_ - min_))
    print(f'_4rsq {_4rsq}')
    return syns_lags_seqs_props


def predict_lag_times(make_plot: bool) -> pDF:
    """
    Predict lag time hours for all synucleins based on mean properties and beta-strand contiguity trained with
    observed lag times. Optionally display plots for each of the properties.
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties, for a plot
    of mprops, a plot for beta-strand contiguity integrals and finally a plot of the combination of mprops and bSC.
    All of the plots are against the natural log of 'lag times' on the y-axes.
    :return: Synucleins mapped to predicted lag times in hours.
    """
    syns_lags_seqs_props = _compute_norm_mprops(make_plot=True)
    syns_lags_seqs_props['bsc'] = syns_lags_seqs_props['seqs'].apply(_compute_salsa_bsc_integrals)
    max_ = np.max(list(syns_lags_seqs_props['bsc']))
    min_ = np.min(list(syns_lags_seqs_props['bsc']))
    syns_lags_seqs_props['nbsc'] = syns_lags_seqs_props['bsc'].apply(lambda row: (row - min_) / (max_ - min_))
    _2coefs, _2intcpts, _2rsq = _calc_relative_weights_for_nmprops_and_nbsc(syns_lags_seqs_props, make_plot=True)
    syns_lags_seqs_props['combo'] = syns_lags_seqs_props.apply(lambda row: row.nbsc * _2coefs['nbsc'] +
                                                                           row.nmprops * _2coefs['nmprops'], axis=1)
    model, rsq = _train_combo_model(syns_lags_seqs_props, make_plot=True)
    coef = round(float(model.coef_), 3)
    intcpt = round(float(model.intercept_), 3)
    print(f'rsq {rsq}')
    preds = model.predict(np.array(syns_lags_seqs_props['combo']).reshape(-1, 1))
    syns_lags_seqs_props['pred'] = np.exp(model.predict(np.array(syns_lags_seqs_props['combo']).reshape(-1, 1)))
    return syns_lags_seqs_props


import time
if __name__ == '__main__':
    st = time.time()
    df = predict_lag_times(True)
    print(f'time {time.time() - st} secs')
    print(f'columns {df.columns}')
    print(df.head(41))
