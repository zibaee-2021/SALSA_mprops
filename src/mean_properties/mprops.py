from typing import Tuple
from pandas import DataFrame as pDF
from src.mean_properties import mbp, mh, mnc_mtc
import numpy as np
from sklearn.linear_model import LinearRegression
from data.aa_properties.read_props import Scales
from src.utils import plotter, utils
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


def _calc_relative_weights_per_prop(syns_lags_seqs_props: pDF, make_plot: bool) -> Tuple[dict, dict, dict]:
    """
    mprops is a weighted sum of the 4 mean properties. These relative weights are the coefficients (i.e. slopes of
    linear plots) of each of following 4 plots (nmbp, nmh, nmnc, nmtc) against the natural log of the lag-times of
    xxx-4xx  recombinant synuclein constructs. TODO input the range of synuclein numbers to be used...

    The lag-times were the time taken for the ThT value to become the squared of its value at time = 0. The 41
    synuclein constructs that were included satisfied the following two requirements:
    1. They assembled with a detectable lag-time within 96 hours;
    2. They had been assayed ≥ 3 times (each from a different protein preparation batch).
    :param syns_lags_seqs_props: Synuclein names, lag times, amino acid sequences and the 4 calculated nean properties.
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties
    against the natural log of 'lag times'.
    :return: The 4 coefficients, 4 y-intercepts and 4 R-squared values of the 4 mean properties: mbp, mh, mnc, mtc.
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
        if make_plot: plotter.plot(model, x=prop_values, y=y, data_labels=list(syns_lags_seqs_props.index),
                                   title=prop_name, x_label=prop_name)
        _4coefs[prop_name] = round(float(model.coef_), 3)
        _4intcpts[prop_name] = round(float(model.intercept_), 3)
        _4rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
    return _4coefs, _4intcpts, _4rsq


def compute_4_normalised_props(syns_lags_seqs: pDF) -> pDF:
    """
    Compute normalised values of the 4 properties for given sequences, (mapped to log of lag times).
    :param syns_lags_seqs: Synucleins, their log of lag times and sequences.
    :return: Synucleins, their log of lag times, sequences and the 4 normalised mean properties.
    ['lag_time_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc']
    """
    syns_lags_seqs['mbp'] = syns_lags_seqs['seqs'].apply(mbp.compute_mean_beta_sheet_prop)
    syns_lags_seqs['mh'] = syns_lags_seqs['seqs'].apply(lambda row:
                                                        mh.compute_mean_hydrophobicity(row, Scales.RW.value))
    syns_lags_seqs['mnc'] = syns_lags_seqs['seqs'].apply(mnc_mtc.compute_mean_net_charge)
    syns_lags_seqs['mtc'] = syns_lags_seqs['seqs'].apply(mnc_mtc.compute_mean_total_charge)
    for prop_name in ['mbp', 'mh', 'mnc', 'mtc']:
        max_ = np.max(list(syns_lags_seqs[prop_name]))
        min_ = np.min(list(syns_lags_seqs[prop_name]))
        syns_lags_seqs[f'n{prop_name}'] = syns_lags_seqs[prop_name].apply(lambda row: (row - min_) / (max_ - min_))
    return syns_lags_seqs


def compute_norm_mprops(syns_lnlags_seqs: pDF, make_plot: bool) -> pDF:
    """
    Compute normalised mprops.
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties
    against the natural log of 'lag times'.
    :return: Synucleins (index), normalised mprops and sequences.
    ['lag_time_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc', 'mprops', 'nmprops']
    Sliced down to two columns: ['lag_time_means', 'ln_lags', 'nmprops', 'seqs'].
    """
    # syns_lnlags_seqs = syns_lnlags_seqs.drop(['a1_80', 'fr_asyn', 'fr_bsyn', 'fr_gsyn1', 'fr_gsyn2', 'gsyn',
    #                                                   'ga'], axis=0)
    syns_lnlags_seqs_4props = compute_4_normalised_props(syns_lnlags_seqs)
    _4coefs, _4intcpts, _4rsq = _calc_relative_weights_per_prop(syns_lnlags_seqs_4props, make_plot=make_plot)
    syns_lnlags_seqs['mprops'] = syns_lnlags_seqs.apply(lambda row: row.nmbp * _4coefs['nmbp'] +
                                                                    row.nmh * _4coefs['nmh'] +
                                                                    row.nmnc * _4coefs['nmnc'] +
                                                                    row.nmtc * _4coefs['nmtc'], axis=1)
    max_ = np.max(list(syns_lnlags_seqs_4props['mprops']))
    min_ = np.min(list(syns_lnlags_seqs_4props['mprops']))
    syns_lnlags_seqs_4props['nmprops'] = syns_lnlags_seqs_4props['mprops'].apply(lambda row: (row - min_) / (max_ - min_))
    print(f'_4rsq {_4rsq}')
    return syns_lnlags_seqs_4props[['lag_time_means', 'ln_lags', 'nmprops', 'seqs']]


if __name__ == '__main__':
    syns_lnlags_seqs = utils.get_ln_lags_and_build_seqs()
    compute_norm_mprops(syns_lnlags_seqs, make_plot=True)
