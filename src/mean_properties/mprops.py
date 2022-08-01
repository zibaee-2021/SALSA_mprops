from typing import Tuple
from pandas import DataFrame as pDF
from src.mean_properties import mbp, mh, mnc_mtc
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from data.aa_properties.read_props import Scales
from src.utils import plotter, utils
"""
The derivation of the coefficients for each of the 4 mean properties (mean beta-sheet propensity, mean hydrophilicity, 
mean absolute net charge and mean total charge) is based on the kinetics of fibrillogenesis for recombinant 
synuclein constructs measured in vitro. The calculations are similar to those described in Zibaee et al 2010 JBC
but differ in some important details. 

The combination of the 4 weighted properties into one is referred to as `mprops`. 
The combination of normalised mprops and normalised SALSA b-strand contiguity integrals into one calculation is just 
referred to as the `combined algorithm`. The use of the combined algorithm produces a fit to the 'lag-times' with an 
Rsquared ("coefficient of determination") of ... (Rsquared of 1.0 is a perfect fit).

Given the stochastic nature of fibrillogenesis, values are likely to be specific, not only to proteins that are 
natively unfolded, and not only to synuclein-sized proteins, but also for the specific experimental conditions of the 
assays as well. Hence I specify these conditions alongside the code and mathematical derivations: 
'Lag-times' of fibrillogenesis of synuclein constructs was based on experiments under the following conditions: 
- Protein concentration at 400 microMolars in 30 mM MOPs buffer pH 7.2 and 10 microMolar ThT. 
- Incubation at 37degC, shaken at 450 rpm. 
- 10 microlitre samples collect at 5 subsequent time points 4h, 8h, 24h, 48h, 96h. All measurements were performed on 
each sample immediately after collection.
'Lag-times' were measured from each experiment, as time taken for ThT values to reach the square of their zero
time emission readings at 480nm. The final value is the mean of the 'lag-times' for each constructs.

A very important caveat is that for certain synuclein constructs for which 'lag-time' data was included, 
which had longer 'lag-times', those proteins did not begin to assemble within the 100 hours limit. Hence, 
the data from these experiments was not included. As such their included data indicates these proteins to be more 
fibrillogenic than they actually were taking in to account the non-inclusion of the aforementioned experiments. 
"""


def _calc_relative_weights_per_prop(pdf: pDF, make_plot: bool, model: LinearRegression) -> Tuple[dict, dict, dict]:
    """
    `mprops` is a weighted sum of the 4 mean properties. These relative weights are the coefficients (i.e. slopes of
    linear plots) of each of following 4 plots (nmbp, nmh, nmnc, nmtc) against the natural log of the
    'lag-times' of recombinant synuclein constructs, typically numbering about 30 or more.
    The 'lag-times' are the time taken for the ThT value to become the square of its value at time = 0.
    41 Synuclein constructs were included because they satisfied the following criteria:
        1. They assembled with a detectable 'lag-time' within 96 hours;
        2. They had been assayed ≥ 3 times (each from a different protein preparation batch).
        3. (For regression models) the proportion of experiments, for a particular Synuclein, with 'NA' ThT values was
        less than a given heuristic threshold (typically 1/8).
    :param pdf: Table of 12 columns including Synucleins as index, 'lag-time' means,
    log of 'lag-times', amino acid sequences and the 4 calculated mean properties and their normalised values:
    ['lagtime_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc'].
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties against the
    natural log of 'lag-times'.
    :param model: Model to use for regression of 'lag-times' to physicochemical properties.
    :return: The 4 coefficients, 4 y-intercepts and 4 R-squared values of the 4 mean properties: mbp, mh, mnc, mtc.
    """
    x_nmbp, x_nmh, x_nmnc, x_nmtc = np.array(pdf.nmbp), np.array(pdf.nmh), np.array(pdf.nmnc), np.array(pdf.nmtc)
    x_nmbp, x_nmh, x_nmnc, x_nmtc = x_nmbp.reshape((-1, 1)), x_nmh.reshape((-1, 1)),\
                                    x_nmnc.reshape((-1, 1)), x_nmtc.reshape((-1, 1))
    y = np.array(pdf.ln_lags)
    _4coefs, _4intcpts, _4rsq = {}, {}, {}
    # NOTE: Pycharm bug `TypeError: 'NoneType' object is not callable` in debugger mode.
    for prop_name, prop_values in zip(['nmbp', 'nmh', 'nmnc', 'nmtc'], [x_nmbp, x_nmh, x_nmnc, x_nmtc]):
        model.fit(prop_values, y)
        if make_plot: plotter.plot(model, x=prop_values, y=y, data_labels=list(syns_lags_seqs_props.index),
                                   title=prop_name, x_label=prop_name)
        _4coefs[prop_name] = round(float(model.coef_), 3)
        _4intcpts[prop_name] = round(float(model.intercept_), 3)
        # _4rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
        _4rsq[prop_name] = round(float(r2_score(y_pred=prop_values, y_true=y)), 3)
    return _4coefs, _4intcpts, _4rsq


def compute_4_normalised_props(pdf: pDF) -> pDF:
    """
    Compute normalised values of the 4 properties for given sequences, (mapped to log of 'lag-times').
    :param pdf: Table of 4 columns including Synucleins as index, 'lag-time' means,
    natural log of 'lag-times' and amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    :return: Table of 12 columns including Synucleins as index, 'lag-time' means, log of 'lag-times',
    amino acid sequences, the 4 mean properties and the 4 mean properties normalised:
    [(index), 'lagtime_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc']
    """
    pdf['mbp'] = pdf['seqs'].apply(mbp.compute_mean_beta_sheet_prop)
    pdf['mh'] = pdf['seqs'].apply(lambda row: mh.compute_mean_hydrophobicity(row, Scales.RW.value))
    pdf['mnc'] = pdf['seqs'].apply(mnc_mtc.compute_mean_net_charge)
    pdf['mtc'] = pdf['seqs'].apply(mnc_mtc.compute_mean_total_charge)
    for prop_name in ['mbp', 'mh', 'mnc', 'mtc']:
        max_ = np.max(list(pdf[prop_name]))
        min_ = np.min(list(pdf[prop_name]))
        pdf[f'n{prop_name}'] = pdf[prop_name].apply(
            lambda row: ((row - min_) / (max_ - min_)) + 0.01)
    return pdf


def compute_norm_mprops(pdf: pDF, make_plot: bool, model: LinearRegression) -> pDF:
    """
    Compute normalised mprops.
    :param pdf: Table of 4 columns including Synucleins as index, 'lag-time' means, natural log of 'lag-times' and
    amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    :param make_plot: True to display 4 plots of linear regression model of each of the 4 mean properties against
    natural log of 'lag-times'.
    :param model: Model to use for regression of 'lag-times' to physicochemical properties.
    :return: Table of 5 columns including Synucleins as index, 'lag-time' means, log of 'lag-times', normalised mprops
    and amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'nmprops', 'seqs'].
    """
    # pdf = pdf.drop(['a1_80', 'fr_asyn', 'fr_bsyn', 'fr_gsyn1', 'fr_gsyn2', 'gsyn',
    #                                                   'ga'], axis=0)
    syns_lnlags_seqs_4props = compute_4_normalised_props(syns_lnlags_seqs)
    _4coefs, _4intcpts, _4rsq = _calc_relative_weights_per_prop(syns_lnlags_seqs_4props, make_plot=make_plot)
    syns_lnlags_seqs['mprops'] = syns_lnlags_seqs.apply(lambda row: row.nmbp * _4coefs['nmbp'] +
                                                                    row.nmh * _4coefs['nmh'] +
                                                                    row.nmnc * _4coefs['nmnc'] +
                                                                    row.nmtc * _4coefs['nmtc'], axis=1)
    max_ = np.max(list(syns_lnlags_seqs_4props['mprops']))
    min_ = np.min(list(syns_lnlags_seqs_4props['mprops']))
    syns_lnlags_seqs_4props['nmprops'] = syns_lnlags_seqs_4props['mprops'].apply(
        lambda row: ((row - min_) / (max_ - min_)) + 0.01)
    print(f'_4rsq {_4rsq}')
    return syns_lnlags_seqs_4props[['lagtime_means', 'ln_lags', 'nmprops', 'seqs']]


if __name__ == '__main__':
    from src.lagtimes import lagtime_calculator as lc
    for degree_used in [2, 3, 4, 5]:
        for tht_end_value_used in [lc.SQUARE_OF_STARTING_VALUE, lc.DOUBLE_STARTING_VALUE]:
            lt_filename = f'lagtime_means_polynDegree_{degree_used}_lagtimeEndvalue_{int(tht_end_value_used)}.csv'
            syns_lnlags_seqs = utils.get_ln_lags_and_build_seqs(lagtime_means_csv_filename=lt_filename)
            compute_norm_mprops(syns_lnlags_seqs, make_plot=True)
