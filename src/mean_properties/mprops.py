import os
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
referred to as the `combined algorithm`. The use of the combined algorithm produces a fit to the lagtimes with an 
Rsquared ("coefficient of determination") of ... (Rsquared of 1.0 is a perfect fit).

Given the stochastic nature of fibrillogenesis, values are likely to be specific, not only to proteins that are 
natively unfolded, and not only to synuclein-sized proteins, but also for the specific experimental conditions of the 
assays as well. Hence I specify these conditions alongside the code and mathematical derivations: 
lagtimes of fibrillogenesis of synuclein constructs was based on experiments under the following conditions: 
- Protein concentration at 400 microMolars in 30 mM MOPs buffer pH 7.2 and 10 microMolar ThT. 
- Incubation at 37degC, shaken at 450 rpm. 
- 10 microlitre samples collect at 5 subsequent time points 4h, 8h, 24h, 48h, 96h. All measurements were performed on 
each sample immediately after collection.
lagtimes were measured from each experiment, as time taken for ThT values to reach the square of their zero
time emission readings at 480nm. The final value is the mean of the lagtimes for each constructs.

A very important caveat is that for certain synuclein constructs for which lagtime data was included, 
which had longer lagtimes, those proteins did not begin to assemble within the 100 hours limit. Hence, 
the data from these experiments was not included. As such their included data indicates these proteins to be more 
fibrillogenic than they actually were taking in to account the non-inclusion of the aforementioned experiments. 
"""


def _calc_relative_weights_per_prop(pdf: pDF, make_plot: bool, model: LinearRegression) -> Tuple[dict, dict, dict]:
    """
    `mprops` is a weighted sum of the 4 mean properties. These relative weights are the coefficients (i.e. slopes of
    linear plots) of each of following 4 plots (nmbp, nmh, nmnc, nmtc) against the natural log of the
    lagtimes of recombinant synuclein constructs, typically numbering about 30 or more.
    The lagtimes are the time taken for the ThT value to become the square of its value at time = 0.
    41 synuclein constructs were included because they satisfied the following criteria:
        1. They assembled with a detectable lagtime within 96 hours;
        2. They had been assayed ≥ 3 times (each from a different protein preparation batch).
        3. (For regression models) the proportion of experiments, for a particular synuclein, with 'NA' ThT values was
        less than a given heuristic threshold (typically 1/8).
    :param pdf: Table of 12 columns including synucleins as index, lagtime means,
    log of lagtimes, amino acid sequences and the 4 calculated mean properties and their normalised values:
    ['lagtime_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc'].
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties against the
    natural log of lagtimes.
    :param model: Model to use for regression of lagtimes to physicochemical properties.
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
        if make_plot:
            plotter.plot(model, x=prop_values, y=y, data_labels=list(pdf.index),
                         title=prop_name, x_label=prop_name)
        _4coefs[prop_name] = round(float(model.coef_), 3)
        _4intcpts[prop_name] = round(float(model.intercept_), 3)
        # _4rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
        _4rsq[prop_name] = round(float(r2_score(y_pred=prop_values, y_true=y)), 3)
    return _4coefs, _4intcpts, _4rsq


def compute_4_normalised_props(pdf: pDF) -> pDF:
    """
    Compute normalised values of the 4 properties for given sequences, (mapped to log of lagtimes).
    :param pdf: Table of 4 columns including synucleins as index, lagtime means,
    natural log of lagtimes and amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    :return: Table of 12 columns including synucleins as index, lagtime means, log of lagtimes,
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
    :param pdf: Table of 4 columns including synucleins as index, lagtime means, natural log of lagtimes and
    amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'seqs'].
    :param make_plot: True to display 4 plots of linear regression model of each of the 4 mean properties against
    natural log of lagtimes.
    :param model: Model to use for regression of lagtimes to physicochemical properties.
    :return: Table of 5 columns including synucleins as index, lagtime means, log of lagtimes, normalised mprops
    and amino acid sequences: [(index), 'lagtime_means', 'ln_lags', 'nmprops', 'seqs'].
    """
    # pdf = pdf.drop(['a1_80', 'fr_asyn', 'fr_bsyn', 'fr_gsyn1', 'fr_gsyn2', 'gsyn',
    #                                                   'ga'], axis=0)
    # To use the same 32 Syns used in JBC 2010 paper:
    _32_syns_JBC = ['asyn', 'a11_140', 'a21_140', 'a31_140', 'a41_140', 'a51_140', 'a61_140',
                    'a1_80', 'a68_71del', 'a71_72del', 'a71_74del', 'a71_78del', 'a71_81del',
                    'a73_83del', 'ba1', 'ba12', 'b5V', 'b5V2Q', 'b5V4Q', 'b5V6Q', 'b5V8Q',
                    'aT72V', 'aV71ET72E', 'aA30P', 'aE46K', 'aA53T', 'aS87E', 'aS129E', 'b1_73',
                    'fr_asyn', 'fr_gsyn1', 'fr_gsyn2']
    pdf = pdf.loc[_32_syns_JBC]
    pdf_ = compute_4_normalised_props(pdf)
    _4coefs, _4intcpts, _4rsq = _calc_relative_weights_per_prop(pdf_, make_plot=make_plot, model=model)
    pdf['mprops'] = pdf.apply(lambda row: row.nmbp * _4coefs['nmbp'] + row.nmh * _4coefs['nmh'] +
                                          row.nmnc * _4coefs['nmnc'] + row.nmtc * _4coefs['nmtc'], axis=1)
    max_ = np.max(list(pdf_['mprops']))
    min_ = np.min(list(pdf_['mprops']))
    pdf_['nmprops'] = pdf_['mprops'].apply(
        lambda row: ((row - min_) / (max_ - min_)) + 0.01)
    print(f'_4rsq {_4rsq}')
    return pdf_[['lagtime_means', 'ln_lags', 'nmprops', 'seqs']]


from src.lagtimes import lagtime_calculator as ltc

if __name__ == '__main__':

    from src.utils.Constants import LagTimeCalc
    constants = LagTimeCalc()
    for degree_used in [5]:
        # for tht_end_value_used in [constants.SQUARE_OF_STARTING_VALUE, constants.DOUBLE_STARTING_VALUE]:
        for tht_end_value_used in [constants.SQUARE_OF_STARTING_VALUE]:
            lt_filename = f'ltMeans_polyDeg{degree_used}_ltEnd{int(tht_end_value_used)}.csv'
            lt_path_f = os.path.join(constants.LAGTIME_MEANS_PATH, lt_filename)
            if not os.path.exists(lt_path_f):
                print(f'{os.path.exists(lt_path_f)} does not exist.')
            else:
                print(f'Reading {os.path.exists(lt_path_f)}')
                _pdf = utils.get_loglags_and_build_seqs(csv_filename=lt_filename)
                syns = list(_pdf.index)
                print(f'This file has lag-times for {len(syns)} constructs. Namely: {syns}.')
                syns_to_exclude = ['a1_80', 'a1_75', 'b1_73', 'fr_asyn', 'fr_bsyn',
                                   'fr_gsyn1', 'fr_gsyn2', 'gsyn', 'ga']
                print(f'Curation of this currently will exclude {len(syns_to_exclude)} constructs. '
                      f'Namely: {syns_to_exclude}. They may not all be present in the data.')
                for syn in syns_to_exclude:
                    if syn in _pdf.index:
                        _pdf = _pdf.drop(syn, axis=0)

                syns = list(_pdf.index)

                print(f'After excluding, there are now {len(syns)} constructs remaining. Namely: {list(_pdf.index)}.')
                # for model_ in [LinearRegression(), Lasso(), Ridge(), ElasticNet()]:
                for model_ in [LinearRegression(), HuberRegressor(epsilon=1.5, alpha=0.0)]:
                    print(f'model used is {str(model_)}')
                    _combo_pdf = compute_norm_mprops(_pdf, make_plot=True, model=model_)

    print('end')
# _syns_not_included_in_32_JBC  = ['gsyn', 'fr_asyn', 'fr_gsyn1', 'fr_gsyn2', 'mus_bsyn',
#                                  'gallus_bsyn', 'b5V', 'a71_140', 'a1_45', 'a1_50', 'a1_55', 'a1_60',
#                                  'a1_70', 'a1_75', 'g1_80', 'a71_76del', 'a71_82del', 'a74_84del',
#                                  'a73_82del', 'bsyn', 'ga', 'aK45V', 'aE46V', 'aK45VE46V',
#                                  'aK45VE46VV71ET72E', 'bR45V', 'bR45VE46V', 'bE46V', 'mus_bsyn',
#                                  'gallus_bsyn']
