from typing import Tuple
from pandas import DataFrame as pDF
from src.mean_properties import mprops
import numpy as np
from sklearn.linear_model import LinearRegression
from src.salsa import salsa
from src.utils import plotter, utils
"""
The derivation of the coefficients for each of the 4 mean properties (mean beta-sheet propensity, mean hydrophilicity, 
mean absolute net charge and mean total charge) is based on the kinetics of fibrillogenesis for recombinant 
Synuclein constructs measured in vitro. The calculations are similar to those described in Zibaee et al 2010 JBC
but differ in some important details. 

The combination of the 4 weighted properties into one is referred to as `mprops`. 
The combination of normalised mprops and normalised SALSA b-strand contiguity integrals into one calculation is just 
referred to as the `combined algorithm`. The use of the combined algorithm produces a fit to the 'lag-times' with an 
R-squared ("coefficient of determination") of ... (R-squared of 1.0 is a perfect fit).

Given the stochastic nature of fibrillogenesis, values are likely to be specific, not only to proteins that are 
natively unfolded, and not only to Synuclein-sized proteins, but also for the specific experimental conditions of the 
assays as well. Hence I specify these conditions alongside the code and mathematical derivations: 
'Lag-times' of fibrillogenesis of Synuclein constructs was based on experiments under the following conditions: 
- Protein concentration at 400 microMolars in 30 mM MOPs buffer pH 7.2 and 10 microMolar ThT. 
- Incubation at 37degC, shaken at 450 rpm. 
- 10 microlitre samples collect at 5 subsequent time points 4h, 8h, 24h, 48h, 96h. All measurements were performed on 
each sample immediately after collection.
'Lag-times' were measured from each experiment, as time taken for ThT values to reach the square of their zero
time emission readings at 480nm. The final value is the mean of the 'lag-times' for each constructs.

A very important caveat is that for certain Synuclein constructs for which 'lag-time' data was included, 
which had longer 'lag-times', those proteins did not begin to assemble within the 96 hours limit. Hence, 
the data from these experiments was not included. As such their included data indicates these proteins to be more 
fibrillogenic than they actually were taking in to account the non-inclusion of the aforementioned experiments. 
"""


def train_combo_model(syns_lags_seqs_props: pDF, make_plot: bool) -> Tuple[LinearRegression, float]:
    """
    Fit linear regression of given log of 'lag-times' to the given combination of physicochemical properties.
    (`sklearn.linear_model.LinearRegression` uses Ordinary Least Squares rather than gradient descent).
    :param syns_lags_seqs_props: Table of 4 columns including Synucleins (index) mapped to 'lag-times',
    log of 'lag-times', amino acid sequences and mean properties.
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties
    against the natural log of 'lag-times'.
    :return: Trained linear regression model of log of 'lag-times' against combined properties anf the corresponding
    'coefficient of determination'.
    """
    x_combo = np.array(syns_lags_seqs_props.combo)
    x_combo = x_combo.reshape((-1, 1))
    y = np.array(syns_lags_seqs_props.ln_lags)
    model = LinearRegression()
    model.fit(x_combo, y)
    rsq = round(float(model.score(x_combo, y)), 3)
    if make_plot: plotter.plot(model, x=x_combo, y=y, data_labels=list(syns_lags_seqs_props.index),
                               title='combo', x_label='combined properties')
    return model, rsq


def _calc_relative_weights_for_nmprops_and_nbsc(syns_lags_seqs_props: pDF, make_plot: bool) -> Tuple[dict, dict, dict]:
    """
    Calculate the weights for the two given properties for the synucleins: normalised mprops (nmprops) and normalised
    integrals of beta-strand contiguity (nbSC).
    :param syns_lags_seqs_props:
    :param make_plot: True to display plot.
    :return: The weights for the nmprops and nbSC, as well as the intercepts and coefficients of determination for the
    two linear regressions.
    """
    x_nbsc, x_nmprops = np.array(syns_lags_seqs_props.nbsc), np.array(syns_lags_seqs_props.nmprops)
    x_nbsc, x_nmprops = x_nbsc.reshape((-1, 1)), x_nmprops.reshape((-1, 1))
    y = np.array(syns_lags_seqs_props.ln_lags)
    _2coefs, _2intcpts, _2rsq = {}, {}, {}
    for prop_name, prop_values in zip(['nbsc', 'nmprops'], [x_nbsc, x_nmprops]):
        model = LinearRegression()
        model.fit(prop_values, y)
        if make_plot:
            plotter.plot(model, x=prop_values, y=y, data_labels=list(syns_lags_seqs_props.index), x_label=prop_name,
                         title=prop_name)
        _2coefs[prop_name] = round(float(model.coef_), 3)
        _2intcpts[prop_name] = round(float(model.intercept_), 3)
        _2rsq[prop_name] = round(float(model.score(prop_values, y)), 3)
    print(f'_2rsq {_2rsq}')
    return _2coefs, _2intcpts, _2rsq


def generate_combo(lagtime_means_csv_filename: str, syns_lnlags_seqs=None, make_plot: bool = True) -> pDF:
    """
    Predict lag time hours for all synucleins based on mean properties and beta-strand contiguity trained with
    observed lag times. Optionally display plots for each of the properties.
    :param lagtime_means_csv_filename: Name of 'lag-time' means csv filename (including csv extension).
    :param syns_lnlags_seqs: Synuclein names, ln of lag times and sequences.
    :param make_plot: True to display 4 plots of the linear regression for each of the 4 mean properties, for a plot
    of mprops, a plot for beta-strand contiguity integrals and finally a plot of the combination of mprops and bSC.
    All of the plots are against the natural log of 'lag-times' on the y-axes.
    :return: Table of 8 columns including Synucleins as index, 'lag-time' means, log of 'lag-times',
    normalised mean properties, amino acid sequences, beta-strand contiguity, normalised beta-strand contiguity and
    combination algorithm values: [(index), 'lagtime_means', 'ln_lags', 'nmprops', 'seqs', 'bsc', 'nbsc', 'combo'].
    """
    if syns_lnlags_seqs is None:
        syns_lnlags_seqs = utils.get_ln_lags_and_build_seqs(lagtime_means_csv_filename)
    syns_lags_seqs_props = mprops.compute_norm_mprops(syns_lnlags_seqs, make_plot)

    syns_lags_seqs_props = salsa.compute_norm_bsc_integrals(syns_lags_seqs_props)
    _2coefs, _2intcpts, _2rsq = _calc_relative_weights_for_nmprops_and_nbsc(syns_lags_seqs_props, make_plot=make_plot)
    syns_lags_seqs_props_ = syns_lags_seqs_props.copy()
    syns_lags_seqs_props_.loc[:, 'combo'] = syns_lags_seqs_props.\
        apply(lambda row: row.nbsc * _2coefs['nbsc'] + row.nmprops * _2coefs['nmprops'], axis=1)
    return syns_lags_seqs_props_


if __name__ == '__main__':
    import time
    st = time.time()
    df = generate_combo(lagtime_means_csv_filename='lagtime_means_polynDegree_4_lagtimeEndvalue_16.csv', make_plot=True)
    print(f'time {round(time.time() - st, 2)} secs')
