import os
import numpy as np
from src.utils import utils
from src.combo import mprops_bsc_combo
from root_path import abspath_root
from pandas import DataFrame as pDF
from sklearn import linear_model
from sklearn.linear_model import LinearRegression




def fit_and_predict_syns(csv_filename: str, make_plots: bool, model: linear_model) -> pDF:
    """
    Generate the combination algorithm, from scratch, by fitting all 4 mean properties, mprops and beta-strand
    contiguity to the log of lagtimes for synucleins.
    :param csv_filename: Name of lagtime means csv filename (including csv extension).
    :param make_plots: True to make plots in IDE.
    :param model: Linear model to use for fitting combination of properties to log of lagtimes.
    :return: Table of 5 columns including synucleins as index, lagtime means, predicted lagtimes, normalised
    mean properties and normalised beta-strand contiguity: [(index), 'lagtime_means', 'pred', 'nmprops', 'nbsc']]
    """
    syns_lagmeans_lnlags_seqs = utils.get_loglags_and_build_seqs(csv_filename)
    syns_lags_seqs_props = mprops_bsc_combo.generate_combo(csv_filename=csv_filename, model=model,
                                                           pdf=syns_lagmeans_lnlags_seqs, make_plot=make_plots)
    combo_model, rsq = mprops_bsc_combo.train_combo_model(syns_lags_seqs_props, make_plots, model=model)
    coef = round(float(combo_model.coef_), 3)
    intcpt = round(float(combo_model.intercept_), 3)
    print(f'{csv_filename}: combo_model coef {coef}, combo_model_intercept {intcpt}, rsq {rsq}')
    preds = combo_model.predict(np.array(syns_lags_seqs_props['combo']).reshape(-1, 1))
    syns_lags_seqs_props['pred'] = np.exp(preds)
    return syns_lags_seqs_props[['lagtime_means', 'pred', 'nmprops', 'nbsc']]


if __name__ == '__main__':

    lagtime_dir_path = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes', 'lagtime_means')
    b_lagtime_dir_path = os.fsencode(lagtime_dir_path)
    syns_lnlags_seqs = None
    models_to_try = [LinearRegression()]

    for b_csv_filename in os.listdir(b_lagtime_dir_path):
        filename = os.fsdecode(b_csv_filename)
        if filename.endswith('.csv'):
            for model_ in models_to_try:
                syns_lnlags_seqs = fit_and_predict_syns(csv_filename=filename, make_plots=True, model=model_)
                print(syns_lnlags_seqs.head(50))

