import numpy as np
from src.utils import utils
from src.combo import mprops_bsc_combo


def fit_and_predict_syns():
    """
    Generate the combo algorithm (from scratch) by fitting all 4 mean properties, mprops and beta-strand contiguity
    to the log of lag times for synucleins.
    :return: Synucleins (index) mapped to observed lag time means (hours) and predicted lag times (hours).
    ['lag_time_means', 'pred']
    """
    syns_lnlags_seqs = utils.get_ln_lags_and_build_seqs()
    syns_lags_seqs_props = mprops_bsc_combo.generate_combo(syns_lnlags_seqs, make_plot=True)
    combo_model, rsq = mprops_bsc_combo.train_combo_model(syns_lags_seqs_props, make_plot=True)
    coef = round(float(combo_model.coef_), 3)
    intcpt = round(float(combo_model.intercept_), 3)
    print(f'combo_model coef {coef}, combo_model_intercept {intcpt}, rsq {rsq}')
    preds = combo_model.predict(np.array(syns_lags_seqs_props['combo']).reshape(-1, 1))
    syns_lags_seqs_props['pred'] = np.exp(preds)
    return syns_lags_seqs_props[['lag_time_means', 'pred']]


if __name__ == '__main__':
    syns_lnlags_seqs = fit_and_predict_syns()
    print(syns_lnlags_seqs.head(50))

