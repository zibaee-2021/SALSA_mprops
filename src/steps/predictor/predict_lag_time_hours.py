import os
import numpy as np
from src.utils import utils
from src.combo import mprops_bsc_combo
from root_path import abspath_root
from pandas import DataFrame as pDF
# from fastapi import FastAPI
# import nest_asyncio
# import uvicorn
# app = FastAPI(title='SALSA_mprops estimator')
# nest_asyncio.apply()
# host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"
# uvicorn.run(app, host=host, port=8000)
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


def fit_and_predict_syns(lagtime_means_csv_filename: str, make_plots: bool) -> pDF:
    """
    Generate the combo algorithm (from scratch) by fitting all 4 mean properties, mprops and beta-strand contiguity
    to the log of lag times for synucleins.
    :param lagtime_means_csv_filename: Name of 'lag-time' means csv filename (including csv extension).
    :return: Synucleins (index) mapped to observed lag time means (hours) and predicted lag times (hours).
    ['lagtime_means', 'pred']
    """
    syns_lnlags_seqs = utils.get_ln_lags_and_build_seqs(lagtime_means_csv_filename)
    syns_lags_seqs_props = mprops_bsc_combo.generate_combo(lagtime_means_csv_filename=lagtime_means_csv_filename,
                                                           syns_lnlags_seqs=syns_lnlags_seqs, make_plot=make_plots)
    combo_model, rsq = mprops_bsc_combo.train_combo_model(syns_lags_seqs_props, make_plot=make_plots)
    coef = round(float(combo_model.coef_), 3)
    intcpt = round(float(combo_model.intercept_), 3)
    lagtime_means_csv_filename = lagtime_means_csv_filename.replace('lagtime_means_polynDegree', 'polydeg')
    lagtime_means_csv_filename = lagtime_means_csv_filename.replace('lagtimeEndvalue_', 'lgEnd')
    print(f'{lagtime_means_csv_filename}: combo_model coef {coef}, combo_model_intercept {intcpt}, rsq {rsq}')
    preds = combo_model.predict(np.array(syns_lags_seqs_props['combo']).reshape(-1, 1))
    syns_lags_seqs_props['pred'] = np.exp(preds)
    return syns_lags_seqs_props[['lagtime_means', 'pred']]


if __name__ == '__main__':

    lagtime_dir_path = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes')
    b_lagtime_dir_path = os.fsencode(lagtime_dir_path)
    syns_lnlags_seqs = None

    for b_csv_filename in os.listdir(b_lagtime_dir_path):
        filename = os.fsdecode(b_csv_filename)
        if filename.endswith(".csv"):
            syns_lnlags_seqs = fit_and_predict_syns(lagtime_means_csv_filename=filename, make_plots=False)

    print(syns_lnlags_seqs.head(50))

