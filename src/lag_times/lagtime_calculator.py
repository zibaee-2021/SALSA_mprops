import math
import os
import csv
import numpy as np
from root_path import abspath_root
from typing import Tuple
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _solve_for_y(poly_coeffs, intercept, y):
    pc = poly_coeffs.copy()
    pc_ = np.flip(pc, axis=0)
    pc_[-1] += intercept
    pc_[-1] -= y
    roots = np.roots(pc_)
    return roots[0]


def _plot(preproc, lin_reg, X, y, y_pred, name):
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color='red')
    plt.scatter(X, y_pred, color='green')
    plt.plot(X_grid, lin_reg.predict(preproc.fit_transform(X_grid)), color='black')
    plt.title(f'Polynomial Regression for {name}')
    plt.xlabel('Time')
    plt.ylabel('ThT (h)')
    plt.show()


def _calculate_lag_times(name, time, tht, lagtimes, plot):
    if tht is None:
        lag_hours = 'NA'
    else:
        X = np.array([num for i, num in enumerate(time) if i == 0 or num != 0.0]).reshape((-1, 1))
        y = np.array(tht[0: len(X)])
        poly_reg = PolynomialFeatures(degree=6)
        X_poly = poly_reg.fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        if plot:
            y_pred = lin_reg.predict(X_poly)
            _plot(preproc=poly_reg, lin_reg=lin_reg, X=X, y=y, y_pred=y_pred, name=name)
        lag_hours = np.real(_solve_for_y(lin_reg.coef_, 4.0, 16.0))
        lag_hours = abs(round(float(lag_hours), 1))
    if name not in lagtimes:
        lagtimes[name] = [lag_hours]
    else:
        lagtimes[name].append(lag_hours)
    return lagtimes


def _include_lag_phase_only(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    for i, tht in enumerate(y):
        if tht >= 16.0:
            return X[i - 1: i + 1], y[i - 1: i + 1]
    return None, None


def _is_date(row_col1: str):
    """
    Determines if given data is date, expecting the format to be 01.01.01
    Accepts this date with any prefix and/or suffix.
    :param row_col1: Input to be evaluated.
    :return: True if the given data has the expected date format.
    """
    return re.match(re.compile(r'\d{2}[.]\d{2}[.]\d{2}'), row_col1) is not None


def _get_data():
    csvpath = os.path.join(abspath_root, 'data', 'tht_data', 'translatedTo4.csv')
    with open(csvpath, 'r') as csv_:
        for row in csv.reader(csv_):
            yield row


def get_lagtimes():
    """
    Read `translatedTo4.csv` file and determine the time taken for each protein's ThT fluorescence to reach the
    square of its starting value, which has been translated to 4.0. This is referred to here as a "lag time".
    The file contains all ThT data for filament-forming synucleins measured between 01.11.01 and 30.10.09.
    It is expected to have a particular format, starting a date. This is immediately followed in the next
    row down by `Time (h)` in the first column and the names of the synuclein constructs in the adjoining columns.
    The subsequent time points at which measurements were taken and the corresponding ThT fluorescence for each
    construct follow in the rows below.
    :return: "Lag times"
    """
    tht_date = ''
    syn_names = ()
    time = np.zeros(10)
    tht = None
    lagtimes = {}

    data_ = iter(_get_data())

    i = 0
    for row in data_:
        if _is_date(row[0]):
            tht_date = row[0]
            print(tht_date)
        elif row[0] == 'Time (h)':
            syn_names += tuple(filter(None, row[1:]))
            tht = np.zeros((10, len(syn_names)))
        elif row[0] == '0' or row[0] == '0.0':
            i = 0
            tht[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time[i] = row[0]
        elif row[0] == '':
            for i, name in enumerate(syn_names):
                time_, tht_ = _include_lag_phase_only(X=time, y=tht[:, i])
                lagtimes = _calculate_lag_times(name=name, time=time_, tht=tht_, lagtimes=lagtimes, plot=False)
            syn_names = ()
        else:
            tht[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time[i] = row[0]
        i += 1
    return lagtimes


if __name__ == '__main__':
    print(get_lagtimes())
