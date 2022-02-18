import os
import csv
import numpy as np
import pandas as pd
from root_path import abspath_root
from typing import Tuple, List
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _calc_mean_of_lagtimes(lagtimes: dict[str: List]) -> dict[str: float]:
    """
    Calculate the mean of the 'lag times' of each synuclein. It is expected that this function is called only on the
    cleaned 'lag times', such that only numerical values are included (`NA` is already removed) and synucleins with
    less than 3 lag times are also removed.
    :param lagtimes: 'Lag times' of synucleins with 3 or more values and without any `NA`, mapped to the synuclein name.
    :return: Mean of the given 'lag times' mapped to the synuclein name.
    """
    return {syn: round(np.mean(lags), 1) for syn, lags in lagtimes.items()}


def _process_na(lagtimes: dict[str: List]) -> dict[str: List]:
    """
    Remove NAs from the lag times of each synuclein. If the synuclein is left with less than 3 lag times, it is
    removed from dataset.
    :param lagtimes: Lag times for each synuclein including `NA` (indicating no filament assembly within 100 h).
    :return: Lag times of synucleins that have 3 or more lag times.
    """
    lagtimes_ = {}
    for syn, lags in lagtimes.items():
        lags_ = [lag for lag in lags if lag != 'NA']
        lagtimes_[syn] = lags_
    return {syn: lags for syn, lags in lagtimes_.items() if len(lags) >= 3}


def clean_and_mean(lagtimes):
    return _calc_mean_of_lagtimes(_process_na(lagtimes))


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


def _calculate_lag_times(name: str, time: list, tht: list, lagtimes: dict, plot: bool, degree: int):
    if tht is None:
        lag_hours = 'NA'
    else:
        x = np.array([num for i, num in enumerate(time) if i == 0 or num != 0.0]).reshape((-1, 1))
        y = np.array(tht[0: len(x)])
        poly_reg = PolynomialFeatures(degree=degree)
        X_poly = poly_reg.fit_transform(x)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        if plot:
            y_pred = lin_reg.predict(X_poly)
            _plot(preproc=poly_reg, lin_reg=lin_reg, X=x, y=y, y_pred=y_pred, name=name)
        lag_hours = np.real(_solve_for_y(lin_reg.coef_, 4.0, 16.0))
        lag_hours = abs(round(float(lag_hours), 1))
    if name not in lagtimes:
        lagtimes[name] = [lag_hours]
    else:
        lagtimes[name].append(lag_hours)
    return lagtimes


def _include_lag_phase_only(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    for i, tht in enumerate(y):
        if tht >= 16.0:
            return x[i - 1: i + 1], y[i - 1: i + 1]
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


def get_lagtimes(plot: bool, degree: int) -> dict[str: List]:
    """
    Read `translatedTo4.csv` file and determine the time taken for each protein's ThT fluorescence to reach the
    square of its starting value, which has been translated to 4.0. This is referred to here as a "lag time".
    The file contains all ThT data for filament-forming synucleins measured between 01.11.01 and 30.10.09.
    It is expected to have a particular format, starting a date. This is immediately followed in the next
    row down by `Time (h)` in the first column and the names of the synuclein constructs in the adjoining columns.
    The subsequent time points at which measurements were taken and the corresponding ThT fluorescence for each
    construct follow in the rows below.
    :param plot: True to generate a plot of the region of the ThT fluorescence curve that includes the value by which
    the 'lag time' is deemed to have ended.
    :param degree: The degree to use for the polynomial regression of the ThT fluorescence data.
    :return: Each synuclein name mapped to its 'lag times'.
    """
    syn_names = ()
    time = np.zeros(10)
    tht = None
    lagtimes = {}

    data_ = iter(_get_data())

    i = 0
    for row in data_:
        if _is_date(row[0]):
            continue
        elif row[0] == 'Time (h)':
            syn_names += tuple(filter(None, row[1:]))
            tht = np.zeros((10, len(syn_names)))
        elif row[0] == '0' or row[0] == '0.0':
            i = 0
            tht[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time[i] = row[0]
        elif row[0] == '':
            for i, name in enumerate(syn_names):
                time_, tht_ = _include_lag_phase_only(x=time, y=tht[:, i])
                lagtimes = _calculate_lag_times(name=name, time=time_, tht=tht_,
                                                lagtimes=lagtimes, plot=plot, degree=degree)
            syn_names = ()
        else:
            tht[i, ] = [float(n) if n != '' else 0.0 for n in row[1: len(syn_names) + 1]]
            time[i] = row[0]
        i += 1
    return lagtimes


def write_lag_time_means(lag_time_means: dict[str: float]):
    col_names = ['Synucleins', 'lagtime_means']
    df = pd.DataFrame.from_dict(data=lag_time_means, orient='index', columns=[col_names[1]])
    df[col_names[0]] = df.index
    df = df[col_names]
    df.reset_index(drop=True, inplace=True)

    lagtime_csv = os.path.join(abspath_root, 'data', 'tht_data', 'lagtime_means.csv')
    df.to_csv(lagtime_csv, index=False)


if __name__ == '__main__':
    _2 = clean_and_mean(get_lagtimes(plot=False, degree=2))
    print(f'degree 2: {_2}')
    _3 = clean_and_mean(get_lagtimes(plot=False, degree=3))
    print(f'degree 3: {_3}')
    _4 = clean_and_mean(get_lagtimes(plot=False, degree=4))
    print(f'degree 4: {_4}')
    _5 = clean_and_mean(get_lagtimes(plot=False, degree=5))
    print(f'degree 5: {_5}')
    _6 = clean_and_mean(get_lagtimes(plot=False, degree=6))
    print(f'degree 6: {_6}')
