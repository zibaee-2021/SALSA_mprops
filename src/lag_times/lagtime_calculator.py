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


def _get_data():
    csvpath = os.path.join(abspath_root, 'data', 'tht_data', 'translatedTo4.csv')
    with open(csvpath, 'r') as csv_:
        for row in csv.reader(csv_):
            yield row


def get_lagtimes():
    tht_date = ''
    syn_names = ()
    time = np.zeros(10)
    tht = None
    lagtimes = {}

    data_ = iter(_get_data())

    i = 0
    for line, row in enumerate(data_):
        print(f'line {line}')
        if is_date(row[0]):
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
                lagtimes = calculate_lag_times(name=name, time=time_, tht=tht_, lagtimes=lagtimes)
            syn_names = ()
        else:
            tht[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time[i] = row[0]
        i += 1
    return lagtimes


def _include_lag_phase_only(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    for i, tht in enumerate(y):
        if tht >= 16.0:
            return X[i - 1: i + 1], y[i - 1: i + 1]
    return None, None


def _solve_for_y(poly_coeffs, intercept, y):
    pc = poly_coeffs.copy()
    pc_ = np.flip(pc, axis=0)
    pc_[-1] += intercept
    pc_[-1] -= y
    roots = np.roots(pc_)
    return roots[0]


def calculate_lag_times(name, time, tht, lagtimes):
    if tht is None:
        lag_hours = 'NA'
    else:
        X = np.array([num for i, num in enumerate(time) if i == 0 or num != 0.0]).reshape((-1, 1))
        y = np.array(tht[0: len(X)])
        poly_reg = PolynomialFeatures(degree=6)
        X_poly = poly_reg.fit_transform(X)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        # y_pred = lin_reg.predict(X_poly)
        # _plot(preproc=poly_reg, lin_reg=lin_reg, X=X, y=y, y_pred=y_pred, name=name)
        lag_hours = np.real(_solve_for_y(lin_reg.coef_, 4.0, 16.0))
        lag_hours = abs(round(float(lag_hours), 1))
    if name not in lagtimes:
        lagtimes[name] = [lag_hours]
    else:
        lagtimes[name].append(lag_hours)
    return lagtimes


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


def get_syn_names(row) -> Tuple:
    syn_names = ()
    if row[0] == 'Time (h)':
        for i in range(10):
            if row[i] == '':
                return syn_names
            else:
                syn_names += tuple(row[i])
    return syn_names


def is_date(row_col1):
    return re.match(re.compile(r'\d{2}[.]\d{2}[.]\d{2}'), row_col1) is not None


if __name__ == '__main__':
    print(get_lagtimes())
