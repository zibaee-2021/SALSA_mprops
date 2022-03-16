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

STARTING_THT_VALUE = 4.0
SQUARE_OF_STARTING_VALUE = np.square(STARTING_THT_VALUE)
DOUBLE_STARTING_VALUE = STARTING_THT_VALUE * 2
MIN_NUM_OF_LAG_TIMES_NEEDED = 3


def _calc_mean_of_lag_times(lag_times: dict[str: List]) -> dict[str: float]:
    """
    Calculate the mean of the 'lag times' of each Synuclein. It is expected that this function is called only on the
    cleaned 'lag times', such that only numerical values are included (`NA` is already removed) and Synucleins with
    less than 3 'lag times' are also removed.
    :param lag_times: 'Lag times' of Synucleins with 3 or more values and without any `NA`, mapped to the Synuclein
    name.
    :return: Mean of the given 'lag times' mapped to the corresponding Synuclein name.
    """
    return {syn: round(np.mean(lags), 1) for syn, lags in lag_times.items()}


def _process_na(lag_times: dict[str: List]) -> dict[str: List]:
    """
    Remove `NA` from the 'lag times' of each Synuclein. Remove the Synuclein entirely if it is then left with less
    than the minimum number of 'lag times' deemed sufficient for including in the 'lag times' dataset, currently 3.
    Replace all hyphens with underscores in Synuclein names, necessary for mprops module.
    :param lag_times: 'Lag times' for each Synuclein including `NA` (indicating no filament assembly within 100 h).
    :return: 'Lag times' of Synucleins that have the minimum number of 'lag times', 3 or more.
    """
    processed_lag_times = {}
    for syn, lags in lag_times.items():
        lags_ = [lag for lag in lags if lag != 'NA']
        processed_lag_times[syn] = lags_

    return {syn.replace('-', '_'): lags for syn, lags in processed_lag_times.items() if len(lags) >=
            MIN_NUM_OF_LAG_TIMES_NEEDED}


def clean_and_mean(lag_times: dict[str: List]) -> dict[str: float]:
    """
    Take the mean of the 'lag times' for each Synuclein that has enough data - that is if 'lag times' from 3 or more
    experiments' have a numerical value.
    :param lag_times: Time taken for fluorescence emission to reach the square of the value at zero time.
    :return:
    """
    return _calc_mean_of_lag_times(_process_na(lag_times))


def _solve_for_y(poly_coefs, intercept, y) -> float:
    """
    Solve the given polynomial function for the given intercept and
    Find the time value corresponding to the given ThT value (expected to be the square of the starting ThT value).
    As the model is created from a polynomial, the roots of the function are required to solve.
    E.g. If degree of polynomial used is 3, function will be: y = m3 * x^3 + m2 * x^2 + m1 + x + c, where `m` is the
    coefficient of each x value. `c` is the y-intercept. Hence to solve using the roots, the equation is translated
    along the ordinate so that y = 0, such that 0 = m3 * x^3 + m2 * x^2 + m1 + x + c. Therefore, to solve for the y
    value of 16.0, the function is subtracted by this value.
    Note numpy.roots expects the order of the equation to be as shown above - starting with higher order and ending
    with the intercept. However, scikit-learn's model.coef_ retuen the coefficient the other way round (hence I
    `flip` the order) and it lacks the intercept value, so I add this to the end of the flipped array,
    before subtracting the function by the 16.0.)
    :param poly_coefs: Coefficients of the polynomial function used to model the data.
    :param intercept: The y-intercept of the function, which is the translated starting ThT value, 4.0.
    :param y: The ThT value that we want to solve the x (time) for. This is the ThT value deemed to mark the end of
    the 'lag time' (assigned to variable called `tht_lagtime_end_value` in other functions in this module).
    This is currently a choice of either double (hence 8) or the square (hence 16) of the starting value.
    :return: The abscissa solution for the given ordinate value
    """
    pc = poly_coefs.copy()
    pc_ = np.flip(pc, axis=0)
    pc_[-1] += intercept
    pc_[-1] -= y
    roots = np.roots(pc_)
    return roots[0]


def _plot(preproc, lin_reg, x, y, y_pred, syn_name):
    """
    Display a plot of the polynomial regression, with the given time points as abscissa. The ordinate includes both
    the ThT values and the predicted values with the regression line.
    :param preproc: Preprocessing method used. (For example the polynomial, according to a specified degree).
    :param lin_reg: The linear regression model, already trained only polynomially processed time point values.
    :param x: Time points (expected to be only two time points).
    :param y: ThT values (expected to be only two ThT values).
    :param y_pred: Predicted ThT value according to the trained polynomial regression model.
    :param syn_name: The Synuclein name.
    """
    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_pred, color='green')
    plt.plot(X_grid, lin_reg.predict(preproc.fit_transform(X_grid)), color='black')
    plt.title(f'Polynomial Regression for {syn_name}')
    plt.xlabel('Time')
    plt.ylabel('ThT (h)')
    plt.show()


def _calculate_lag_times(syn_name: str, two_time_points: list, two_tht_values: list, lag_times: dict, make_plot: bool,
                         degree_to_use: int, tht_lagtime_end_value: float) -> dict[str: List]:
    """
    Calculate the 'lag time' as the time at which the square of the starting value is reached, from the given ThT
    values which are expected to only include values that reach this otherwise they are None. Use a polynomial
    regression with the given degree and map all calculated values to the given Synuclein name. (The implementation
    for assigning values to variables `x` and `y` includes filtering out empty rows which were present in an older
    version but are left in place as it is possible that I may alter the upstream functionality so that all the ThT
    values for the experiment are passed here.)
    :param syn_name: Synuclein name. Examples: `asyn`, `a11-140`, `a1-80`, `ba1`, `fr_asyn`, `b45V46V`, `a68-71Del`.
    :param two_time_points: Time point immediately before and time point immediately after end of 'lag time'.
    :param two_tht_values: ThT value immediately before and ThT value immediately after end of 'lag time'.
    :param lag_times: All 'Lag times' calculated thus far, to which the 'lag time' being calculated here is added.
    :param make_plot: True to display a plot of the polynomial regression used.
    :param degree_to_use: The degree to use for the polynomial regression.
    :param tht_lagtime_end_value: The ThT value that is deemed to be the end of the 'lag time'. Currently a choice of
    either double (8.0) the starting value or the square (16.0) of the starting value.
    :return: All 'lag times', mapped to the corresponding Synuclein name.
    """
    if two_tht_values is None:
        lag_hours = 'NA'
    else:
        x = np.array([num for i, num in enumerate(two_time_points) if i == 0 or num != 0.0]).reshape((-1, 1))
        y = np.array(two_tht_values[0: len(x)])
        poly_reg = PolynomialFeatures(degree=degree_to_use)
        x_poly = poly_reg.fit_transform(x)
        lin_reg = LinearRegression()
        lin_reg.fit(x_poly, y)
        if make_plot:
            y_pred = lin_reg.predict(x_poly)
            _plot(preproc=poly_reg, lin_reg=lin_reg, x=x, y=y, y_pred=y_pred, syn_name=syn_name)
        lag_hours = np.real(_solve_for_y(lin_reg.coef_, STARTING_THT_VALUE, tht_lagtime_end_value))
        lag_hours = abs(round(float(lag_hours), 1))
    if syn_name not in lag_times:
        lag_times[syn_name] = [lag_hours]
    else:
        lag_times[syn_name].append(lag_hours)
    return lag_times


def _include_lag_phase_only(time_points: np.array, tht_values: np.array) -> Tuple[np.array, np.array]:
    """
    Find the first time point that the ThT fluorescence emission has reach and/or exceeded the square of the starting
    value and return this and the time point immediately before it, as well as the corresponding ThT values.
    :param time_points: All time points for a particular experiment (including any additional empty rows after the
    final time point). `time_points` will typically look like: `[ 0, 4, 8, 24, 48, 96,  0,  0,  0,]`.
    :param tht_values: All ThT fluorescence data for a particular experiment (including any additional empty rows after
    the final time point). The expected data has already been translated to a starting value of 4.0.
    `tht_values` might look something like: `[4, 4, 5, 20, 150, 250, 0, 0, 0]`, corresponding to the `time_points`
    example shown.
    :return: Two consecutive time points including the time point immediately before and the time point
    immediately after the ThT value has reached the square of the starting value.
    """
    for i, tht_value in enumerate(tht_values):
        if tht_value >= SQUARE_OF_STARTING_VALUE:
            return time_points[i - 1: i + 1], tht_values[i - 1: i + 1]
    return None, None


def _is_date(row_col1: str) -> bool:
    """
    Determines if given data is date, expecting the format to be 01.01.01
    Accepts this date with any prefix and/or suffix.
    :param row_col1: Input to be evaluated.
    :return: True if the given data has the expected date format.
    """
    return re.match(re.compile(r'\d{2}[.]\d{2}[.]\d{2}'), row_col1) is not None


def _get_data() -> List[str]:
    csvpath = os.path.join(abspath_root, 'data', 'tht_data', 'translatedTo4.csv')
    with open(csvpath, 'r') as csv_:
        for row in csv.reader(csv_):
            yield row


def get_lag_times(make_plot: bool, degree_to_use: int, tht_lagtime_end_value: float) -> dict[str: List]:
    """
    Read `translatedTo4.csv` file and determine the time taken for each protein's ThT fluorescence to reach the
    square of its starting value, which has been translated to 4.0. This is referred to here as a "lag time".
    The file contains all ThT data for filament-forming Synucleins measured between 01.11.01 and 30.10.09.
    It is expected to have a particular format, starting a date. This is immediately followed in the next
    row down by `Time (h)` in the first column and the names of the Synuclein constructs in the adjoining columns.
    The subsequent time points at which measurements were taken and the corresponding ThT fluorescence for each
    construct follow in the rows below.
    :param make_plot: True to generate a plot of the region of the ThT fluorescence curve that includes the value by which
    the 'lag time' is deemed to have ended.
    :param degree_to_use: The degree to use for the polynomial regression of the ThT data.
    :param tht_lagtime_end_value: The ThT value that is deemed to be the end of the 'lag time'. Currently a choice of
    either double (8.0) the starting value or the square (16.0) of the starting value.
    :return: Each Synuclein name mapped to its 'lag times'.
    """
    syn_names = ()
    time_points = np.zeros(10)
    tht_values = None
    lag_times = {}

    data_ = iter(_get_data())

    i = 0
    for row in data_:
        if _is_date(row[0]):
            continue
        elif row[0] == 'Time (h)':
            syn_names += tuple(filter(None, row[1:]))
            tht_values = np.zeros((10, len(syn_names)))
        elif row[0] == '0' or row[0] == '0.0':
            i = 0
            tht_values[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time_points[i] = row[0]
        elif row[0] == '':
            for i, syn_name in enumerate(syn_names):
                two_time_points, two_tht_values = _include_lag_phase_only(time_points=time_points,
                                                                          tht_values=tht_values[:, i])
                lag_times = _calculate_lag_times(syn_name=syn_name, two_time_points=two_time_points,
                                                 two_tht_values=two_tht_values, lag_times=lag_times,
                                                 make_plot=make_plot, degree_to_use=degree_to_use,
                                                 tht_lagtime_end_value=tht_lagtime_end_value)
            syn_names = ()
        else:
            tht_values[i, ] = [float(n) if n != '' else 0.0 for n in row[1: len(syn_names) + 1]]
            time_points[i] = row[0]
        i += 1
    return lag_times


def write_lag_time_means(lag_time_means: dict[str: float], degree_used: int, tht_lagtime_end_value_used: float):

    col_names = ['Synucleins', 'lag_time_means']
    df = pd.DataFrame.from_dict(data=lag_time_means, orient='index', columns=[col_names[1]])
    lag_time_filename = f'lag_time_Degree_{degree_used}_End_value_{tht_lagtime_end_value_used}.csv'
    lag_time_csv = os.path.join(abspath_root, 'data', 'tht_data', lag_time_filename)
    df.to_csv(lag_time_csv, index=True)


if __name__ == '__main__':
    # for degree in [1, 2, 3, 4, 5, 6]:
    for degree in [4]:
        for tht_lagtime_end_value_ in [SQUARE_OF_STARTING_VALUE, DOUBLE_STARTING_VALUE]:
            write_lag_time_means(clean_and_mean(
                get_lag_times(make_plot=False, degree_to_use=degree, tht_lagtime_end_value=tht_lagtime_end_value_)),
                degree_used=degree, tht_lagtime_end_value_used=tht_lagtime_end_value_)
