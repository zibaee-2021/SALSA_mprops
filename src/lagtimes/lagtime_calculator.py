import os
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame as pDF
from root_path import abspath_root
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.utils import util_data

STARTING_THT_VALUE = 4.0
SQUARE_OF_STARTING_VALUE = np.square(STARTING_THT_VALUE)
DOUBLE_STARTING_VALUE = STARTING_THT_VALUE * 2
_1_5_STARTING_VALUE = STARTING_THT_VALUE * 1.5
MIN_NUM_OF_LAGTIMES_NEEDED = 3
STR_TIME_H = 'Time (h)'
ALL_THT_DATA_CSV_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.csv')
LAGTIMES_PATH = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes')
LAGTIME_MEANS_PATH = os.path.join(LAGTIMES_PATH, 'lagtime_means')
THT_PATH = os.path.join(abspath_root, 'data', 'tht_data')
ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES = 1 / 8
STANDARD_ASYN_END_THT_VALUE = 250


def _has_enough_nonnull_lagtimes(lagtimes: list) -> bool:
    """
    Checks that the number of 'lag-times' in the given list is above the globally-defined value, (typically 3).
    The given list of 'lag-times' may have already had all nulls removed, but this function can still handle a
    null-containing list.
    :param lagtimes: 'Lag-times'.
    :return: True if the given list has at least the minimum number of non-null 'lag-times'.
    """
    has_enough_nonnull_lagtimes = len([lagtime for lagtime in lagtimes if lagtime != 'NA']) >= \
                                  MIN_NUM_OF_LAGTIMES_NEEDED
    return has_enough_nonnull_lagtimes


def _has_high_enough_proportion_of_non_null_lagtimes(lagtimes: list) -> bool:
    """
    Checks that the number of 'lag-times' in the given list has a sufficient proportion of non-null to null values.
    (Typically it is approximated that there should be more than 7 non-null in 8 'lag-time' values, otherwise the
    subsequent mean value is not reliably representative of the Synuclein's fibrillogenic propensity and is removed
    from the subsequent linear regression-based predictions. Note, such Synuclein datasets will still be valuable
    for classification-based predictions).
    :param lagtimes: 'Lag-times' that have not had nulls previously removed.
    :return: True if the given list has at least the minimum proportion of non-null to null values.
    """
    num_of_nones = len([lag for lag in lagtimes if lag == 'NA'])
    has_enough_prop_of_lagtimes = (num_of_nones / len(lagtimes)) <= ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES
    return has_enough_prop_of_lagtimes


def _remove_nulls(lagtimes: list) -> list:
    """
    Remove the value representing null within the given list of 'lag-times'.
    (Currently, it is represented by the string 'NA', although it may change without this comment being updated).
    :param lagtimes: 'Lag-times', potentially containing null values.
    :return: Non-null 'lag-times' only.
    """
    nonnulls = [lag for lag in lagtimes if lag != 'NA']
    return nonnulls


def _calc_mean_of_nonnulls(lagtimes: list) -> Tuple[float, float]:
    lagtimes = _remove_nulls(lagtimes)
    mean_ = np.mean(lagtimes)
    stdev_ = np.std(lagtimes)
    return round(float(mean_), 1), round(float(stdev_), 1)
    # This float cast operation is unnecessary but PyCharm's type checker is incorrectly flagging it.


def calculate_mean(syn_lagtimes: Dict[str, list]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate mean of observed 'lag-times' for each Synucleins construct. It is expected that this function is called
    only on the cleaned 'lag-times', such that only numerical values are included (`NA` is already removed) and
    Synucleins with less than 3 'lag-times' are also removed.
    :param syn_lagtimes: Synuclein names mapped to 'lag-times', with 3 or more values and without any `NA`,
    mapped to the Synuclein name.
    :return: Mean and standard deviation of given 'lag-times' mapped to corresponding Synuclein name.
    """
    return {syn: _calc_mean_of_nonnulls(lags) for syn, lags in syn_lagtimes.items()}


def clean(syn_lagtimes: Dict[str, list]) -> Dict[str, list]:
    """
    Validate and correct syntax of Synuclein names.
    Remove Synuclein expts that do not have enough ThT data - i.e. less than 3 'lag-times'.
    Some Synucleins may be slow to assemble such that the ThT value did not increase (above the heuristic value)
    within the 96-hour cut-off, in one or more experiments. For example, given the following 'lag-times' 68 h, 75 h,
    80 h, NA, NA, there are 3 numerical 'lag-time' values, which satisfies the minimum number required. However,
    the mean cannot include 2 of the 5 experiments because it is non-numeric. Therefore, the 74 h mean of the 3 is
    likely overestimating the fibrillogenic propensity of the protein, because in theory, the other 2
    experiments may have 'lag-times' of 115 h and 125 h. The "real" mean would have been 93 h. To address this, another
    heuristic cut-off is used - `ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES` - whereby even if the protein has the
    minimum number of numerical 'lag-times', it will still be excluded from the subsequent regression modelling if
    there is a null in more than 1/8 proportion of experiments.
    :param syn_lagtimes: Synuclein names mapped to 'lag-times'.
    :return: Synucleins and expts mapped to their 'lag-times', where sufficient data for the Synuclein is available.
    """
    syn_lagtimes_cleaned = {syn.strip().replace('-', '_'): lags for syn, lags in syn_lagtimes.items()}
    try:
        util_data.check_syn_names(syn_names=list(syn_lagtimes_cleaned))
        syn_lagtimes_cleaned = {syn: _remove_nulls(lags) for syn, lags in syn_lagtimes_cleaned.items()
                                if _has_high_enough_proportion_of_non_null_lagtimes(lags) and
                                _has_enough_nonnull_lagtimes(lags)}
    except ValueError as ve:
        print(ve)
    return syn_lagtimes_cleaned


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
    the 'lag-time' (assigned to variable called `tht_lagtime_end_value` in other functions in this module).
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


def _calculate_lagtimes(syn_name: str, two_time_points: list, two_tht_values: list, lagtimes: Dict[str, list],
                        make_plot: bool, degree_to_use: int, tht_lagtime_end_value: float) -> Dict[str, list]:
    """
    Calculate the 'lag-time' as the time at which the square of the starting value is reached, from the given ThT
    values which are expected to only include values that reach this otherwise they are None. Use a polynomial
    regression with the given degree and map all calculated values to the given Synuclein name. (The implementation
    for assigning values to variables `x` and `y` includes filtering out empty rows which were present in an older
    version but are left in place as it is possible that I may alter the upstream functionality so that all the ThT
    values for the experiment are passed here.)
    :param syn_name: Synuclein name. Examples: `asyn`, `a11-140`, `a1-80`, `ba1`, `fr_asyn`, `b45V46V`, `a68-71Del`.
    :param two_time_points: Time point immediately before and time point immediately after end of 'lag-time'.
    :param two_tht_values: ThT value immediately before and ThT value immediately after end of 'lag-time'.
    :param lagtimes: All 'lag-times' calculated thus far, to which the 'lag-time' being calculated here is added.
    :param make_plot: True to display a plot of the polynomial regression used.
    :param degree_to_use: The degree to use for the polynomial regression.
    :param tht_lagtime_end_value: Value of ThT (relative to starting value) that is used to mark end of lag-phase.
    :return: All 'lag-times', mapped to the corresponding Synuclein name.
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
    if syn_name not in lagtimes:
        lagtimes[syn_name] = [lag_hours]
    else:
        lagtimes[syn_name].append(lag_hours)
    return lagtimes


def _include_lag_phase_only(time_points: np.array, tht_values: np.array, tht_lagtime_end_value: float) \
        -> Tuple[np.array, np.array]:
    """
    Find the first time point that the ThT fluorescence emission has reach and/or exceeded the square of the starting
    value and return this and the time point immediately before it, as well as the corresponding ThT values.
    :param time_points: All time points for a particular experiment (including any additional empty rows after the
    final time point). `time_points` will typically look like: `[ 0, 4, 8, 24, 48, 96,  0,  0,  0,]`.
    :param tht_values: All ThT fluorescence data for a particular experiment (including any additional empty rows after
    the final time point). The expected data has already been translated to a starting value of 4.0.
    `tht_values` might look something like: `[4, 4, 5, 20, 150, 250, 0, 0, 0]`, corresponding to the `time_points`
    example shown.
    :param tht_lagtime_end_value: Value of ThT (relative to starting value) that is used to mark end of lag-phase.
    :return: Two consecutive time points including the time point immediately before and the time point
    immediately after the ThT value has reached the square of the starting value. Returns nulls if tht ThT values do
    not exceed the minimum threshold value.
    """
    for i, tht_value in enumerate(tht_values):
        if tht_value >= tht_lagtime_end_value:
            return time_points[i - 1: i + 1], tht_values[i - 1: i + 1]
    return None, None


def _is_null(var) -> bool:
    """
    Returns true for any type of missing value with Python, numpy or pandas libraries.
    (Note: This method seems to have problems when run as a PySpark udf).
    :param var: Any object or a missing value.
    :return: True for missing value.
    """
    is_null = var is None
    if not is_null:
        if isinstance(var, str):
            is_null = var == '' or var.lower() == 'nan' or var.lower() == 'na' or var.lower() == 'null'
        else:
            is_null = var == np.NaN or var == np.NAN or var == np.nan or pd.isna(var) or np.isnan(var)
    return is_null


def _is_date(row_col_0: str) -> bool:
    """
    Determines if given data is date, expecting the format to be 01.01.01
    Accepts this date with any prefix and/or suffix.
    :param row_col_0: Input to be evaluated. It is expected that a null check is performed prior to passing this
    argument to this function.
    :return: True if the given data is a date with the expected date format.
    """
    return re.match(re.compile(r'\d{2}[.]\d{2}[.]\d{2}'), row_col_0) is not None


def _get_data() -> List[str]:
    csvpath = os.path.join(THT_PATH, f"standardised_tht_start_at_{str(STARTING_THT_VALUE).replace('.', '_')}.csv")
    with open(csvpath, 'r') as csv_:
        for row in csv.reader(csv_):
            yield row


def get_lagtimes(make_plot: bool, degree_to_use: int, tht_lagtime_end_value: float) -> Dict[str, list]:
    """
    Read `translatedTo4.csv` file and determine the time taken for each protein's ThT fluorescence to reach the
    square of its starting value, which has been translated to 4.0. This is referred to here as a "lag-time".
    The file contains all ThT data for filament-forming Synucleins measured between 01.11.01 and 30.10.09.
    It is expected to have a particular format, starting a date. This is immediately followed in the next
    row down by `Time (h)` in the first column and the names of the Synuclein constructs in the adjoining columns.
    The subsequent time points at which measurements were taken and the corresponding ThT fluorescence for each
    construct follow in the rows below.
    :param make_plot: True to generate a plot of the region of the ThT fluorescence curve that includes the value by which
    the 'lag-time' is deemed to have ended.
    :param degree_to_use: The degree to use for the polynomial regression of the ThT data.
    :param tht_lagtime_end_value: Value of ThT (relative to starting value) that is used to mark end of 'lag-phase'.
    :return: Each Synuclein name mapped to its 'lag-times', or 'NA' where the ThT does not or increases by
    not above the threshold (typically double the starting value or the square of the starting value).
    """
    syn_names = ()
    time_points = np.zeros(10)
    tht_values = None
    syn_lagtimes = {}

    data_ = iter(_get_data())
    i = 0
    for row in data_:
        if _is_date(row[0]):
            continue
        elif row[0] == STR_TIME_H:
            syn_names += tuple(filter(None, row[1:]))
            tht_values = np.zeros((10, len(syn_names)))
        elif row[0] == '0' or row[0] == '0.0':
            i = 0
            tht_values[i, ] = [float(n) if n != '' else 0.0 for n in row[1:len(syn_names) + 1]]
            time_points[i] = row[0]
        elif row[0] == '':
            for i, syn_name in enumerate(syn_names):
                two_time_points, two_tht_values = _include_lag_phase_only(time_points=time_points,
                                                                          tht_values=tht_values[:, i],
                                                                          tht_lagtime_end_value=tht_lagtime_end_value)
                syn_lagtimes = _calculate_lagtimes(syn_name=syn_name, two_time_points=two_time_points,
                                                   two_tht_values=two_tht_values, lagtimes=syn_lagtimes,
                                                   make_plot=make_plot, degree_to_use=degree_to_use,
                                                   tht_lagtime_end_value=tht_lagtime_end_value)
            syn_names = ()
        else:
            tht_values[i, ] = [float(n) if n != '' else 0.0 for n in row[1: len(syn_names) + 1]]
            time_points[i] = row[0]
        i += 1
    return syn_lagtimes


def _standardise_tht(pre_standardised_tht: list, translate_by: Dict[str, float], scaling_factor: float,
                     syn_names: list) -> list:
    """
    Scale all ThT values to the concurrently run alpha-Synuclein and translate all values to start at 4.0 for the
    given experiment.
    :param pre_standardised_tht: Raw ThT values for one experiments.
    :param translate_by: How much to translate each Synuclein's ThT values according to starting at 4.0.
    :param scaling_factor: How much to scale all Synuclein's ThT values in one experiment.
    :param syn_names:
    :return:
    """
    standardised_tht = []
    for row_dict in pre_standardised_tht:
        if _is_null(row_dict[0]):
            standardised_tht.append(row_dict)
        elif _is_date(row_dict[0]):
            standardised_tht.append(row_dict)
        elif row_dict[0].lower().startswith('time'):
            standardised_tht.append(row_dict)
        elif row_dict[0].isnumeric():
            row_dict_copy = row_dict.copy()  # Definitely no nested objects in this dict, hence shallow copy suffices.
            for i, syn in enumerate(syn_names):
                row_dict_copy[i + 1] = float(row_dict_copy[i + 1]) * scaling_factor
                row_dict_copy[i + 1] = float(row_dict_copy[i + 1]) + translate_by[syn]
            standardised_tht.append(row_dict_copy)
        else:
            continue
    return standardised_tht


def standardise_tht() -> List[dict]:
    """
    Scale all ThT values of all Synucleins in one experiment according to the final value (96 hours) of the
    concurrently run alpha-Synuclein, according to the equation: (100 / alpha-Synuclein's 96 hour ThT value).
    Translate all scaled ThT values for each Synuclein in one experiment according to the starting value (0 hours)
    for each Synuclein, such that all will start at the STARTING_THT_VALUE (typically 4.0) and all subsequent time
    points will be translated by the same difference. (Note: the necessary order of operations, i.e. scaling followed
    by translation, means that alpha-Synuclein's final standardised ThT value is rarely exactly 100.0)
    :return: Standardised ThT data
    """
    df = pd.read_csv(ALL_THT_DATA_CSV_PATH, header=None)
    num_of_cols = df.shape[1]
    temp_col_names = tuple(range(num_of_cols))
    one_tht_expt = []
    syn_names = []
    zero_time_values = {}
    standardised_tht_all = []

    for row in df.itertuples(index=False):
        row = tuple(row)
        if _is_null(row[0]):
            one_tht_expt.append(dict(zip(temp_col_names, row)))
        elif _is_date(row[0]):
            one_tht_expt.append(dict(zip(temp_col_names, row)))
        elif row[0].lower().startswith('time'):
            row_list = list(row)
            row_list[0] = STR_TIME_H
            row = tuple(row_list)
            one_tht_expt.append(dict(zip(temp_col_names, row)))
            assert(row[1] == 'asyn')
            syn_names = [r for r in row[1:] if r is not np.nan]
        elif float(row[0]) == 0:
            one_tht_expt.append(dict(zip(temp_col_names, row)))
            zero_time_values = {syn_name: float(row[1 + i]) for i, syn_name in enumerate(syn_names)}
        elif 0 < float(row[0]) < 90:
            one_tht_expt.append(dict(zip(temp_col_names, row)))
        elif 90 <= float(row[0]) < 101:
            one_tht_expt.append(dict(zip(temp_col_names, row)))
            scaling_factor = STANDARD_ASYN_END_THT_VALUE / float(row[1])
            translate_by = {syn_name: STARTING_THT_VALUE - (scaling_factor * zero_time_values[syn_name]) for
                            syn_name in syn_names}
            standardised_tht_all.extend(_standardise_tht(one_tht_expt, translate_by, scaling_factor, syn_names))
            one_tht_expt = []
        else:
            continue

    return standardised_tht_all


def write_lagtime_means(lagtime_means: Dict[str, float], degree_used: int, tht_lagtime_end_value_used: float):
    """

    :param lagtime_means: Mean & standard deviation of the 'lag-times' for each Synuclein.
    :param degree_used: Polynomial degree used to fit the two ThT values that span the end of 'lag-phase'.
    :param tht_lagtime_end_value_used: Value of ThT (relative to starting value) used to mark end of 'lag-phase'.
    """
    col_names = ['Synucleins', 'lagtime_means', 'std_devs']
    df = pd.DataFrame.from_dict(data=lagtime_means, orient='index', columns=col_names[1: 3])
    lagtime_filename = f'lagtime_means_polynDegree_{degree_used}_lagtimeEndvalue_{int(tht_lagtime_end_value_used)}.csv'
    df.to_csv(os.path.join(LAGTIME_MEANS_PATH, lagtime_filename), index=True)


def write_standardised_tht_data_all(standardised_tht_all: List[dict]):
    standardised_tht_filename = f"standardised_tht_start_at_{str(STARTING_THT_VALUE).replace('.', '_')}.csv"
    df = pd.DataFrame(standardised_tht_all)
    df.to_csv(os.path.join(THT_PATH, standardised_tht_filename), index=False, header=False)


if __name__ == '__main__':
    # from src.utils.file_manipulator import read_xls_and_write_csv
    # read_xls_and_write_csv(xls_path=os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.xls'))

    # write_standardised_tht_data_all(standardise_tht())
    for _degree_to_use in [3, 4, 5]:
        for lagtime_end_value_to_use in [_1_5_STARTING_VALUE, DOUBLE_STARTING_VALUE, SQUARE_OF_STARTING_VALUE]:
            _lagtimes = get_lagtimes(make_plot=False, degree_to_use=_degree_to_use,
                                     tht_lagtime_end_value=lagtime_end_value_to_use)
            write_lagtimes(lagtimes=_lagtimes, degree_used=_degree_to_use,
                           tht_lagtime_end_value_used=lagtime_end_value_to_use)
            _lagtimes_cleaned = clean(syn_lagtimes=_lagtimes)
            _lagtime_means_stdev = calculate_mean(_lagtimes_cleaned)
            write_lagtime_means(lagtime_means=_lagtime_means_stdev, degree_used=_degree_to_use,
                                tht_lagtime_end_value_used=lagtime_end_value_to_use)
