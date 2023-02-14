import os
import csv
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.utils import util_data
from src.utils.Constants import LagTimeCalc
constants = LagTimeCalc()
"""
Logic of the data pipeline that produces lag-time data from raw ThT data. 

1. Read raw ThT data ('AllThtData.csv').
2. Standardise and scale the fluorescence values.
3. Write to csv ('standardised_tht_start_at_4_0.csv').
write_standardised_tht_data_all(standardise_tht())

4. Read in processed ThT data (('standardised_tht_start_at_4_0.csv').
5. Parse data and calculate lag-times via polynomial fit of the early time-point values. 
6. Write to csv.
write_lagtimes(get_lagtimes())

7. Clean lag time data. This includes a number of threshold heuristics that determine which synucleins have 
sufficient data to include them in subsequent calculations. For example, synucleins that do not have enough ThT 
lag-time data is currently determined to be those with less than 3 lagtimes. 
Some synucleins may be too slow to assemble such that the ThT value did not increase (above the threshold 
heuristic) within the 96-hour cut-off, leading to no ThT value for that experiment. Some synucleins had a proportion 
of experiments with no lag-time, so a determination must be made as to what proportion of experiments a synuclein is 
permitted to have no recorded lag-time for the average of the remaining experiments to be considered valid and 
representative.
8. Calculate the means of lag-times and the standard deviations.
9. Write to csv.
write_lagtime_means(calculate_means_and_stdev(clean(syn_lagtimes=_lagtimes)))

Most of the steps in the pipeline described above in steps 1-9 above take heuristics are parameters, 
enabling experimentation to find optimal settings, which can be inferred from the closeness of fit to protein 
physicochemical properties by various models, performed downstream in the pipeline via functions in mprops.py, 
salsa.py and mprops_bsc_combo.py modules.  

"""


def _has_enough_nonnull_lagtimes(lagtimes: list) -> bool:
    """
    Checks that the number of lagtimes in the given list is above the globally-defined value, (typically 3).
    The given list of lagtimes may have already had all nulls removed, but this function can still handle a
    null-containing list.
    :param lagtimes: lagtimes.
    :return: True if the given list has at least the minimum number of non-null lagtimes.
    """
    has_enough_nonnull_lagtimes = len([lagtime for lagtime in lagtimes if lagtime != 'NA']) >= \
                                  constants.MIN_NUM_OF_LAGTIMES_NEEDED
    return has_enough_nonnull_lagtimes


def _has_high_enough_proportion_of_non_null_lagtimes(lagtimes: list) -> bool:
    """
    Checks that the number of lagtimes in the given list has a sufficient proportion of non-null to null values.
    (Typically it is approximated that there should be more than 7 non-null in 8 lagtime values, otherwise the
    subsequent mean value is not reliably representative of the synuclein's fibrillogenic propensity and is removed
    from the subsequent linear regression-based predictions. Note, such synuclein datasets will still be valuable
    for classification-based predictions).
    :param lagtimes: lagtimes that have not had nulls previously removed.
    :return: True if the given list has at least the minimum proportion of non-null to null values.
    """
    num_of_nones = len([lag for lag in lagtimes if lag == 'NA'])
    has_enough_prop_of_lagtimes = (num_of_nones / len(lagtimes)) <= constants.ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES
    return has_enough_prop_of_lagtimes


def _remove_nulls(lagtimes: list) -> list:
    """
    Remove the value representing null within the given list of lagtimes.
    (Currently, it is represented by the string 'NA', although it may change without this comment being updated).
    :param lagtimes: lagtimes, potentially containing null values.
    :return: Non-null lagtimes only.
    """
    nonnulls = [lag for lag in lagtimes if lag != 'NA']
    return nonnulls


def _calc_means_and_stdev_of_nonnulls(lagtimes: list) -> List[float]:
    """
    Calculate means and standard deviations for given lagtimes. (The parameter passed is expected to be the
    lag-times of a single synuclein.)
    :param lagtimes: lagtimes (expected to be in hours).
    :return: Means and standard deviations of given lagtimes.
    """
    lagtimes = _remove_nulls(lagtimes)
    mean_ = np.mean(lagtimes)
    stdev_ = np.std(lagtimes)
    return [round(float(mean_), 1), round(float(stdev_), 1)]
    # This float cast operation is unnecessary but PyCharm's type checker is currently incorrectly flagging it.


def calculate_means_and_stdev(syn_lagtimes: Dict[str, list]) -> Dict[str, List[float]]:
    """
    Calculate mean of observed lagtimes for each synucleins construct. It is expected that this function is called
    only on the cleaned lagtimes, such that only numerical values are included (`NA` is already removed) and
    synucleins with less than 3 lagtimes are also removed.
    :param syn_lagtimes: Synuclein names mapped to lagtimes, with 3 or more values and without any `NA`,
    mapped to the synuclein name.
    :return: Mean and standard deviation of given lagtimes mapped to corresponding synuclein name.
    """
    return {syn: _calc_means_and_stdev_of_nonnulls(lags) for syn, lags in syn_lagtimes.items()}


def clean(syn_lagtimes: Dict[str, list]) -> Dict[str, list]:
    """
    Validate and correct syntax of synuclein names.
    Remove synuclein experiments that do not have enough ThT data - i.e. less than 3 lagtimes.
    Some synucleins may be slow to assemble such that the ThT value did not increase (above the heuristic value)
    within the 96-hour cut-off, in one or more experiments. For example, given the following lagtimes 68 h, 75 h,
    80 h, NA, NA, there are 3 numerical lagtime values, which satisfies the minimum number required. However,
    the mean cannot include 2 of the 5 experiments because it is non-numeric. Therefore, the 74 h mean of the 3 is
    likely overestimating the fibrillogenic propensity of the protein, because in theory, the other 2
    experiments may have lagtimes of 115 h and 125 h. The "real" mean would have been 93 h. To address this, another
    heuristic cut-off is used - `ACCEPTABLE_MAX_PROPORTION_OF_NULL_LAGTIMES` - whereby even if the protein has the
    minimum number of numerical lagtimes, it will still be excluded from the subsequent regression modelling if
    there is a null in more than 1/8 proportion of experiments.
    :param syn_lagtimes: Synuclein names mapped to lagtimes.
    :return: Synucleins & experiments mapped to their lagtimes where sufficient data for the synuclein is available.
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


def _solve_lagtime(poly_coefs, intercept, y) -> float:
    """
    Find time taken to reach the end of the 'lag-phase' according to the pre-determined start value (e.g. 4.0) and
    the pre-determined function of that start value (e.g. square of 4.0).

    Implementation details:
    Solve given polynomial for given intercept and ordinate, returning the corresponding abscissa value.
    As the model is created from a polynomial, the roots of the function are required to solve it.
    E.g. If degree of polynomial used is 3, function will be:
                y = (m3 * x^3) + (m2 * x^2) + (m1 + x) + c,
    where `m` is the coefficient of each x value and `c` is the y-intercept.
    Hence, to solve using the roots, the equation is translated along the ordinate so that y = 0, such that:
                0 = (m3 * x^3) + (m2 * x^2) + (m1 + x) + c.
    Therefore, to solve for the y value of 16.0, the function must be subtracted by 16.0.

    Note: While `numpy.roots` expects the order of the equation to be as shown above - starting with higher order and
    ending with the intercept, `poly_coefs` comes from `sklearn.linear_model.LinearRegression.coef_` which
    returns the coefficient the other way round (hence I must `flip` the order). Furthermore, sklearn's `poly_coefs`
    lacks the intercept value, so I must add this to the end of the flipped array, before subtracting the function by
    the 16.0.)
    :param poly_coefs: Coefficients of the polynomial function used to model the data.
    :param intercept: The y-intercept of the function, which is the translated starting ThT value, 4.0.
    :param y: ThT value that we want to solve abscissa (time) for. This heuristic is used to mark the end of the
    'lag-phase' (assigned to variable called `tht_lagtime_end_value` in other functions in this module).
    E.g. 16.0 (square of starting value 4.0).
    :return: The abscissa solution for the given ordinate value
    """
    pc = poly_coefs.copy()
    pc_ = np.flip(pc, axis=0)
    pc_[-1] += intercept
    pc_[-1] -= y
    roots = np.roots(pc_)
    return roots[0]


def _plot(preproc: PolynomialFeatures, lin_reg: LinearRegression, x: list, y: np.ndarray, y_pred: float, syn_name: str,
          degree_used: int, tht_lag: float, lag_hrs: float):
    """
    Display a plot of the polynomial regression, with the given time points as abscissa. The ordinate includes both
    the ThT values and the predicted values with the regression line.
    :param preproc: Preprocessing method used. (For example the polynomial, according to a specified degree).
    :param lin_reg: The linear regression model, already trained only polynomially processed time point values.
    :param x: Time points (expected to be only two time points).
    :param y: ThT values (expected to be only two ThT values).
    :param y_pred: Predicted ThT value according to the trained polynomial regression model.
    :param syn_name: The synuclein name.
    :param degree_used: Polynomial degree used for fitting ThT data (for plot title).
    :param tht_lag: ThT value immediately after the end of lagtime (for plot title).
    :param lag_hrs: lagtime in hours.
    """
    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_pred, color='green')
    plt.plot(X_grid, lin_reg.predict(preproc.fit_transform(X_grid)), color='black')
    plt.title(f'{syn_name.upper()}/ lag={lag_hrs}hr/ lag ThT={round(tht_lag, 1)}/ poly-degr={degree_used} ')
    plt.xlabel('Time')
    plt.ylabel('ThT (h)')
    plt.show()


def _calculate_lagtimes(syn_name: str, two_time_points: np.ndarray, two_tht_values: np.ndarray,
                        lagtimes: Dict[str, list], make_plot: bool, degree_to_use: int,
                        tht_lagtime_end_value: float) -> Dict[str, list]:
    """
    Calculate the lagtime as the time at which the square of the starting value is reached, from the given ThT
    values which are expected to only include values that reach this otherwise they are None. Use a polynomial
    regression with the given degree and map all calculated values to the given synuclein name. (The implementation
    for assigning values to variables `x` and `y` includes filtering out empty rows which were present in an older
    version but are left in place as it is possible that I may alter the upstream functionality so that all the ThT
    values for the experiment are passed here.)
    :param syn_name: Synuclein name. Examples: `asyn`, `a11-140`, `a1-80`, `ba1`, `fr_asyn`, `b45V46V`, `a68-71Del`.
    :param two_time_points: Time point immediately before and time point immediately after end of lagtime.
    :param two_tht_values: ThT value immediately before and ThT value immediately after end of lagtime.
    :param lagtimes: All lagtimes calculated thus far, to which the lagtime being calculated here is added.
    :param make_plot: True to display a plot of the polynomial regression used.
    :param degree_to_use: The degree to use for the polynomial regression.
    :param tht_lagtime_end_value: Value of ThT (relative to starting value) that is used to mark end of lag-phase.
    :return: All lagtimes, mapped to the corresponding synuclein name.
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
        lag_hours = np.real(_solve_lagtime(poly_coefs=lin_reg.coef_, intercept=constants.STARTING_THT_VALUE,
                                           y=tht_lagtime_end_value))
        lag_hours = abs(round(float(lag_hours), 1))

        if make_plot:
            y_pred = lin_reg.predict(x_poly)
            _plot(preproc=poly_reg, lin_reg=lin_reg, x=x, y=y, y_pred=y_pred, syn_name=syn_name,
                  degree_used=degree_to_use, tht_lag=two_tht_values[-1], lag_hrs=lag_hours)

    if syn_name not in lagtimes:
        lagtimes[syn_name] = [lag_hours]
    else:
        lagtimes[syn_name].append(lag_hours)
    return lagtimes


def _include_lag_phase_only(time_points: np.ndarray, tht_values: np.ndarray, tht_lagtime_end_value: float) \
        -> Tuple[np.ndarray, np.ndarray]:
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
    csvpath = os.path.join(constants.THT_PATH,
                           f"standardised_tht_start_at_{str(constants.STARTING_THT_VALUE).replace('.', '_')}.csv")
    with open(csvpath, 'r') as csv_:
        for row in csv.reader(csv_):
            yield row


def get_lagtimes(make_plot: bool, degree_to_use: int, tht_lagtime_end_value: float) -> Dict[str, list]:
    """
    Read `translatedTo4.csv` file and determine the time taken for each protein's ThT fluorescence to reach the
    square of its starting value, which has been translated to 4.0. This is referred to here as a "lag-time".
    The file contains all ThT data for filament-forming synucleins measured between 01.11.01 and 30.10.09.
    It is expected to have a particular format, starting a date. This is immediately followed in the next
    row down by `Time (h)` in the first column and the names of the synuclein constructs in the adjoining columns.
    The subsequent time points at which measurements were taken and the corresponding ThT fluorescence for each
    construct follow in the rows below.
    :param make_plot: True to generate a plot of the region of the ThT fluorescence curve that includes the value by
    which the lagtime is deemed to have ended.
    :param degree_to_use: The degree to use for the polynomial regression of the ThT data.
    :param tht_lagtime_end_value: Value of ThT (relative to starting value) that is used to mark end of 'lag-phase'.
    :return: Each synuclein name mapped to its lagtimes, or 'NA' where the ThT does not or increases by
    not above the threshold.
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
        elif row[0] == constants.STR_TIME_H:
            syn_names += tuple(filter(None, row[1:]))
            tht_values = np.zeros((10, len(syn_names)))
        elif row[0] == '0' or row[0] == '0.0':
            i = 0
            tht_values[i, ] = [float(n) if n != '' else 0.0 for n in row[1: len(syn_names) + 1]]
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


def _standardise_tht(raw_tht: list, translate_by: Dict[str, float], scaling_factor: float,
                     syn_names: list) -> list:
    """
    Scale all ThT values to the concurrently run alpha-synuclein and translate all values to start at 4.0 for the
    given experiment.
    :param raw_tht: Raw ThT values for one experiment.
    :param translate_by: How much to translate each synuclein's ThT values according to starting at 4.0.
    :param scaling_factor: How much to scale all synuclein's ThT values in one experiment.
    :param syn_names:
    :return:
    """
    standardised_tht = []
    for row_dict in raw_tht:
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
    Scale all ThT values of all synucleins in one experiment according to the final value (4 days) of the
    concurrently run alpha-synuclein, according to the equation: (100 / alpha-synuclein's 4-day ThT value).
    Translate all scaled ThT values for each synuclein in one experiment according to the starting value (0 hours)
    for each synuclein, such that all will start at the STARTING_THT_VALUE (typically 4.0) and all subsequent time
    points will be translated by the same difference. (Note: the necessary order of operations, i.e. scaling followed
    by translation, means that alpha-synuclein's final standardised ThT value is rarely exactly 100.0)
    :return: Standardised ThT data
    """
    df = pd.read_csv(constants.ALL_THT_DATA_CSV_PATH, header=None)
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
            row_list[0] = constants.STR_TIME_H
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
            scaling_factor = constants.STANDARD_ASYN_END_THT_VALUE / float(row[1])
            translate_by = {syn_name: constants.STARTING_THT_VALUE - (scaling_factor * zero_time_values[syn_name]) for
                            syn_name in syn_names}
            standardised_tht_all.extend(_standardise_tht(one_tht_expt, translate_by, scaling_factor, syn_names))
            one_tht_expt = []
        else:
            continue

    return standardised_tht_all


def write_lagtimes(lagtimes: Dict[str, list], degree_used: int, tht_lagtime_end_value_used: float):
    """
    Write lagtimes to csv for each and every assembly experiment.
    May be used to explore effect of weighting subsequent regression models of lagtime means to properties.
    :param lagtimes: lagtimes for each and every synuclein experiment. Not the averages.
    :param degree_used: Polynomial degree used to fit the two ThT values that span the end of 'lag-phase'.
    :param tht_lagtime_end_value_used: Value of ThT (relative to starting value) used to mark end of 'lag-phase'.
    """
    pdf = pd.Series(lagtimes).to_frame('lagtimes (h)')
    lagtime_filename = f'lt_polyDeg{degree_used}_ltEnd{int(tht_lagtime_end_value_used)}.csv'
    pdf.to_csv(os.path.join(constants.LAGTIMES_PATH, lagtime_filename), index=True)


def write_lagtime_means(lagtime_means_stdev: Dict[str, List[float]], degree_used: int,
                        tht_lagtime_end_value_used: float, dst_dir: str = constants.LAGTIME_MEANS_PATH):
    """
    Write means and standard deviations of lagtimes per synuclein.
    (These values are used for fitting regression models to the protein physicochemical properties.)
    :param lagtime_means_stdev: Mean & standard deviation of the lagtimes for each synuclein.
    :param degree_used: Polynomial degree used to fit the two ThT values that span the end of 'lag-phase'.
    :param tht_lagtime_end_value_used: Value of ThT (relative to starting value) used to mark end of 'lag-phase'.
    :param dst_dir: Destination directory (including absolute path). Default is data/tht_data/lagtimes/lagtime_means.
    """
    col_names = ['lagtime_means', 'std_devs']
    df = pd.DataFrame.from_dict(data=lagtime_means_stdev, orient='index', columns=col_names)
    lagtime_filename = f'ltMeans_polyDeg{degree_used}_ltEnd{int(tht_lagtime_end_value_used)}.csv'
    df.to_csv(os.path.join(dst_dir, lagtime_filename), index=True)


def write_standardised_tht_data_all(standardised_tht_all: List[dict]):
    standardised_tht_filename = f"standardised_tht_start_at_{str(constants.STARTING_THT_VALUE).replace('.', '_')}.csv"
    df = pd.DataFrame(standardised_tht_all)
    df.to_csv(os.path.join(constants.THT_PATH, standardised_tht_filename), index=False, header=False)


if __name__ == '__main__':
    # from src.utils.file_manipulator import read_xls_and_write_csv
    # from root_path import abspath_root
    # read_xls_and_write_csv(xls_path=os.path.join(abspath_root, 'data', 'tht_data', 'AllThTData.xls'))
    # standardised_tht_all_ = standardise_tht()
    # write_standardised_tht_data_all(standardised_tht_all_)

    for _degree_to_use in [4, 5]:
        for lagtime_end_value_to_use in [constants.SQUARE_OF_STARTING_VALUE]:
            _lagtimes = get_lagtimes(make_plot=True, degree_to_use=_degree_to_use,
                                     tht_lagtime_end_value=lagtime_end_value_to_use)
    #         write_lagtimes(lagtimes=_lagtimes, degree_used=_degree_to_use,
    #                        tht_lagtime_end_value_used=lagtime_end_value_to_use)
    #         _lagtimes_cleaned = clean(syn_lagtimes=_lagtimes)
    #         _lagtime_means_stdev = calculate_means_and_stdev(_lagtimes_cleaned)
    #         write_lagtime_means(lagtime_means_stdev=_lagtime_means_stdev, degree_used=_degree_to_use,
    #                             tht_lagtime_end_value_used=lagtime_end_value_to_use)
