from unittest import TestCase
import os
import pandas as pd
from pandas import testing as pdt
from root_path import abspath_root
from src.lag_times import lagtime_calculator as lagcalc


class TestLagTimeCalculator(TestCase):

    def test__include_lag_phase_only(self):
        X = [0, 4, 8, 24, 48, 96]
        y = [4.0, 6.1, 12.4, 16.0, 97.5, 123.6]
        actual = lagcalc._include_lag_phase_only(X, y)
        expected = [8, 24], [12.4, 16.0]
        self.assertTupleEqual(expected, actual)
        self.assertCountEqual(expected[0], actual[0])
        self.assertCountEqual(expected[1], actual[1])

    def test_process_na(self):
        lagtimes = {'asyn': [80, 60]}
        actual = lagcalc._process_na(lagtimes)
        expected = {}
        self.assertDictEqual(expected, actual)

        lagtimes = {'asyn': [80, 60, 'NA']}
        actual = lagcalc._process_na(lagtimes)
        expected = {}
        self.assertDictEqual(expected, actual)

        lagtimes = {'asyn': [80, 60, 70, 'NA']}
        actual = lagcalc._process_na(lagtimes)
        expected = {'asyn': [80, 60, 70]}
        self.assertDictEqual(expected, actual)

        lagtimes = {'asyn': [80, 'NA', 60, 'NA', 80]}
        actual = lagcalc._process_na(lagtimes)
        expected = {'asyn': [80, 60, 80]}
        self.assertCountEqual(expected['asyn'], actual['asyn'])

    def test_process_na2(self):
        lagtimes = {'11-140': [80, 60, 40]}
        actual = lagcalc._process_na(lagtimes)
        expected = {'11_140': [80, 60, 40]}
        self.assertDictEqual(expected, actual)

    def test_calc_mean_of_lagtimes(self):
        lagtimes = {'syn': [20, 30, 40, 10]}
        actual = lagcalc._calc_mean_of_lag_times(lagtimes)
        expected = {'syn': 25}
        self.assertEqual(expected, actual)

    def test_write_lag_time_means(self):
        lag_time_means = {'asyn': 23, 'bsyn': 400}
        lag_time_means_df = pd.DataFrame.from_dict(data=lag_time_means, orient='index',
                                                            columns=['lag_time_means'])
        lagcalc.write_lag_time_means(lag_time_means=lag_time_means, degree_used=4, tht_lagtime_end_value_used=16)

        expected_lag_time_filename = f'lag_time_Degree_{4}_End_value_{16}.csv'
        expected_lag_time_csv = os.path.join(abspath_root, 'data', 'tht_data', expected_lag_time_filename)
        expected_lag_time_means_df = pd.read_csv(expected_lag_time_csv, index_col=[0])

        pdt.assert_frame_equal(expected_lag_time_means_df, lag_time_means_df)
