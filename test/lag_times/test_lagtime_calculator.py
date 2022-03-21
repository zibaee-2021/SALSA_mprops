from unittest import TestCase
import os
import pandas as pd
from pandas import testing as pdt
from root_path import abspath_root
from src.lagtimes import lagtime_calculator as lagcalc


class TestLagTimeCalculator(TestCase):

    def test__remove_nulls(self):
        self.assertEqual([80, 60, 40], lagcalc._remove_nulls([80, 60, 40]))
        self.assertEqual([80, 60], lagcalc._remove_nulls([80, 60, 'NA']))
        self.assertEqual([], lagcalc._remove_nulls(['NA', 'NA', 'NA']))

    def test__has_enough_nonnull_lagtimes(self):
        self.assertFalse(lagcalc._has_enough_nonnull_lagtimes([80, 'NA', 'NA', 80]))
        self.assertFalse(lagcalc._has_enough_nonnull_lagtimes([80, 60]))
        self.assertTrue(lagcalc._has_enough_nonnull_lagtimes([80, 'NA', 60, 'NA', 80]))
        self.assertTrue(lagcalc._has_enough_nonnull_lagtimes([80, 60, 80]))

    def test__has_high_enough_proportion_of_non_null_lagtimes(self):
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 'NA', 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 'NA']))
        self.assertFalse(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 6, 'NA']))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1]))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2]))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 6, 7, 'NA']))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 6, 7, 8, 'NA']))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 6, 7, 8, 9, 'NA']))
        self.assertTrue(lagcalc._has_high_enough_proportion_of_non_null_lagtimes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'NA']))

    def test__calc_mean_of_lagtimes(self):
        self.assertDictEqual({'asyn': 2}, lagcalc._calc_mean_of_lagtimes({'asyn': [1, 2, 3]}))
        self.assertDictEqual({}, lagcalc._calc_mean_of_lagtimes({'asyn': [80, 60, 'NA']}))
        self.assertDictEqual({}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 'NA']}))
        self.assertDictEqual({}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 10, 'NA']}))
        self.assertDictEqual({}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 10, 10, 'NA']}))
        self.assertDictEqual({}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 10, 10, 10, 'NA']}))
        self.assertDictEqual({'asyn': 10}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 10, 10, 10, 10, 'NA']}))
        self.assertDictEqual({'asyn': 10}, lagcalc._calc_mean_of_lagtimes({'asyn': [10, 10, 10, 10, 10, 10, 10, 10,
                                                                                    'NA']}))

    def test__include_lag_phase_only(self):
        X = [0, 4, 8, 24, 48, 96]
        y = [4.0, 6.1, 12.4, 16.0, 97.5, 123.6]
        actual = lagcalc._include_lag_phase_only(X, y)
        expected = [8, 24], [12.4, 16.0]
        self.assertTupleEqual(expected, actual)
        self.assertCountEqual(expected[0], actual[0])
        self.assertCountEqual(expected[1], actual[1])

    def test_write_lagtime_means(self):
        lagtime_means = {'asyn': 23, 'bsyn': 400}
        lagtime_means_df = pd.DataFrame.from_dict(data=lagtime_means, orient='index', columns=['lagtime_means'])
        lagcalc.write_lagtime_means(lagtime_means=lagtime_means, degree_used=4, tht_lagtime_end_value_used=16)

        expected_lagtime_filename = 'lagtime_means_polynDegree_4_lagtimeEndvalue_16.csv'
        expected_lagtime_csv = os.path.join(abspath_root, 'data', 'tht_data', 'lagtimes', expected_lagtime_filename)
        expected_lagtime_means_df = pd.read_csv(expected_lagtime_csv, index_col=[0])
        pdt.assert_frame_equal(expected_lagtime_means_df, lagtime_means_df)
