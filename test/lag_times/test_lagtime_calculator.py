from unittest import TestCase
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

    def test_calc_mean_of_lagtimes(self):
        lagtimes = {'syn': [20, 30, 40, 10]}
        actual = lagcalc._calc_mean_of_lag_times(lagtimes)
        expected = {'syn': 25}
        self.assertEqual(expected, actual)

    def test_write_lag_time_means(self):
        lag_time_means = {'syn': 23}
        actual = lagcalc.write_lag_time_means(lag_time_means)
