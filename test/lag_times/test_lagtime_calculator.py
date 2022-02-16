from unittest import TestCase
from src.lag_times import lagtime_calculator


class TestLagTimeCalculator(TestCase):

    def test__include_lag_phase_only(self):
        X = [0, 4, 8, 24, 48, 96]
        y = [4.0, 6.1, 12.4, 16.0, 97.5, 123.6]
        actual = lagtime_calculator._include_lag_phase_only(X, y)
        expected = [8, 24], [12.4, 16.0]
        self.assertTupleEqual(expected, actual)
        self.assertCountEqual(expected[0], actual[0])
        self.assertCountEqual(expected[1], actual[1])