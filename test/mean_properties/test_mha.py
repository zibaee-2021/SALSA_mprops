from unittest import TestCase
from src.mean_properties import mha


class TestMHA(TestCase):

    def test_compute_mean_helical_amphipathicity(self):

        actual = mha.compute_mean_helical_amphipathicity('A', periodicity=100)
        self.assertEqual(0, actual)
        actual = mha.compute_mean_helical_amphipathicity('Q', periodicity=100)
        self.assertEqual(-1.0, actual)
        actual = mha.compute_mean_helical_amphipathicity('G', periodicity=100)
        self.assertEqual(1.0, actual)
        actual = mha.compute_mean_helical_amphipathicity('AA', periodicity=180)
        self.assertEqual(0, actual)
        actual = mha.compute_mean_helical_amphipathicity('AQ', periodicity=100)
        self.assertEqual(0, actual)
        actual = mha.compute_mean_helical_amphipathicity('QQ', periodicity=100)
        self.assertEqual(0, actual)
