from unittest import TestCase
from src.mean_properties import medm


class TestMEDM(TestCase):

    def test_compute_mean_electric_dipole_moment(self):

        actual = medm.compute_mean_electric_dipole_moment('M', periodicity=180)
        self.assertEqual(0, actual)
        actual = medm.compute_mean_electric_dipole_moment('D', periodicity=180)
        self.assertEqual(-1.0, actual)
        actual = medm.compute_mean_electric_dipole_moment('K', periodicity=180)
        self.assertEqual(1.0, actual)
        actual = medm.compute_mean_electric_dipole_moment('DD', periodicity=180)
        self.assertEqual(0, actual)
        actual = medm.compute_mean_electric_dipole_moment('KK', periodicity=180)
        self.assertEqual(0, actual)
        actual = medm.compute_mean_electric_dipole_moment('DK', periodicity=180)
        self.assertEqual(-1, actual)
        actual = medm.compute_mean_electric_dipole_moment('KD', periodicity=180)
        self.assertEqual(1, actual)
