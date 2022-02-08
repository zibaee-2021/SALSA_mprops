from unittest import TestCase
from src.mean_properties import mnc_mtc
from data.protein_sequences import read_seqs


class TestMNCMTC(TestCase):

    def test_compute_mean_total_charge(self):
        expected_zero = 0
        expected_one = 1
        expected_two_thirds = 2/3
        expected_one_third = 1/3
        self.assertEqual(expected_zero, mnc_mtc.compute_mean_total_charge('AAA'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_total_charge('KKK'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_total_charge('RRR'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_total_charge('DDD'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_total_charge('EEE'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_total_charge('KEA'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_total_charge('RDA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_total_charge('KAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_total_charge('RAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_total_charge('EAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_total_charge('DAA'))

    def test_compute_mean_total_charge_asyn(self):
        asyn = read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')
        actual = mnc_mtc.compute_mean_total_charge(asyn)
        expected = 0.2785714285714286
        self.assertEqual(expected, actual)

    def test_compute_mean_net_charge(self):
        expected_zero = 0
        expected_one = 1
        expected_two_thirds = 2/3
        expected_one_third = 1/3
        self.assertEqual(expected_zero, mnc_mtc.compute_mean_net_charge('AAA'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_net_charge('KKK'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_net_charge('RRR'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_net_charge('DDD'))
        self.assertEqual(expected_one, mnc_mtc.compute_mean_net_charge('EEE'))
        self.assertEqual(expected_zero, mnc_mtc.compute_mean_net_charge('KEA'))
        self.assertEqual(expected_zero, mnc_mtc.compute_mean_net_charge('RDA'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_net_charge('RKA'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_net_charge('RAR'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_net_charge('EDA'))
        self.assertEqual(expected_two_thirds, mnc_mtc.compute_mean_net_charge('EAE'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_net_charge('KAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_net_charge('RAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_net_charge('EAA'))
        self.assertEqual(expected_one_third, mnc_mtc.compute_mean_net_charge('DAA'))

    def test_compute_mean_net_charge_asyn(self):
        asyn = read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')
        actual = mnc_mtc.compute_mean_net_charge(asyn)
        expected = 0.06428571428571428
        self.assertEqual(expected, actual)