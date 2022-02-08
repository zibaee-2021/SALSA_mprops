from unittest import TestCase
from src.mean_properties import mh
from data.protein_sequences import read_seqs


class TestMNCMTC(TestCase):

    def test_compute_mean_hydrophilicity(self):
        rw = {'A': -0.45, 'C': -3.63, 'D': -13.34, 'E': -12.63}
        expected = (rw['A'] + rw['C'] + rw['D'] + rw['E']) / 4
        actual = mh.compute_mean_hydrophilicity('ACDE')
        self.assertEqual(expected, actual)

    def test_compute_mean_hydrophilicity_asyn(self):
        expected = -5.713357142857143
        asyn = read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')
        actual = mh.compute_mean_hydrophilicity(asyn)
        self.assertEqual(expected, actual)