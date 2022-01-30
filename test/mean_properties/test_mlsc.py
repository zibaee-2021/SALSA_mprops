from unittest import TestCase
from data.protein_sequences import read_seqs
from src.mean_properties.mlsc import compute_low_sequence_complexity


class TestComputeSequenceComplexity(TestCase):

    def setUp(self) -> None:
        self.asyn = read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')
        self.tau4N2R = read_seqs.read_protein_sequence_txt('TAU_HUMAN_2N4R.txt')
        self.tdp43 = read_seqs.read_protein_sequence_txt('TADBP_HUMAN_TDP43.txt')

    def test__normalise_to_mbp_AAAA(self):
        actual = compute_low_sequence_complexity('AAAA')
        expected = 2.85
        self.assertEqual(expected, actual)

    def test__normalise_to_mbp_ACDEFGHIKLMNPQRSTVWY(self):
        actual = compute_low_sequence_complexity('ACDEFGHIKLMNPQRSTVWY')
        expected = 0.35
        self.assertEqual(expected, actual)

    def test__normalise_to_mbp_asyn(self):
        actual = compute_low_sequence_complexity(self.asyn)
        expected = 0.3732198
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_asyn(self):
        actual = compute_low_sequence_complexity(self.asyn)
        expected = 0.3732198
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_tau4n2r(self):
        actual = compute_low_sequence_complexity(self.tau4N2R)
        expected = 0.3569252
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_tdp43(self):
        actual = compute_low_sequence_complexity(self.tdp43)
        expected = 0.35
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_10Q(self):
        actual = compute_low_sequence_complexity('QQQQQQQQQQ')
        expected = 2.85
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_20Q(self):
        actual = compute_low_sequence_complexity('QQQQQQQQQQQQQQQQQQQQ')
        expected = 2.85
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_30Q(self):
        actual = compute_low_sequence_complexity('QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ')
        expected = 2.85
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_40Q(self):
        actual = compute_low_sequence_complexity('QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ')
        expected = 2.85
        self.assertAlmostEqual(expected, actual)

    def test__compute_low_sequence_complexity_AAAA(self):
        actual = compute_low_sequence_complexity('AAAA')
        expected = 2.85
        self.assertEqual(expected, actual)

    def test__compute_low_sequence_complexity_AAAAAAAAAAAAAAAAAAAA(self):
        actual = compute_low_sequence_complexity('AAAAAAAAAAAAAAAAAAAA')
        expected = 2.85
        self.assertEqual(expected, actual)

    def test__compute_low_sequence_complexity_ACDEFGHIKLMNPQRSTVWY(self):
        actual = compute_low_sequence_complexity('ACDEFGHIKLMNPQRSTVWY')
        expected = 0.35
        self.assertEqual(expected, actual)

    def test__compute_low_sequence_complexity_ACDEFGHIKLMNPQRSTVW(self):
        actual = compute_low_sequence_complexity('ACDEFGHIKLMNPQRSTVW')
        expected = 0.35
        self.assertNotEqual(expected, actual)