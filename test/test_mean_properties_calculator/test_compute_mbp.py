from unittest import TestCase
from data.protein_sequences import read_seqs
from src.mean_properties_calculator.compute_mean_betasheet_propensity import compute_mean_beta_sheet_prop


class TestComputeMBP(TestCase):

    def setUp(self) -> None:
        self.asyn = read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt')

    def test__compute_mean_beta_sheet_prop(self):
        actual = compute_mean_beta_sheet_prop(sequence=self.asyn[36:40])
        print(f'asyn[36:40] {self.asyn[36:40]}')
        expected = 2.1260364842454393
        self.assertEqual(expected, actual)

    def test__compute_mean_beta_sheet_prop_2(self):
        actual = compute_mean_beta_sheet_prop(sequence=read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt'))
        expected = 0.9179316215065967
        self.assertEqual(expected, actual)


   # def test__compute_mean_beta_sheet_prop_2(self):
   #      actual = compute_mean_beta_sheet_prop(sequence=read_seqs.read_protein_sequence_txt('SYUA_HUMAN.txt'))
   #      expected = 2.1260364842454393
   #      self.assertEqual(expected, actual)
