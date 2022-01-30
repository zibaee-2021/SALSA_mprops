from unittest import TestCase
from data.protein_sequences import read_seqs
from src.mean_properties.mbp import compute_mean_beta_sheet_prop


class TestMBP(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV' \
                    'VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

    def test__compute_mean_beta_sheet_prop(self):
        actual = compute_mean_beta_sheet_prop(sequence=self.asyn[36:40])
        print(f'asyn[36:40] {self.asyn[36:40]}')
        expected = 2.1260364842454393
        self.assertEqual(expected, actual)

    def test__compute_mean_beta_sheet_prop_2(self):
        actual = compute_mean_beta_sheet_prop(sequence=self.asyn)
        expected = 0.9179316215065967
        self.assertEqual(expected, actual)


   # def test__compute_mean_beta_sheet_prop_2(self):
   #      actual = compute_mean_beta_sheet_prop(self.asyn)
   #      expected = 2.1260364842454393
   #      self.assertEqual(expected, actual)
