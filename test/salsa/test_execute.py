import os
from unittest import TestCase
import numpy as np
from numpy import savetxt, genfromtxt
from src.salsa import execute
from numpy import testing as npt
from src.salsa.Options import Props, DefaultBSC
from root_path import abspath_root


class Test(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV' \
                    'VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'

        csv_asyn_top400 = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_top400.csv')
        self.asyn_top400 = genfromtxt(csv_asyn_top400, delimiter=',')
        csv_asyn_summed = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_summed.csv')
        self.asyn_summed = genfromtxt(csv_asyn_summed, delimiter=',')

    def test_compute(self):
        _property = Props.bSC.value
        params = {'window_len_min': DefaultBSC.window_len_min.value,
                  'window_len_max': DefaultBSC.window_len_max.value,
                  'top_scoring_windows_num': DefaultBSC.top_scoring_windows_num.value,
                  'threshold': DefaultBSC.threshold.value}
        csv_asyn = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_top400.csv')
        expected_2d = genfromtxt(csv_asyn, delimiter=',')
        actual_2d = execute.compute(sequence=self.asyn, _property=Props.bSC.value, params=params)
        for expected, actual in zip(expected_2d, actual_2d):
            npt.assert_array_equal(expected, actual)

    def test_sum_scores_for_plot_simple(self):
        scored_windows_all = np.ones((2, 2))
        expected = np.array([2.0, 2.0])
        actual = execute.sum_scores_for_plot(scored_windows_all)
        npt.assert_array_equal(expected, actual)

    def test_plot_summed_scores_simple(self):
        summed_scores1 = np.ones((10,))
        summed_scores0 = np.zeros((20,))
        summed_scores_dict = dict()
        summed_scores_dict['bli'] = summed_scores0
        summed_scores_dict['blo'] = summed_scores1
        execute.plot_summed_scores(summed_scores_dict, _property='bla', protein_names=['ones', 'zeros'])

    def test_plot_summed_scores_asyn(self):
        execute.plot_summed_scores(summed_scores=self.asyn_summed, _property=Props.bSC.value, protein_names='asyn')
        # This is obviously just a visual test, to be assessed by simply seeing that a
        # plot is produced without throwing an error and that it looks correct, by eye.

    def test__get_length_of_longest_prot_seq(self):
        summed_scores = [np.ones(10), np.zeros(20)]
        expected = 20
        actual, _ = execute._get_length_of_longest_prot_seq(summed_scores)
        self.assertEqual(expected, actual)

    def test__move_longest_seq_to_first_in_list(self):
        summed_scores = [np.ones(10), np.zeros(20)]
        expected = [np.zeros(20), np.ones(10)]
        execute._move_longest_seq_to_first_in_list(summed_scores, index_of_longest=1)
        actual = summed_scores
        self.assertCountEqual(expected[0], actual[0])
        npt.assert_array_equal(expected[0], actual[0])
        self.assertCountEqual(expected[1], actual[1])
        npt.assert_array_equal(expected[1], actual[1])

    def test_integrate_salsa_plot_simple(self):
        summed_scores = np.ones((10,))
        expected = 10
        actual = execute.integrate_salsa_plot(summed_scores)
        self.assertEqual(expected, actual)

    def test_integrate_salsa_plot_asyn(self):
        expected = 3739.2514875 # value taken from original Java implementation of salsa.
        actual = execute.integrate_salsa_plot(self.asyn_summed)
        self.assertEqual(expected, actual)