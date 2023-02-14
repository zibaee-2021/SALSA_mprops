import os
from unittest import TestCase
import numpy as np
import pandas as pd
from numpy import genfromtxt
from src.salsa import salsa
from numpy import testing as npt
from src.salsa.Options import Props, DefaultBSC
from root_path import abspath_root


class TestSalsa(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV' \
                    'VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
        self.bsyn = 'MDVFMKGLSMAKEGVVAAAEKTKQGVTEAAEKTKEGVLYVGSKTREGVVQGVASVAEKTKEQASHLGGAV' \
                    'FSGAGNIAAATGLVKREEFPTDLKPEEVAQEAAEEPLIEPLMEPEGESYEDPPQEEYQEYEPEA'
        csv_asyn_top400 = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_top400.csv')
        self.asyn_top400 = genfromtxt(csv_asyn_top400, delimiter=',')
        csv_asyn_summed = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_summed.csv')
        self.asyn_summed = genfromtxt(csv_asyn_summed, delimiter=',')
        self.asyn_id_summed_scores = {'ASYU': self.asyn_summed}
        self.dflt_prms = {'window_len_min': DefaultBSC.window_len_min.value,
                          'window_len_max': DefaultBSC.window_len_max.value,
                          'top_scoring_windows_num': DefaultBSC.top_scoring_windows_num.value,
                          'threshold': DefaultBSC.threshold.value,
                          'abs_threshold': DefaultBSC.abs_threshold.value}

    def test_compute(self):
        csv_asyn = os.path.join(abspath_root, 'test', 'salsa', 'data', 'asyn_bsc_top400.csv')
        expected_2d = genfromtxt(csv_asyn, delimiter=',')
        actual_2d = salsa.compute_all_scored_windows(sequence=self.asyn, _property=Props.bSC.value, params=self.dflt_prms)
        for expected, actual in zip(expected_2d, actual_2d):
            npt.assert_array_equal(expected, actual)

    def test_sum_scores_for_plot_simple(self):
        scored_windows_all = np.ones((2, 2))
        expected = np.array([2.0, 2.0])
        actual = salsa.sum_scores_for_plot(scored_windows_all)
        npt.assert_array_equal(expected, actual)

    def test_plot_summed_scores_simple(self):
        summed_scores1 = np.ones((10,))
        summed_scores0 = np.zeros((20,))
        summed_scores_dict = dict()
        summed_scores_dict['foo'] = summed_scores0
        summed_scores_dict['bar'] = summed_scores1
        salsa.plot_summed_scores(summed_scores_dict, _property='bla', prot_name_labels=['ones', 'zeros'],
                                   params=self.dflt_prms)

    def test_plot_summed_scores_asyn(self):
        salsa.plot_summed_scores(prot_id_summed_scores=self.asyn_id_summed_scores,
                                   _property=Props.bSC.value, prot_name_labels='asyn', params=self.dflt_prms)
        # This is obviously just a visual test, to be assessed by simply seeing that a
        # plot is produced without throwing an error and that it looks correct, by eye.

    def test__get_length_of_longest_prot_seq(self):
        summed_scores = {'ones': np.ones(10), 'zeros': np.zeros(20)}
        expected = 20
        actual, _ = salsa._get_length_of_longest_prot_seq(summed_scores)
        self.assertEqual(expected, actual)

    def test_integrate_salsa_plot_simple(self):
        summed_scores = {'ones': np.ones((10,))}
        expected = 10
        actual = salsa.integrate_salsa_plot(summed_scores)
        self.assertEqual(expected, actual)

    def test_integrate_salsa_plot_asyn(self):
        expected = 3739.2514875 # value taken from original Java implementation of salsa.
        actual = salsa.integrate_salsa_plot(self.asyn_id_summed_scores)
        self.assertEqual(expected, actual)

    def test__compute_bsc_integrals_1_2_threshold(self):# NOTE: Options are currently manually set
        actual = salsa._compute_bsc_integrals(self.asyn)
        self.assertEqual(3513.9, actual)
        actual = salsa._compute_bsc_integrals(self.bsyn)
        self.assertEqual(1145.11, actual)

    def test__compute_bsc_integrals_1_1_5_threshold(self):# NOTE: Options are currently manually set
        actual = salsa._compute_bsc_integrals(self.asyn)
        self.assertEqual(4915.88, actual)

    def test_compute_norm_bsc_integrals(self):
        col_names = ['Synucleins', 'seqs']
        seqs = {'asyn': self.asyn, 'bsyn': self.bsyn}
        input_df = pd.DataFrame.from_dict(data=seqs, orient='index', columns=[col_names[1]])

        col_names = ['Synucleins', 'seqs', 'bsc', 'nbsc']
        seqs = {'asyn': [self.asyn, 3513.9, 1.0], 'bsyn': [self.bsyn, 1145.11, 0.0]}
        expected_df = pd.DataFrame.from_dict(data=seqs, orient='index', columns=[col_names[1:]])

        actual_df = salsa.compute_norm_bsc_integrals(input_df)
        self.assertEqual(expected_df, actual_df)
