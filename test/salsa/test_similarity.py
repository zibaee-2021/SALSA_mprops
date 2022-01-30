from unittest import TestCase
from src.salsa import similarity
import numpy as np
from numpy import testing as npt


class TestSimilarity(TestCase):

    def test__calc_sum_of_products_of_summed_scores(self):
        base = np.array([0, 0, 2, 0, 0])
        query = np.array([2])
        actual = similarity._calc_sum_of_products_of_summed_scores(base=base, query=query)
        expected = np.array([0, 0, 4, 0, 0])
        npt.assert_array_equal(expected, actual)

    def test_compute_sum_of_products_of_summed_scores(self):
        base = np.array([0, 0, 2, 0, 0])
        query = np.array([0, 0, 2, 0, 0])
        actual = similarity.compute_sum_of_products_of_summed_scores(base_summed_scores=base,
                                                                     query_summed_scores=query,
                                                                     query_window_len_min=5, query_window_len_max=5)
        expected = np.array([0, 0, 4, 0, 0])
        npt.assert_array_equal(expected, actual)

    def test_compute_sum_of_products_of_summed_scores2(self):
        base = np.array([0, 0, 2, 0, 0])
        query = np.array([0, 1, 2, 0, 0])
        actual = similarity.compute_sum_of_products_of_summed_scores(base_summed_scores=base,
                                                                     query_summed_scores=query,
                                                                     query_window_len_min=4, query_window_len_max=4)
        expected = np.array([0, 0, 10, 0, 0])
        npt.assert_array_equal(expected, actual)

    def test_compute_sum_of_products_of_summed_scores3(self):
        base = np.array([0, 0, 2, 0, 0])
        query = np.array([0, 1, 2, 0, 0])
        actual = similarity.compute_sum_of_products_of_summed_scores(base_summed_scores=base,
                                                                     query_summed_scores=query,
                                                                     query_window_len_min=4, query_window_len_max=5)
        expected = np.array([0, 0, 14, 0, 0])
        npt.assert_array_equal(expected, actual)

    def test_compute_sum_of_products_of_summed_scores4(self):
        base = np.array([0, 0, 2, 0, 0, 1, 2])
        query = np.array([0, 1, 2, 0, 0])
        actual = similarity.compute_sum_of_products_of_summed_scores(base_summed_scores=base,
                                                                     query_summed_scores=query,
                                                                     query_window_len_min=4, query_window_len_max=5)
        expected = np.array([0, 0, 18, 0, 0, 2, 0])
        npt.assert_array_equal(expected, actual)