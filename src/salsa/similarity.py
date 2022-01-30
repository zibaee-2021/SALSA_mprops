import numpy as np
"""
Generate a scalar score of similarity between two or more SALSA summed scores, that 
gives an indication of similarity in terms of both the relative positions of peaks to each other
as well as their magnitude (and therefore shape).
"""


def _calc_sum_of_products_of_summed_scores(base: np.array, query: np.array) -> np.array:
    """
    Calculate the sum of products of summed scores', by taking the product of the query sequence's scores and a
    sliding window (of the same length as the query) over the base sequence's scores.
    :param base: SALSA summed_scores, expected to be for the whole sequence of one protein.
    :param query: SALSA summed scores' query window.
    :return: The sum of the products of the base and query. Has the same length as the base summed scores.
    """
    base_len = len(base)
    query_len = len(query)
    assert(base_len >= query_len)
    num_of_product_windows = base_len - query_len + 1
    products_of_summed_scores = np.zeros((num_of_product_windows, base_len))
    for i in range(num_of_product_windows):
        prod = base[i: i + query_len] * query
        products_of_summed_scores[i, i: i + query_len] = np.sum(prod)
        # products_of_summed_scores[i, i: i + query_len] = prod
    return np.sum(products_of_summed_scores, axis=0)


def compute_sum_of_products_of_summed_scores(base_summed_scores: np.array, query_summed_scores: np.array,
                                             query_window_len_min: int, query_window_len_max: int) -> np.array:
    """
    Generate a similarity score between two SALSA summed scores.
    :param base_summed_scores: SALSA summed_scores, expected to be for the whole sequence of one protein.
    :param query_summed_scores: SALSA summed_scores, expected to be for the whole of the other protein's sequence.
    :param query_window_len_min: Minimum window length (in range of windows) for query_summed_scores.
    :param query_window_len_max: Maximum window length (in range of windows) for query_summed_scores.
    :return:
    """
    base_len = len(base_summed_scores)
    assert(base_len >= query_window_len_min)
    assert(query_window_len_min <= query_window_len_max)
    num_of_prods = np.sum([len(query_summed_scores) - ws + 1
                           for ws in range(query_window_len_min, query_window_len_max + 1)])
    sums_of_products = np.zeros((num_of_prods, base_len))
    pos_in_final = 0
    for query_window_len in range(query_window_len_min, min(base_len, query_window_len_max) + 1):
        num_of_query_windows = len(query_summed_scores) - query_window_len + 1
        for i in range(num_of_query_windows):
            query = query_summed_scores[i: i + query_window_len]
            sum_of_prods = _calc_sum_of_products_of_summed_scores(base_summed_scores, query)
            sums_of_products[pos_in_final, :] = sum_of_prods
            pos_in_final += 1
    return np.sum(sums_of_products, axis=0)


    def find_peaks(summed_scores)