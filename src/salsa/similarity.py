from typing import Tuple
import numpy as np

from salsa import compute
import salsa
from data.protein_sequences import read_seqs
from Options import Props, DefaultBSC
"""
Functions for detecting the similarity between two SALSA plots.
The utility of this module is not determined or clear at this moment.
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


def _find_peaks(prot_id_summed_scores: dict[str: np.array], min_cut_off: float) -> dict[str: Tuple[str]]:
    """
    For the given protein's summed scores, identify the start and end positions of peaks, identified as regions
    scoring above the given cut off.
    :param prot_id_summed_scores: SALSA summed scores mapped to the protein name/id.
    :param min_cut_off: Ignore peaks that do not surpass this cut-off value.
    :return: Peaks of the given SALSA summed scores, mapped to its protein id,
    e.g. of the format: 'SYUA_HUMAN': (11-34, 56-59, 83-124)
    """
    tuple_of_peaks = ()
    for prot_id, summed_scores in prot_id_summed_scores.items():
        start_pos = None
        is_peak = False
        for i, score in enumerate(summed_scores):
            if score > min_cut_off and not is_peak:
                is_peak = True
                if start_pos is None:
                    start_pos = i + 1
            elif score <= min_cut_off and is_peak:
                end_pos = i + 1
                is_peak = False
                tuple_of_peaks += (str(start_pos) + '-' + str(end_pos), )
                start_pos = None
            elif is_peak and i == len(summed_scores) - 1:
                end_pos = i + 1
                tuple_of_peaks += (str(start_pos) + '-' + str(end_pos),)
    prot_id, = prot_id_summed_scores.keys()
    return {prot_id: tuple_of_peaks}


def find_peaks(prot_ids_summed_scores: dict[str: np.array], min_cut_off: float) -> dict:
    """
    For the given proteins' summed scores, identify the start and end positions of peaks, identified as regions
    scoring above the given cut off.
    :param prot_ids_summed_scores: SALSA summed scores mapped to the proteins' names/ids.
    :param min_cut_off: Ignore peaks that do not surpass this cut-off value.
    :return: Peaks of the given SALSA summed scores mapped to the corresponding proteins,
    e.g. of the format: 'SYUA_HUMAN': (11-34, 56-59, 83-124), 'SYUB_HUMAN': (11-34, 83-124)
    """
    prot_ids_peaks = dict()
    for prot_id, summed_scores in prot_ids_summed_scores.items():
        prot_ids_peaks.update(_find_peaks({prot_id: summed_scores}, min_cut_off))
    return prot_ids_peaks


def _detect_similarity_by_sharpened_cosine_distance():
    """
    A feature detector shown to have advantages over convolution by Brandon Rohrer in a tweet in 24Feb20, `@_brohrer_`.
    It should allow you to run a peak from the SALSA plot of one protein over the SALSA plot of another protein by
    sliding window and detect the similarity of peaks.
    :return:
    """
    # TODO




if __name__ == '__main__':
    # prot_seq = read_seqs.read_protein_sequence_txt(read_seqs.TxtFiles.ASYN.value)
    prot_ids = 'P10636-8'
    prot_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_ids)
    scored_windows_all = compute(sequence=prot_seq[prot_ids], _property=Props.bSC.value,
                                 params=DefaultBSC.all_params.value)
    summed_scores = execute.sum_scores_for_plot(scored_windows_all)
    blabla = dict()
    blabla['protein_A'] = summed_scores

    peaks = find_peaks(blabla, 10)
    print(peaks)




