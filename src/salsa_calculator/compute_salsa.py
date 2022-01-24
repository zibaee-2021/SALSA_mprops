from enum import Enum, unique
from typing import List
from src.mean_properties_calculator import compute_mean_betasheet_propensity as mbp
from src.mean_properties_calculator import compute_sequence_complexity as lsc
from src.mean_properties_calculator import compute_mean_helical_amphipathicity as mha
import numpy as np
from matplotlib import pyplot as plt
from src.steps.SALSA.salsa_prot_props import SalsaProtProps


def integrate_salsa_plot(summed_scores: np.array) -> np.array:
    return round(summed_scores.sum(), 2)


def plot_summed_salsa(summed_scores, _property: str, protein_name) -> None:
    plt.plot(np.arange(1, len(summed_scores) + 1), summed_scores)
    plt.xticks(np.arange(1, len(summed_scores), 10))
    plt.xticks(fontsize=8, rotation=90)
    plt.title(protein_name)
    plt.ylabel(_property)
    plt.xlabel('amino acid sequence')
    plt.show()


def sum_scores_for_plot(scored_peptides_all: np.array) -> np.array:
    return scored_peptides_all.sum(axis=0)


def _compute_salsa(seq: str, opt: str, ws: int, ss: int, thd: float) -> np.array:
    """
    Calculate the SALSA score for the given protein sequence, using the given option (e.g. mean beta sheet
    propensities), sliding aver the sequence with the given window size and step size, returning only those values
    that score over the given threshold.
    Worked example for a 5-residue protein sequence ACDEF, a window size of 2, a step size of 1 and a threshold of
    1.2:
    Therefore len(seq) is 5, and `number_of_windows` is 5-2+1 = 4. The calculated scores are assigned to a 2D array
    representing one amino acid per "column" position (j) in the array. The number of "rows" (i) depends on the size
    of the window (stored in `number_of_windows`), hence the dimensions are (4 x 5) for ACDEF:

                     A C D E F
                  j  0 1 2 3 4
                i
                0    0 0 0 0 0
                1    0 0 0 0 0
                2    0 0 0 0 0
                3    0 0 0 0 0

    If the score for the first 2-residue peptide window starting from the N-terminus, (located by index=0 to
    index=1 of protein sequence) is 1.3. Row i=0 get this same value for every residue in that window, hence at j=0
    and j=1. If the next window scores 4.7, the next 2.5 and last one 1.4, the resulting 2D should look like this (
    note the values are floats, not integers):

                      A   C   D   E   F
                  j   0   1   2   3   4
                i
                0    1.3 1.3 0.0 0.0 0.0
                1    0.0 4.7 4.7 0.0 0.0
                2    0.0 0.0 2.5 2.5 0.0
                3    0.0 0.0 0.0 1.4 1.4

    :param seq: Full protein sequence in 1-letter notation.
    :param opt: The option of which property to use for calculating SALSA scores. E.g. mean beta-sheet propensity is
    used to compute SALSA beta-strand contiguity.
    :param ws: Window size to use for sliding over the sequence.
    :param ss: Step size for sliding the window over the sequence.
    :param thd: Threshold value of peptide scores. Any that score below this are discarded.
    :return: All SALSA scores for given window size.
    """
    num_of_windows = len(seq) - ws + 1
    scored_peptides = np.zeros(shape=(num_of_windows, len(seq)), dtype=float)
    score = 0.0

    for i in range(num_of_windows):
        peptide_seq = seq[i: i + ws]
        if opt == SalsaProtProps.SALSA_bSC.value:
            score = mbp.compute_mean_beta_sheet_prop(peptide_seq)
        elif opt == SalsaProtProps.SALSA_LSC.value:
            score = lsc.compute_low_sequence_complexity(peptide_seq)
        elif opt == SalsaProtProps.SALSA_mHA.value:
            score = mha.compute_mean_helical_amphipathicity(peptide_seq, periodicity=100)
        if score >= thd:
            scored_peptides[i, i:i + ws] = score
        else:
            continue
    return scored_peptides


def _filter_top_scoring_peptides(scored_peptides_all: np.array, top_scoring_peptides_num: int) -> np.array:
    with_values_at_end = np.sort(scored_peptides_all, axis=1)
    # print(f'with_values_at_end {with_values_at_end}')
    indices_of_order = np.argsort(with_values_at_end[:, -1])
    # print(f'indices_of_order {indices_of_order}')
    _indices_of_order = np.flip(indices_of_order)
    # print(f'_indices_of_order {_indices_of_order}')
    ordered_peptides = scored_peptides_all[_indices_of_order]
    # print(f'ordered_peptides {ordered_peptides}')
    result = ordered_peptides[:top_scoring_peptides_num, :]
    # print(f'result {result}')
    return result


def compute_salsa(sequence: str, option: str, window_len_min: int, window_len_max: int,
                  top_scoring_peptides_num: int, threshold: float, step_size=1) -> np.array:
    scored_peptides_all = np.empty(shape=(0, len(sequence)), dtype=float)
    """
    Calculate the SALSA score for the given protein sequence, using the given option (e.g. mean beta-sheet
    propensities), sliding aver the sequence with the given range of window sizes and step size, returning only those
    values that score over the given threshold.
    To clarify the meaning of the minimum and maximum window sizes:
    if for example the minimum is 4 and maximum is 20, the algorithm uses all of the following 17 window sizes: 4, 5, 
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
    :param sequence: Full protein sequence in 1-letter notation.
    :param option: The option of which property to use for calculating SALSA scores. E.g. mean beta-sheet propensity is
    used to compute SALSA beta-strand contiguity.
    :param window_len_min: The minimum window size, for a range of window sizes, to use for sliding over the 
    sequence. 
    :param window_len_max: The maximum window size, for a range of window sizes, to use for sliding over the 
    sequence. 
    :param top_scoring_peptides_num: The number of highest scoring peptides to include in the SALSA calculation. All 
    other peptide scores are discarded.
    :param threshold: Threshold value of peptide scores. Any that score below this are discarded.
    :param step_size: Step size for sliding the window over the sequence. Default is 1.
    :return: All SALSA scores for all window sizes in the given range.
    """
    if window_len_min > len(sequence):
        print(f'Invalid window_len_min {window_len_min} is greater than the size of the protein sequence'
              f' {len(sequence)}')
        return scored_peptides_all
    if window_len_max > len(sequence):
        print(f'Invalid window_len_max {window_len_max} is greater than the size of the protein sequence'
              f' {len(sequence)}')
        return scored_peptides_all
    window_len_increments = 1
    for window_size in range(window_len_min, window_len_max + 1, window_len_increments):
        scored_peptides_window_size = _compute_salsa(seq=sequence, opt=option, ws=window_size, ss=step_size,
                                                     thd=threshold)
        scored_peptides_all = np.concatenate((scored_peptides_all, scored_peptides_window_size), axis=0)
        scored_peptides_all = _filter_top_scoring_peptides(scored_peptides_all, top_scoring_peptides_num)
    return scored_peptides_all


def compute_low_sequence_complexity(sequence: str, window_len_min=4, window_len_max=20, top_scoring_peptides_num=400,
                                    threshold=1.2) -> np.array:
    return compute_salsa(sequence=sequence, option=SalsaProtProps.SALSA_LSC.value, window_len_min=window_len_min,
                         window_len_max=window_len_max, top_scoring_peptides_num=top_scoring_peptides_num,
                         threshold=threshold)


def compute_mean_helical_amphipathicity(sequence: str, window_len_min=11, window_len_max=33,
                                        top_scoring_peptides_num=10000, threshold=0.8) -> np.array:
    return compute_salsa(sequence=sequence, option=SalsaProtProps.SALSA_mHA.value, window_len_min=window_len_min,
                         window_len_max=window_len_max, top_scoring_peptides_num=top_scoring_peptides_num,
                         threshold=threshold)


def compute_beta_strand_contiguity(sequence: str, window_len_min=4, window_len_max=20, top_scoring_peptides_num=400,
                                   threshold=1.2) -> np.array:
    """
    Calculate the beta-Strand Contiguity score for the given protein sequence with SALSA,
    :param sequence: Full protein sequence in 1-letter notation.
    :param window_len_min: The minimum window size, for a range of window sizes, to use for sliding over the
    sequence. Default is 4.
    :param window_len_max: The maximum window size, for a range of window sizes, to use for sliding over the
    sequence. Default is 20.
    :param top_scoring_peptides_num: The number of highest-scoring peptides to include in the SALSA calculation. All
    other peptide scores are discarded. Default is 400.
    :param threshold: Threshold value of peptide scores. Any that score below this are discarded. Default is 1.2.
    :return: All SALSA scores for all window sizes in the given range.
    """
    return compute_salsa(sequence=sequence, option=SalsaProtProps.SALSA_bSC.value, window_len_min=window_len_min,
                         window_len_max=window_len_max, top_scoring_peptides_num=top_scoring_peptides_num,
                         threshold=threshold)


def _has_valid_window_params(sequence: str, window_len_min: int, window_len_max: int) -> bool:
    valid = True
    if window_len_min > len(sequence):
        print(f'Invalid window_len_min {window_len_min} is greater than the size of the protein sequence'
              f' {len(sequence)}')
        valid = False
    if window_len_max > len(sequence):
        print(f'Invalid window_len_max {window_len_max} is greater than the size of the protein sequence'
              f' {len(sequence)}')
        valid = False
    return valid


def compute(sequence: str, _property: str, window_len_min=None, window_len_max=None,
            top_scoring_peptides_num=None,
            threshold=None) -> np.array:
    """
    Compute given property using SALSA for given sequence.
    :param _property:
    :param threshold:
    :param top_scoring_peptides_num:
    :param window_len_max:
    :param window_len_min:
    :param sequence: Protein sequence in 1-character format.
    :param property: Protein property to measure using SALSA.
    :return: SALSA score of the given property for each residue in the given protein.
    """
    if not _has_valid_window_params(sequence, window_len_min, window_len_max):
        return np.empty(shape=(0, len(sequence)), dtype=float)
    if _property == SalsaProtProps.bSC.value:
        return compute_beta_strand_contiguity(protein,
                                              window_len_min=window_len_min,
                                              window_len_max=window_len_max,
                                              top_scoring_peptides_num=top_scoring_peptides_num,
                                              threshold=threshold)
    if _property == SalsaProtProps.mHA.value:
        return compute_mean_helical_amphipathicity(protein, window_len_min=window_len_min,
                                                   window_len_max=window_len_max,
                                                   top_scoring_peptides_num=top_scoring_peptides_num,
                                                   threshold=threshold)
    if _property == SalsaProtProps.SALSA_LSC:
        return compute_low_sequence_complexity(protein,
                                               window_len_min=window_len_min,
                                               window_len_max=window_len_max,
                                               top_scoring_peptides_num=top_scoring_peptides_num,
                                               threshold=threshold)


def run(accs, prot_names, _property: str) -> np.array:
    """
    Use SALSA to compute beta-strand contiguity of given protein(s) identified either by accession number(s) and/or
    protein name(s). Beta-strand contiguity is returned a sum of scores for each residue, so it can be used for
    generating an x-y plot and can be used to calculate the integral of that to give a single scalar SALSA score for
    a protein.
    :param accs: Protein accession numbers according to UniProt, e.g. 'Q16143'. Given as string or list of strings.
    :param prot_names: Protein names according to UniProt mnemonic names, e.g. 'SYUA_HUMAN'. String or list of strings.
    :return:
    """
    prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(accs, prot_names)
    all_summed_scores = dict()
    for prot_id, prot_seq in prot_id_seqs:
        print(f'prot_seq {prot_seq}')
        print(f'type(prot_seq) {type(prot_seq)}')
        scored_peptides_all = compute(sequence=prot_seq, _property=_property)
        summed_scores = sum_scores_for_plot(scored_peptides_all)
        all_summed_scores[prot_id] = summed_scores
    return all_summed_scores


from data.protein_sequences import read_seqs

if __name__ == '__main__':
    import time
    # _aa_props = read_props.read_aa_props_csv()
    prot_seqs = read_seqs.read_protein_sequences_csv()
    prot_seqs.head()
    protein = prot_seqs.loc[(prot_seqs.name == 'PRIO_HUMAN')].iloc[0]['sequence']
    print(protein)
    # # protein = prot_seqs.loc[(prot_seqs.name == 'PRIO_MOUSE')].iloc[0]['sequence']
    # protein = prot_seqs.loc[(prot_seqs.name == 'SYUA_HUMAN')].iloc[0]['sequence']
    # protein = prot_seqs.loc[(prot_seqs.name == 'SYUB_HUMAN')].iloc[0]['sequence']
    # print(protein)
    # __property = SalsaOptions.SALSA_mHA.value
    # st = time.time()
    # _scored_peptides_all = compute(sequence=protein, _property=__property)
    # num_rows, num_cols = _scored_peptides_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000 * (time.time() - st), 1)} ms')
    # _summed_salsa = sum_scores_for_plot(_scored_peptides_all)
    # plot_summed_salsa(salsa_integral=_summed_salsa,
    #                   option_used=__property,
    #                   protein_name='asyn')
    #
    # _salsa_integral = integrate_salsa_plot(_summed_salsa)

    # print(f'salsa_integral {_salsa_integral}')
    # _scored_peptides_all = compute_salsa(sequence=_tdp43, option=SalsaOptions.SALSA_LSC.value,
    #                                      top_scoring_peptides_num=4000, threshold=0.5,
    #                                      window_len_min=10, window_len_max=30)
    # _salsa_integral = sum_scores_for_plot(_scored_peptides_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='tdp43')

    # _scored_peptides_all = compute_salsa(sequence='QQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value,
    #                                      window_len_max=10)
    # num_rows, num_cols = _scored_peptides_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_peptides_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='10Q')
    #
    # _scored_peptides_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_peptides_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_peptides_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='20Q')
    #
    # _scored_peptides_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_peptides_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_peptides_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='30Q')
    #
    # _scored_peptides_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_peptides_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_peptides_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='40Q')

    # x = np.arange(1, len(_asyn_seq) + 1)
    # print(x)
    # print(_salsa_integral)
    # plt.plot(x, _salsa_integral)
    # plt.title('SYUA_HUMAN')
    # plt.ylabel('beta-Strand Contiguity')
    # plt.xlabel('amino acid sequence')
    # plt.show()

    # aa_props.plot.bar(
    #     x="aa",
    #     y=["Pbeta", "Palpha", "Pturn"],
    #     rot=0,
    #     figsize=(12, 6),
    #     ylabel="CF Prefs",
    #     title="Chou-Fasman conformational preferences per amino acid",
    # )
    # plt.show()
    #
