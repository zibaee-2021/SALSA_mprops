import math
from pandas import DataFrame as pDF
from src.mean_properties import mbp as mbp
from src.mean_properties import mlsc as lsc
from src.mean_properties import mha as mha
from src.mean_properties import medm as medm
import numpy as np
from matplotlib import pyplot as plt
from src.salsa.Options import Props, DefaultBSC


def integrate_salsa_plot(summed_scores: dict) -> dict:
    integrals = dict()
    for prot_id, summed_scores_ in summed_scores.items():
        integrals[prot_id] = round(summed_scores_.sum(), 2)
    return integrals


# def plot_summed_scores_against_aa_sequence(summed_scores, _property: str, protein_names):
#     TODO
#     Plot the given summed salsa scores against the given protein sequence, using 1-character format for the (sequence)
#     Unlike the plotting function that uses plots this against the amino acid sequence position, this plotting tool
#     cannot plot two overlapping proteins as a result.


def _get_length_of_longest_prot_seq(summed_scores: dict) -> int:
    max_len, index_of_longest = 0, 0
    for prot_id_name, summed_scores_ in summed_scores.items():
        if len(summed_scores_) > max_len:
            max_len = len(summed_scores_)
    return max_len


def plot_summed_scores(prot_id_summed_scores: dict, _property: str, prot_name_labels: list, params: dict):
    """
    NOTE: The pydevd debugger throws a Nonetype exception with matplotlib in this function.
    Plot the given protein(s)' amino acid sequence (numerical position) against the calculated SALSA property from the
    given array of summed scores per residue. The amino acid sequence is on the x-axis.
    :param prot_id_summed_scores: SALSA scores (summed to one per residue), mapped to corresponding protein id/name key.
    :param _property: SALSA property, e.g. beta-strand contiguity.
    :param prot_name_labels: Protein id(s)/name(s) for labelling plot in legend.
    :param params: The SALSA parameters used for generating the scores.
    """
    plt_title = f"{params['window_len_min']}-{params['window_len_max']} | {params['top_scoring_windows_num']} |" \
                f" {'abs('+str(params['threshold'])+')' if params['abs_threshold'] else params['threshold']}" \
                f"{' |'  + str(params['periodicity'])+'°' if params['periodicity'] else ''} "
    assert(len(prot_name_labels) == len(prot_id_summed_scores))
    max_len_xaxis = _get_length_of_longest_prot_seq(prot_id_summed_scores)
    xtick_interval = math.ceil(max_len_xaxis/40)
    fig, ax = plt.subplots()
    for prot_name, summed_scores_ in prot_id_summed_scores.items():
        ax.plot(np.arange(1, len(summed_scores_) + 1), summed_scores_, label=prot_name)
    ax.legend(loc='upper right', fontsize='large', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(np.arange(1, max_len_xaxis + 1, xtick_interval))
    plt.xticks(fontsize=8, rotation=90)
    plt.xlim([1, max_len_xaxis])
    plt.ylim(bottom=0)
    plt.suptitle(plt_title, fontsize=8)
    plt.ylabel(_property)
    plt.xlabel('amino acid sequence')
    plt.show()


def sum_scores_for_plot(scored_windows_all: np.array) -> np.array:
    return scored_windows_all.sum(axis=0)


def _filter_top_scoring_windows(scored_windows_all: np.array, top_scoring_windows_num: int) -> np.array:
    with_values_at_end = np.sort(scored_windows_all, axis=1)
    # print(f'with_values_at_end {with_values_at_end}')
    indices_of_order = np.argsort(with_values_at_end[:, -1])
    # print(f'indices_of_order {indices_of_order}')
    _indices_of_order = np.flip(indices_of_order)
    # print(f'_indices_of_order {_indices_of_order}')
    ordered_windows = scored_windows_all[_indices_of_order]
    # print(f'ordered_windows {ordered_windows}')
    top_scoring_windows = ordered_windows[:top_scoring_windows_num, :]
    # print(f'top_scoring_windows {top_scoring_windows}')
    return top_scoring_windows


def _compute(seq: str, prop: str, ws: int, params: dict) -> np.array:
    """
    Calculate the salsa score for the given protein sequence, using the given option (e.g. mean beta sheet
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

    If the score for the first 2-residue sequence window starting from the N-terminus, (located by index=0 to
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
    :param prop: The option of which property to use for calculating salsa scores. E.g. mean beta-sheet propensity is
    used to compute salsa beta-strand contiguity.
    :param ws: Window size to use for sliding over the sequence.
    :param params: Parameters required for SALSA. Although this holds all 4 or 5 parameters, only `Threshold` and
    `Periodicity` (for mean helical amphipathicity) are required here.
    :return: All salsa scores for given window size.
    """
    num_of_windows = len(seq) - ws + 1
    scored_windows = np.zeros(shape=(num_of_windows, len(seq)), dtype=float)
    score = 0.0

    for i in range(num_of_windows):
        window_seq = seq[i: i + ws]
        if prop == Props.bSC.value:
            score = mbp.compute_mean_beta_sheet_prop(window_seq)
        elif prop == Props.mLSC.value:
            score = lsc.compute_low_sequence_complexity(window_seq)
        elif prop == Props.mHA.value:
            score = mha.compute_mean_helical_amphipathicity(window_seq, periodicity=params['periodicity'])
        elif prop == Props.mEDM.value:
            score = medm.compute_mean_electric_dipole_moment(window_seq, periodicity=params['periodicity'])
        score_ = np.abs(score) if params['abs_threshold'] else score
        if score_ >= params['threshold']:
            scored_windows[i, i:i + ws] = score_
        else:
            continue
    return scored_windows


def _has_valid_window_params(seq: str, window_len_min: int, window_len_max: int) -> bool:
    valid = True
    if window_len_min > len(seq):
        print(f'Invalid window_len_min {window_len_min} is greater than the size of the protein sequence {len(seq)}')
        valid = False
    if window_len_max > len(seq):
        print(f'Invalid window_len_max {window_len_max} is greater than the size of the protein sequence {len(seq)}')
        valid = False
    return valid


def compute(sequence: str, _property: str, params: dict) -> np.array:
    """
    Calculate the salsa score for the given protein sequence, using the given option (e.g. mean beta-sheet
    propensities), sliding aver the sequence with the given range of window sizes and step size, returning only those
    values that score over the given threshold.
    To clarify the meaning of the minimum and maximum window sizes:
    if for example the minimum is 4 and maximum is 20, the algorithm uses all of the following 17 window sizes: 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
    :param sequence: Full protein sequence in 1-letter notation.
    :param _property: The protein property to use for calculating salsa scores. E.g. mean beta-sheet.
    :param params: salsa parameters. They are minimum window size, for a range of window sizes, to use for sliding over
    the sequence.
    :return: All salsa scores for all window sizes in the given range. An array of arrays.
    """
    scored_windows_all = np.empty(shape=(0, len(sequence)), dtype=float)
    if not _has_valid_window_params(seq=sequence, window_len_min=params['window_len_min'],
                                    window_len_max=params['window_len_max']):
        return scored_windows_all
    window_len_increments = 1
    for window_size in range(params['window_len_min'], params['window_len_max'] + 1, window_len_increments):
        scored_windows_for_this_window_size = _compute(seq=sequence, prop=_property, ws=window_size, params=params)
        scored_windows_all = np.concatenate((scored_windows_all, scored_windows_for_this_window_size), axis=0)
    scored_windows_all = _filter_top_scoring_windows(scored_windows_all, params['top_scoring_windows_num'])
    return scored_windows_all


def _compute_bsc_integrals(seq: str) -> float:
    """
    Compute SALSA beta-strand contiguity integral for the given sequence.
    :param seq: Protein sequence in 1-letter notation.
    :return: SALSA beta-strand contiguity integral.
    """
    scored_windows_all = compute(sequence=seq, _property=Props.bSC.value, params=DefaultBSC.all_params.value)
    summed_scores = sum_scores_for_plot(scored_windows_all)
    return integrate_salsa_plot({'seq': summed_scores})['seq']


def compute_norm_bsc_integrals(df: pDF) -> pDF:
    """

    :param df:
    :return:
    ['lagtime_means', 'ln_lags', 'seqs', 'mbp', 'mh', 'mnc', 'mtc', 'nmbp', 'nmh', 'nmnc', 'nmtc', 'mprops',
    'nmprops', 'pred']
    """
    # df['bsc'] = df['seqs'].apply(_compute_bsc_integrals)
    df_ = df.copy()
    df_.loc[:, 'bsc'] = df['seqs'].apply(_compute_bsc_integrals)
    max_ = np.max(list(df_['bsc']))
    min_ = np.min(list(df_['bsc']))
    # df['nbsc'] = df['bsc'].apply(lambda row: (row - min_) / (max_ - min_))
    df__ = df_.copy()
    df__.loc[:, 'nbsc'] = df_['bsc'].apply(lambda row: (row - min_) / (max_ - min_))
    return df__


# if __name__ == '__main__':
    # from data.protein_sequences import read_seqs
    # from src.salsa.Options import DefaultBSC
    # _acc = ['P37840']
    # _names = ['']
    # _prot_id_seqs = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=_acc)
    # # STEP 1 - Define property and corresponding parameters.
    # _property = Props.bSC.value
    # _params = {'window_len_min': DefaultBSC.window_len_min.value,
    #            'window_len_max': DefaultBSC.window_len_max.value,
    #            'top_scoring_windows_num': DefaultBSC.top_scoring_windows_num.value,
    #            'threshold': DefaultBSC.threshold.value}
    # # STEP 2 - salsa produces an array holding a single numbers for each residue.
    # _all_summed_scores = dict()
    # for prot_id, prot_seq in _prot_id_seqs.items():
    #     _scored_windows_all = compute(sequence=prot_seq, _property=_property, params=_params)
    #     _summed_scores = sum_scores_for_plot(_scored_windows_all)
    #     _all_summed_scores[prot_id] = _summed_scores
    #
    # # STEP 3
    # # currently only able to plot one protein per plot.
    # plot_summed_scores(_all_summed_scores, _property, prot_name_labels=list(_all_summed_scores.keys()))

    # summed_scores1 = np.ones((10,))
    # summed_scores0 = np.zeros((20,))
    # summed_scores_dict = dict()
    # summed_scores_dict['blo'] = summed_scores1
    # summed_scores_dict['bli'] = summed_scores0
    # plot_summed_scores(summed_scores_dict, _property='bla', protein_names=['ones', 'zeros'])
    #
    # from data.protein_sequences import read_seqs
    # import time
    # st = time.time()
    # # _aa_props = read_props.read_aa_props_csv()
    # prot_seqs = read_seqs.read_protein_sequences_csv()
    # _protein_name = 'SYUA_HUMAN'
    # _protein_name = 'PRIO_HUMAN'
    # _protein_name = 'HETS_PODAS'
    # _protein_name = 'K4HY00_YEASX'
    # _protein_name = 'URE2_YEAST'
    # _property = Props.bSC.value
    # _property = Props.LSC.value
    # protein = prot_seqs.loc[(prot_seqs.name == _protein_name)].iloc[0]['sequence']
    # print(protein)
    # params = {'window_len_min': 4, 'window_len_max': 20, 'top_scoring_windows_num': 400, 'threshold': 1}
    # _scored_windows_all = compute(sequence=protein, _property=_property, params=params)
    # num_rows, num_cols = _scored_windows_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    #
    # _summed_scores = sum_scores_for_plot(_scored_windows_all)
    # plot_summed_scores(summed_scores=_summed_scores, _property=_property, protein_names=_protein_name)
    #
    # print(f'{round(1000 * (time.time() - st), 1)} ms')
    # # _salsa_integral = integrate_salsa_plot(_summed_scores)
    # print(f'integral {_salsa_integral}')

    # print(f'salsa_integral {_salsa_integral}')
    # _scored_windows_all = compute_salsa(sequence=_tdp43, option=SalsaOptions.SALSA_LSC.value,
    #                                      top_scoring_windows_num=4000, threshold=0.5,
    #                                      window_len_min=10, window_len_max=30)
    # _salsa_integral = sum_scores_for_plot(_scored_windows_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='tdp43')

    # _scored_windows_all = compute_salsa(sequence='QQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value,
    #                                      window_len_max=10)
    # num_rows, num_cols = _scored_windows_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_windows_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='10Q')
    #
    # _scored_windows_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_windows_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_windows_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='20Q')
    #
    # _scored_windows_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_windows_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_windows_all)
    # plot_salsa_integral(salsa_integral=_salsa_integral,
    #                     option_used=SalsaOptions.SALSA_LSC.value,
    #                     protein_name='30Q')
    #
    # _scored_windows_all = compute_salsa(sequence='QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ',
    #                                      option=SalsaOptions.SALSA_LSC.value)
    # num_rows, num_cols = _scored_windows_all.shape
    # print(f'num_rows is {num_rows}, num_cols is {num_cols}')
    # print(f'{round(1000*(time.time() - st), 1)} ms')
    # _salsa_integral = integrate_salsa_plot(_scored_windows_all)
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
