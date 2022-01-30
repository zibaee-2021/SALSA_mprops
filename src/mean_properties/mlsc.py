

def compute_low_sequence_complexity(sequence: str) -> float:
    """
    Compute a low-sequence-complexity score, normalised to the same numerical range as mean beta-sheet propensity
    scores.
    For a sequence of identical residues, the low-sequence-complexity score is its maximum valye.
    For a sequence of different residues, the low-sequence-complexity score is its minimum valye.
    e.g.
    AAAAA scores the maximum possible value of low-sequence-complexity, equal to maximum Mean Beta-Sheet Propensity.
    ACDEF scores the minimum possible value of low-sequence-complexity, equal to minimum Mean Beta-Sheet Propensity.
    number of unique residues / number of residues
    AAAAA: 1 --> reciprocal = 1
    AAAAAAAAAA: 1 --> reciprocal = 1
    AAAAAAAAAC: 2 --> reciprocal = 0.5
    ACDEF: 5  --> reciprocal = 1/5 = 0.2
    ACDEFGHIKL:10 --> reciprocal = 1/10 = 0.1
    ACDEFACDEF: 5 --> reciprocal = 1/5 = 0.2
    Take the reciprocal of the value such that a higher value corresponds to a lower sequence complexity and vice versa.
    Calculate number of total residues by len(sequence). Calculate number of unique residues by len(set(sequence))

    The highest possible sequence complexity is 20. So the lowest low-sequence-complexity score possible is 0.05
    The lowest possible sequence complexity is 1. So the highest low-sequence-complexity score possible is 1.0

    :param sequence:
    :return:
    """
    lsc_score = 1 / len(set(sequence))
    return _normalise_to_mbp(lsc_score)


def _normalise_to_mbp(lsc_score: float) -> float:
    """
    Normalise low-sequence-complexity (LSC) score to match the numerical range of mean beta-sheet propensity (MBP)
    scores.
    Lowest possible LSC (ACDEFGHIKLMNPQRSTVWY) is 0.05. Highest possible LSC (AAAA...etc) is 1.0.
    Lowest possible MBP (DDDD) is 0.35. Highest possible MBP (VVVV) is 2.85.
    Normalise by first scaling LSC range of 0.05-1.0 to MBP range 0.35-2.85.
    Then translate scaled LSC values by the difference between the lowest MBP score and lowest scaled LSC score.

    :param lsc_score: Low-sequence-complexity score, not normalised.
    :return: Normalised low-sequence-complexity score.
    """
    lowest_possible_lsc_score = 0.05
    highest_possible_lsc_score = 1.0
    lowest_possible_mbp_score = 0.35
    highest_possible_mbp_score = 2.85
    scaling_range_mbp = highest_possible_mbp_score - lowest_possible_mbp_score
    scaling_range_lsc = highest_possible_lsc_score - lowest_possible_lsc_score
    scaling_factor = scaling_range_mbp / scaling_range_lsc
    lowest_possible_scaled_lsc_score = scaling_factor * lowest_possible_lsc_score
    translation_factor = lowest_possible_mbp_score - lowest_possible_scaled_lsc_score
    scaled_lsc_score = lsc_score * scaling_factor
    return scaled_lsc_score + translation_factor

# if __name__ == '__main__':


