def _norm(num_to_normalise: float, translate_by, scale_by) -> list:
    return (num_to_normalise + translate_by) / scale_by


def norm(numbers_to_normalise: list) -> list:
    """
    Normalise the given list of numbers to between 0.0 and 1.0. Hence this assumes that all the numbers are
    included such that it is possible to know the minimum and maximum numbers.
    list of numbers in this function.
    :param numbers_to_normalise: A list of numbers.
    :return: The normalised values of the given list of numbers.
    """
    translate_by = 0 - min(numbers_to_normalise)
    scale_by = max(numbers_to_normalise) - min(numbers_to_normalise)
    return [_norm(num, translate_by, scale_by) for num in numbers_to_normalise]


def norm_dict(mapped_numbers_to_normalise: dict) -> dict:
    """
    Normalise the given collection of numeric values to between 0.0 and 1.0. Each number is mapped to a unique key,
    hence this function strictly expects a dict and will return a dict mapping the same keys to the same numbers,
    normalised.
    :param mapped_numbers_to_normalise: Numbers to normalise, mapped to unique keys.
    :return: The normalised values of the given numbers, mapped to same unique keys.
    """
    translate_by = 0 - min(mapped_numbers_to_normalise.values())
    scale_by = max(mapped_numbers_to_normalise.values()) - min(mapped_numbers_to_normalise.values())
    return {k: _norm(num, translate_by, scale_by) for k, num in mapped_numbers_to_normalise.items()}


def norm_with_precalculated_min_max(num_to_normalise, mini: float, maxi: float):
    """
    Normalise the given number(s) to between 0.0 and 1.0, according to the given minimum and maximum values.
    :param num_to_normalise: Number(s) to normalise according to given minimum and maximum values.
     :param mini: Minimum number by which you normalise.
     :param maxi: Maximum number by which you will normalise.
    :return: The normalised value of the given number(s), according to the given minimum and maximum numbers.
    """
    translate_by, scale_by = (0 - mini), (maxi - mini)
    if not isinstance(num_to_normalise, list):
        num_to_normalise = [num_to_normalise]
    normalised_num = [_norm(num, translate_by, scale_by) for num in num_to_normalise]
    return normalised_num[0] if len(normalised_num) == 1 else normalised_num


if __name__ == '__main__':
    bla = norm_dict(mapped_numbers_to_normalise={'A': 0.5, 'C': 1.1, 'D': 2.0})
    bla