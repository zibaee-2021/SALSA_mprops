import cProfile
import sys


def get_size_of(obj) -> str:
    return f'{sys.getsizeof(obj)} bytes.'


def get_perf_stats_for_function(func):
    cProfile.run(func)
