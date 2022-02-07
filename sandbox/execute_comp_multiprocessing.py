"""
Try ThreadPoolExecutor and ProcessPoolExecutor for speed ups.
"""
from typing import List, Tuple
import time
from random import random
from time import sleep
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

def _generate_new_confs(conformation) -> List[Tuple[Tuple[int, int]]]:
    anchor_pos = conformation[-1]
    occupied = conformation
    occupied = (occupied, ) if occupied == (0, 0) else occupied
    new_confs = []

    for rel_pos in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        neighbour_pos = (anchor_pos[0] + rel_pos[0], anchor_pos[1] + rel_pos[1])
        if neighbour_pos not in occupied:
            new_conf = conformation + (neighbour_pos,)
            new_confs.append(new_conf)
    return new_confs


def generate_confs(conformations: List[Tuple[Tuple[int, int]]]) -> List:
    with ProcessPoolExecutor(max_workers=4) as ppe:
        new_confs = []
        for result in ppe.map(_generate_new_confs, conformations):
            new_confs.extend(result)
    return new_confs


def generate_all_possible_conformations(sequence: str, starting_conformations=None):
    all_conformations = [((0, 0),)] if starting_conformations is None else starting_conformations
    for i, aa in enumerate(sequence):
        print(f'starting residue {aa} at pos {i+1} of protein {sequence}')
        if i > 0:
            all_conformations = generate_confs(all_conformations)
    return all_conformations


if __name__ == '__main__':
    st = time.time()
    all_confs = generate_all_possible_conformations('ABCDEFGHIJ')
    print(f'{round((time.time() - st), 1)} s')
    print(f'Number of possible conformations: {len(all_confs)}')

# THIS IS 44 TIMES SLOWER THAN WITHOUT ProcessPoolExecutor
# starting residue H at pos 8 of protein ABCDEFGH
# 44.2 s
# Number of possible conformations: 550504

# starting residue I at pos 9 of protein ABCDEFGHI
# 262.1 s
# Number of possible conformations: 3349864

