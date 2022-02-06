"""
Module to apply a sampling strategy for detecting the optimal configuration of side-chain-side-chain
interactions in a stacked amyloid fold.
The theory is that by scoring expected side-chain-side-chain interactions according to relative
levels of complementarity (or otherwise), the sum of all these values for each possible configuration
will correlate with observed data. Such a finding would strongly suggest that the population of
proteins sample every possible arrangement and that those which are observed to form are the most
stable on average for all the residues in the amyloid core.

A further demonstration will be to compare the results of using the whole protein sequence versus
just the region(s) identified by high beta-strand contiguity.

This would provide evidence that the formation of the filament might be a combination of beta-stand
contiguity causing one b-sheet face to start to grow, before the chains collapse into the folds
that further stabilise and provide a template for 'true' seeding, leading to exact replicas of the
origin seed(s).
The way in which every possible conformation of polypeptide in the grid/graph representation would
be calculated from is shown here using a very small 3-residue peptide `ACD`.
It should be possible to see from this representation that `A` is fixed at the central position,
while `C` and `D` sample every possible conformation relative to `A`.
Blank grids indicate where the intended conformation is not possible due co-location of two residues
for that intended conformation.
In this sequential representation, it can be seen that each residue samples every neighbouring position
of its N-terminal neighbour in a clockwise order and for each of 8 possible positions, i.e. north,
north-east, east, south-east, south, south-west, west, north-west:
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - 2 3 - -   - - 2 3 -   - - - 2 3
- - 1 2 3   - - 1 - -   - - 1 - -   - - 1 - -   - - - - -   - - 1 - -   - - 1 - -   - - 1 - -
- - - - -   - - - 2 3   - - 2 3 -   - 2 3 - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -

- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - 2 - -   - - - 2 -
- - 1 2 -   - - 1 - -   - - 1 - -   - - 1 - -   - 2 1 - -   - - - - -   - - 1 3 -   - - 1 - 3
- - - - 3   - - - 2 -   - - 2 - -   - 2 - - -   - - 3 - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - 3   - - - 3 -   - - 3 - -   - - - - -   - - - - -   - - - - -   - - - - -

- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - - -   - - - - -   - - - - -   - - - - -   - 2 - - -   - - - - -   - - - 2 -
- - 1 2 -   - - 1 - -   - - 1 - -   - - 1 - -   - 2 1 - -   - 3 1 - -   - - - - -   - - 1 3 -
- - - 3 -   - - - 2 -   - - 2 - -   - 2 - - -   - 3 - - -   - - - - -   - - - - -   - - - - -
- - - - -   - - - 3 -   - - 3 - -   - 3 - - -   - - - - -   - - - - -   - - - - -   - - - - -
etc..

The sampling calculates scores of complementarity for every pair of neighbouring residues for
each conformation according to the properties of the side-chain pairs.

(The above can be also implemented by a simple graph implementation of the 2d array (think of
grid format above), but I think it also as likely that a fully-connected graph would servce
some purpose for increasing the flexibility and
model to go about this task. I will need to investigate this after first implementing the original
idea.

Each protein's possible `conformations` is a list of `conformation`. The list contains a tuple of tuples. Each inner
tuple represents the coordinates of a residue, and the outer tuple simply contains all of these inner tuples. The
reason for choosing to encapsulate `conformation` as a python tuple of tuples instead of a python list of tuples is
because a tuple is immutable and as a result using less memory than a list, which is mutable. There is no requirement
for changing any of the positions in each conformation so there is no need to use a list of tuples.
E.g. if you have a 2-residue sequence like 'AC', then your `conformations` would look like:
[ ((0, 0), (-1,  0)),
  ((0, 0), (-1,  1)),
  ((0, 0), ( 0,  1)),
  ((0, 0), ( 1,  1)),
  ((0, 0), ( 1,  0)),
  ((0, 0), ( 1, -1)),
  ((0, 0), ( 0, -1)),
  ((0, 0), (-1, -1))]
As in the illustration above, the first residue is anchored at one position (0, 0) and the neighbouring residue can now
occupy each of 8 positions around the first residue. As such a 2-residue protein has 8 possible conformations. And so
it is represented by a list of 8 tuples of tuples contain each pair of positional coordinates.

** NOTE **:
You can generate all possible conformations without needing to know the sequence.
In fact you can generate them in advance for any size of protein and then just fill it in
with the sequence later!
"""
from __future__ import annotations
# `annotations` because dataclass cannot recognise Node type self-reference.
from typing import List, Tuple
from data.protein_sequences import read_seqs
import time
import math
import pandas as pd

# def _print(conformation):

def _get_euclidean_dist(res_pos1, res_pos2):
    return math.sqrt((res_pos1[0] - res_pos2[0]) ** 2 + (res_pos1[1] - res_pos2[1]) ** 2)


def _get_manhattan_dist(res_pos1, res_pos2):
    return abs(res_pos1[0] - res_pos2[0]) + abs(res_pos1[1] - res_pos2[1])


def find_adjacent_residues(conformation: Tuple[Tuple]) -> Tuple[dict, dict]:
    """
    From the given conformation, find the 8 residues that are in the immediate neighbourhood of each residue.
    Find the 16 residues that are in the next neighbourhood out of each residue:

                    o  o  o  o  o
                    o  i  i  i  o
                    o  i  .  i  o
                    o  i  i  i  o
                    o  o  o  o  o
                                        . is central residue
                                        i is immediate neighbours
                                        o is outer neighbours

    (Note: In current implementation, residues are only considered valid neighbours if they are not the next residue
    along in sequence. This might not be a valid consideration and will possibly be removed subject to what is
    reported and observed in amyloid structures in literature.)
    :param conformation:
    :return:
    """
    adjacent, adjacent_outer = {}, {}
    for x, residue in enumerate(conformation):
        res_num = x + 1
        for xx, residue_ in enumerate(conformation):
            res_num_ = xx + 1
            if abs(res_num - res_num_) > 1:
                if _get_euclidean_dist(residue, residue_) <= math.sqrt(2):
                    if adjacent.get(res_num, None) is not None:
                        adjacent[res_num] += (res_num_, )
                    else:
                        adjacent[res_num] = (res_num_, )
                elif _get_euclidean_dist(residue, residue_) <= math.sqrt(8):
                    if adjacent_outer.get(res_num) is not None:
                        adjacent_outer[res_num] += (res_num_, )
                    else:
                        adjacent_outer[res_num] = (res_num_, )
                else:
                    continue
    return adjacent, adjacent_outer


def _extract_unique_pairs(adjacent, adjacent_outer):
    pairs = tuple()
    for res_num, neighbours in adjacent.items():
        for neighbour in neighbours:
            pair = (res_num, neighbour)
            pair = tuple(sorted(pair))
            if pair not in pairs:
                pairs += (pair, )
    pairs_outer = tuple()
    for res_num, neighbours in adjacent_outer.items():
        for neighbour in neighbours:
            pair = (res_num, neighbour)
            pair = tuple(sorted(pair))
            if pair not in pairs_outer:
                pairs_outer += (pair, )
    return pairs, pairs_outer






def _make_residue_pairs_table()


# def compute_complementarity_score(conformation: Conformation) -> float:
#     """
#     Compute the mean of possible interactions from the given conformation.
#     :param conformation: Current conformation of the protein.
#     :return: Mean of possible interactions
#     """
#     return 4.4


def _get_conformations(anchor_pos: Tuple, occupied) -> list:
    """
    Generate all possible conformations for a residue relative to its N-terminal neighbour, termed the `anchor`.
    This name is given because as far as the neighbour residue is concerned, the anchor remains fixed at its position
    (for the purposes of this function).
    :param anchor_pos: Position of the `anchor` residue that is the N-terminal neighbour to the residue for which to
    all possible positions are being found.
    :param occupied: The positions occupied by the rest of this conformation, N-terminal to the current residue.
    :return: All possible conformations for the given `anchor` residue and its C-terminal neighbour.
    """
    conformations = list()
    if anchor_pos == (0, 0):
        occupied = (occupied, )
    for rel_pos in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        neighbour_pos = (anchor_pos[0] + rel_pos[0], anchor_pos[1] + rel_pos[1])
        if neighbour_pos not in occupied:
            conformations.append((anchor_pos, neighbour_pos))
    return conformations


def get_conformations(conformations: List[Tuple[Tuple[int, int]]]) -> list:
    """
    Generate all possible conformations for a residue relative to its N-terminal neighbour, termed the `anchor`.
    This name is given because as far as the neighbour residue is concerned, the anchor remains fixed at its position
    (for the purposes of this function).
    Assumption here is that if `len(conformations) is one`, this can only the case for the first (N-terminal) residue.
    Therefore `conformation` has only one position, because the N-terminal anchor residue never
    moves. As such, I enclose it inside an empty tuple, otherwise `_get_conformations()` will fail due to its logic
    expecting `conformation` to be a tuple of tuples.
    :param conformations: Conformations
    :return: All possible conformations for the given `anchor` residue and its C-terminal neighbour.
    """
    combined_confs = list()
    for conformation in conformations:
        anchor_pos = conformation[-1]
        new_confs = _get_conformations(anchor_pos, occupied=conformation)
        for i, new_conf in enumerate(new_confs):
            new_conf = conformation[:-1] + new_conf
            combined_confs.append(new_conf)
    return combined_confs


def build_conformations(seq: str, starting_conformations:List[Tuple[Tuple]]=None):
    if starting_conformations is None:
        all_protein_confs = []
    else:
        all_protein_confs = starting_conformations
    for i, aa in enumerate(seq):
        print(f'starting residue {aa} at pos {i + 1}')
        print(f'{round(1000 * (time.time() - st), 1)} ms')
        if i == 0:
            conformations = [((0, 0),)]
            all_protein_confs.extend(get_conformations(conformations))
        else:
            all_protein_confs.extend(get_conformations(all_protein_confs))
    return all_protein_confs


if __name__ == '__main__':
    # print(multiprocessing.cpu_count())
    st = time.time()
    # prot_id = 'P05067(672-711)'
    # prot_id_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_id)
    # seq = prot_id_seq[prot_id]
    seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
    res = build_conformations(seq[:8])

    print(f'Number of possible conformations: {len(res)}')
    # print(f'{round(1000 * (time.time() - st), 1)} ms')

# First 8 residues is about 1 second.
# 9th takes