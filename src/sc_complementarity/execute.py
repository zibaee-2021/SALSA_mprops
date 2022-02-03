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

** NOTE **:
You can generate all possible conformations without needing to know the sequence.
In fact you can generate them in advance for any size of protein and then just fill it in
with the sequence later!
"""
from __future__ import annotations
# `annotations` because dataclass cannot recognise Node type self-reference.
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from data.protein_sequences import read_seqs


@dataclass
class Node:
    aa: str = None
    current_position: Tuple[int, int] = None
    next_node: Node = None

    def __eq__(self, other):
        return self.aa == other.aa and \
               self.next_node == other.next_node and \
               self.current_position == other.current_position


@dataclass
class Conformation:
    n_term: Node
    complementarity_score: float = None

    def __eq__(self, other):
        return self.n_term == other.n_term and \
               self.complementarity_score == other.complementarity_score


def compute_complementarity_score(conformation: Conformation) -> float:
    """
    Compute the mean of possible interactions from the given conformation.
    :param conformation: Current conformation of the protein.
    :return: Mean of possible interactions
    """
    return 4.4


def get_conformations(anchor_pos: Tuple, occupied: Tuple) -> list:
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
    for rel_pos in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        neighbour_pos = (anchor_pos[0] + rel_pos[0], anchor_pos[1] + rel_pos[1])
        if neighbour_pos not in occupied:
            conformations.append((anchor_pos, neighbour_pos))
    return conformations


# if __name__ == '__main__':
#     print()


    # prot_id = 'P05067(672-711)'
    # prot_id_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_id)
    # res = init_conformation_(prot_id_seq[prot_id])
    # seq_nodes_ = init_conformation_('ACD')
    # unavailable_pos_ = [(0, 0)]
    # seq_nodes_ = find_all_possible_positions_for_current_conformation(seq_nodes_, unavailable_pos_)
    # scores_seq_nodes = dict()
    # scores_seq_nodes[seq_nodes_] = compute_complementarity_score(seq_nodes_)
    # update_to_next_conformation(seq_nodes_)
