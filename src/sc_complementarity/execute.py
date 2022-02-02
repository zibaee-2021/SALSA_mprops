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
    current_pos: Tuple[int, int] = None
    next_node: Node = None
    possible_pos: list = None

    def __init__(self, aa: str, pos: Tuple[int, int]):
        self.aa = aa
        self.current_pos = pos


def find_all_possible_positions_relative_to(anchor: Node, unavailable_pos: List[Tuple]) -> Tuple[Node, List]:
    """
    Get all possible positions (out of a maximum of 8) for the next node of the given "anchor node". Hence from the
    current position of the anchor node, the anchor node's C-terminal neighbour can find all available positions
    according to current position of the anchor as well as any neighbouring residues further N-terminal from the
    anchor. Positions are unavailable when currently occupied by another residue. This information is passed through
    a chain of calls to this function.
    :param anchor: Residue node immediately N-terminal to node we want to find all possible positions for.
    :param unavailable_pos: Any unavailable positions based on the conformation of all preceding (i.e. N-terminal)
    residue nodes. Expected to at least include the current position of the anchor node.
    :return: Given anchor node updated to include all possible positions in its C-terminal neighbour node.
    """
    all_possible_pos = list(tuple)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            possible_pos = (anchor.current_pos[0] + i, anchor.current_pos[1] + j)
            if possible_pos in unavailable_pos:
                continue
            else:
                all_possible_pos.append(possible_pos)
                unavailable_pos.append(possible_pos)
    assert(len(all_possible_pos) <= 8)
    anchor.next_node.possible_pos = all_possible_pos
    return anchor, unavailable_pos


def find_all_possible_positions_for_protein(seq_nodes: list) -> list:
    """
    Produce list of all possible positional coordinates for each residue in the given protein.
    :param seq_nodes: Full protein sequence with information of current positions.
    :return: Full protein sequence with new information of all possible positions for all residues, given the
    current position of other residues.
    """
    unavailable_pos = list()
    for anchor in seq_nodes:
        unavailable_pos.append(anchor.current_pos)
        new_anchor, new_unavailable_pos = find_all_possible_positions_relative_to(anchor, unavailable_pos)
        unavailable_pos.append(new_unavailable_pos)
    return seq_nodes


def update_conformation(seq_nodes):
    second_node = seq_nodes[1]
    # unavailable_pos = list()
    unavailable_pos = seq_nodes[0].current_pos
    second_node.current_pos = second_node.possible_pos[0]
    unavailable_pos.append(second_node.current_pos)
    second_node, unavailable_pos = find_all_possible_positions_relative_to(second_node, unavailable_pos)




    def init_conformation(seq: str) -> List:
    seq_nodes = list()
    node = Node
    for i, aa in enumerate(seq):
        node.aa, node.current_pos = aa, (i * -1, 0)
        if i == 0:
            # This is the first residue at the N-terminus. It is anchored to one position (0,0).
            node.possible_pos = [(0,0)]
    seq_nodes.append(node)
    for i, node in enumerate(seq_nodes):
        if i < len(seq_nodes) - 1:
            node.next_node = seq_nodes[i + 1]
    return seq_nodes

# def get_all_8_pos_for_sequence(prot_id: str):
#     prot_id_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_id)
#     seq = prot_id_seq[prot_id]
#     seq_nodes = _init_conformation(seq)
#     for residue_node in seq_nodes:
#         get_all_8_pos_relative_to()


if __name__ == '__main__':
    # prot_id = 'P05067(672-711)'
    # prot_id_seq = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_id)
    # res = init_conformation(prot_id_seq[prot_id])
    res = init_conformation('ABC')
    print(res)

    # from pympler import asizeof
    # namedtuple_size = asizeof.asizeof(Nterm_node_)
    # dict_size = asizeof.asizeof(Nterm_node_._asdict(namedtuple))
    # gain = 100 - namedtuple_size / dict_size * 100

    # print(f"namedtuple: {namedtuple_size} bytes ({gain:.2f}% smaller)")
    # print(f"dict:       {dict_size} bytes")