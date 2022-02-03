from unittest import TestCase
from src.sc_complementarity import execute
from dataclasses import dataclass


class TestComp(TestCase):

    def setUp(self) -> None:
        node3 = execute.Node(aa='D')
        node2 = execute.Node(aa='C')
        self.node1 = execute.Node(aa='A', current_position=(0, 0))
        self.node1.next_node = node2
        node2.next_node = node3
        self.conf = execute.Conformation(self.node1)

    def test_get_conformations(self):
        seq = 'AC'
        # note a residue is represented by a tuple
        # a conformation as a tuple of tuples
        # `conformations` is a list of conformations
        anchor_pos = (0, 0)
        expected = [((0, 0), (-1, 0)),
                    ((0, 0), (-1, 1)),
                    ((0, 0), (0, 1)),
                    ((0, 0), (1, 1)),
                    ((0, 0), (1, 0)),
                    ((0, 0), (1, -1)),
                    ((0, 0), (0, -1)),
                    ((0, 0), (-1, -1))]

        actual = execute.get_conformations(anchor_pos, occupied=anchor_pos)
        self.assertCountEqual(expected, actual)

    def test_get_conformations_3aa(self):
        seq = 'ACD'
        # note a residue is represented by a tuple
        # a conformation as a tuple of tuples
        conformations = [((0, 0), (-1, 0)),
                         ((0, 0), (-1, 1))]
        expected = [((0, 0), (-1, 0), (-2, 0)),
                    ((0, 0), (-1, 0), (-2, 1)),
                    ((0, 0), (-1, 0), (-1, 1)),
                    ((0, 0), (-1, 0), (0,  1)),
                    ((0, 0), (-1, 0), (0, -1)),
                    ((0, 0), (-1, 0), (-1,-1)),
                    ((0, 0), (-1, 0), (-2,-1)),
                    ((0, 0), (-1, 1), (-2, 1)),
                    ((0, 0), (-1, 1), (-2, 2)),
                    ((0, 0), (-1, 1), (-1, 2)),
                    ((0, 0), (-1, 1), (0,  2)),
                    ((0, 0), (-1, 1), (0,  1)),
                    ((0, 0), (-1, 1), (-1, 0)),
                    ((0, 0), (-1, 1), (-2, 0))]
        new_confs = list()
        combined_confs = list()
        for conformation in conformations:
            anchor_pos = conformation[-1]
            new_confs = execute.get_conformations(anchor_pos, occupied=conformation)
            for i, new_conf in enumerate(new_confs):
                new_conf = conformation[:-1] + new_conf
                combined_confs.append(new_conf)
        actual = combined_confs
        self.maxDiff = None
        self.assertCountEqual(expected, actual)