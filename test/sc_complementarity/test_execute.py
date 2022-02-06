from unittest import TestCase
from src.sc_complementarity import execute
from dataclasses import dataclass


class TestComp(TestCase):

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

        actual = execute._get_conformations(anchor_pos, occupied=anchor_pos)
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
        combined_confs = list()
        for conformation in conformations:
            anchor_pos = conformation[-1]
            new_confs = execute.get_conformations(anchor_pos)
            for i, new_conf in enumerate(new_confs):
                new_conf = conformation[:-1] + new_conf
                combined_confs.append(new_conf)
        actual = combined_confs
        self.maxDiff = None
        self.assertCountEqual(expected, actual)

    def test_build_conformations(self):
        actual = execute.build_conformations(seq='ACD')
        expected = []
        self.assertCountEqual(expected, actual)

    def test__get_manhattan_dist(self):
        res_pos1, res_pos2 = (3, 4), (6, 9)
        expected = 8
        actual = execute._get_manhattan_dist(res_pos1, res_pos2)
        self.assertEqual(expected, actual)

    def test__get_euclidean_dist(self):
        res_pos1, res_pos2 = (3, 5), (6, 9)
        expected = 5
        actual = execute._get_euclidean_dist(res_pos1, res_pos2)
        self.assertEqual(expected, actual)

    def test_find_adjacent_residues(self):
        conformation = ((0, 0), (-1, 1), (-1, 2), (-2, 3), (-3, 2), (-4, 1),
                        (-3, 0), (-2, 0), (-1, -1), (-1, -2), (0, -1))
        actual_adj, actual_adj_out = execute.find_adjacent_residues(conformation)
        expected_adj = {1: (9, 11), 2: (8, ), 8: (2, ), 9: (1, 11), 11: (1, 9)}
        expected_adj_outer = {1: (3, 8, 10), 2: (4, 5, 7, 9, 11), 3: (1, 5, 7, 8), 4: (2, 6), 5: (2, 3, 7, 8),
                              6: (4, 8), 7: (2, 3, 5, 9, 10), 8: (1, 3, 5, 6, 10, 11), 9: (2, 7), 10: (1, 7, 8),
                              11: (2, 8)}
        self.assertDictEqual(expected_adj, actual_adj)
        self.assertDictEqual(expected_adj_outer, actual_adj_out)

    def test__extract_unique_pairs(self):
        adj = {1: (9, 11), 2: (8, ), 8: (2, ), 9: (1, 11), 11: (1, 9)}
        adj_outer = {1: (3, 8, 10), 2: (4, 5, 7, 9, 11), 3: (1, 5, 7, 8), 4: (2, 6), 5: (2, 3, 7, 8),
                     6: (4, 8), 7: (2, 3, 5, 9, 10), 8: (1, 3, 5, 6, 10, 11), 9: (2, 7), 10: (1, 7, 8), 11: (2, 8)}
        pairs_adj, pairs_adj_outer = execute._extract_unique_pairs(adj, adj_outer)
        print(pairs_adj)
        print(pairs_adj_outer)

# def test_get_neighbouring_pairs(self):
    #     conformation = ((0, 0), (-1, 0), (-2, 0), (-3, 0), (-2, 1), (-1, 2), (0, 1))
    #     actual = execute.get_neighbouring_pairs(conformation)
    #     expected = []

    # def test__print(self):
    #     conformation = ((0, 0), (-1, 0), (-2, 0), (-3, 0), (-2, 1), (-3, 2), (-2, 1), (-1, 1), (0, 1))
    #     execute._print(conformation)
