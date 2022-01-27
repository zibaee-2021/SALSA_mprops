from unittest import TestCase
from unittest.mock import patch, Mock
from src.mutation import mutate
from data.protein_sequences import read_seqs


class TestMutate(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV' \
                    'VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
        self.prot_id = 'SYUA_HUMAN'
        self.prot_id_seq = {self.prot_id: self.asyn}

    def test__make_mutant_name_simple(self):
        actual = mutate._make_mutant_name(prot_id_seq={'simple': 'AAP'}, point_mutation_to_make={3: 'D'})
        expected = 'simple(P3D)'
        self.assertEqual(expected, actual)

    def test__are_valid_positions_to_mutate_simple(self):
        # expected format for `prot_id_point_mutations_to_make` is:
        # {'SYUA_HUMAN': [{30: 'P'}], 'P10636-7': [{301: 'S'}, {301: 'L'}]}
        self.assertTrue(mutate._are_valid_positions_to_mutate(prot_id_seqs={'simple': 'AAP'},
                                                              prot_id_point_mutations_to_make={'simple': [{3: 'D'}]}))
        self.assertTrue(mutate._are_valid_positions_to_mutate(prot_id_seqs={'simple': 'AAP'},
                                                              prot_id_point_mutations_to_make={'simple': [{3: 'D'},
                                                                                                          {2: 'E'}]}))
        self.assertTrue(mutate._are_valid_positions_to_mutate(prot_id_seqs={'simple': 'AAP'},
                                                              prot_id_point_mutations_to_make={'simple': {3: 'D'}}))

        self.assertFalse(mutate._are_valid_positions_to_mutate(prot_id_seqs={'simple': 'AAP'},
                                                               prot_id_point_mutations_to_make={'simple': {4: 'D'}}))

        self.assertFalse(mutate._are_valid_positions_to_mutate(prot_id_seqs={'simple': 'AAP'},
                                                               prot_id_point_mutations_to_make={'simple': {0: 'D'}}))

    def test__make_point_mutant_simple(self):
        actual = mutate._make_point_mutants(prot_id_seq={'simple': 'AAP'}, point_mutants_to_make={3: ['D', 'T']})
        expected = {'simple': 'AAP', 'simple(P3D)': 'AAD', 'simple(P3T)': 'AAT'}
        self.assertEqual(expected, actual)

    @patch.object(read_seqs, 'get_sequences_by_uniprot_accession_nums_or_names')
    def test_make_point_mutants(self, mock_get_sequences_by_uniprot_accession_nums_or_names):
        # mock_get_sequences_by_uniprot_accession_nums_or_names.return_value =
        mock_get_sequences_by_uniprot_accession_nums_or_names.side_effect = [{'prot': 'AAP'}, {'prot2': 'KRI'}]
        prot_id_mutants_to_make = {'prot': {1: ['H'], 3: ['D', 'F']}, 'prot2': {2 : ['G']}}
        actual = mutate.make_point_mutants(prot_id_mutants_to_make)
        expected = {'prot': {'prot': 'AAP', 'prot(A1H)': 'HAP', 'prot(P3D)': 'AAD', 'prot(P3F)': 'AAF'},
                    'prot2':  {'prot2': 'KRI', 'prot2(R2G)': 'KGI'}}
        self.assertDictEqual(expected, actual)
