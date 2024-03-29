from unittest import TestCase
from unittest.mock import patch, Mock
from src.mutation import mutator
from data.protein_sequences import read_seqs


class TestMutator(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAV' \
                    'VTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'
        self.prot_id = 'SYUA_HUMAN'
        self.prot_id_seq = {self.prot_id: self.asyn}

    def test__make_mutant_name_simple(self):
        actual = mutator._make_mutant_name(prot_id_seq={'simple': 'AAP'}, point_mutation_to_make={3: 'D'})
        expected = 'simple(P3D)'
        self.assertEqual(expected, actual)

    def test__are_valid_positions_to_mutate_simple(self):
        # expected format for `prot_id_point_mutations_to_make` is:
        # {'SYUA_HUMAN': [{30: 'P'}], 'P10636-7': [{301: 'S'}, {301: 'L'}]}
        prot_id = 'test_peptide_AAP'
        self.assertTrue(mutator._are_valid_positions_to_mutate(prot_id_seqs={prot_id: 'AAP'},
                                                               prot_id_point_mutations_to_make={prot_id: [{3: 'D'}]}))
        self.assertTrue(mutator._are_valid_positions_to_mutate(prot_id_seqs={prot_id: 'AAP'},
                                                               prot_id_point_mutations_to_make={prot_id: [{3: 'D'},
                                                                                                         {2: 'E'}]}))
        self.assertTrue(mutator._are_valid_positions_to_mutate(prot_id_seqs={prot_id: 'AAP'},
                                                               prot_id_point_mutations_to_make={prot_id: {3: 'D'}}))

        self.assertFalse(mutator._are_valid_positions_to_mutate(prot_id_seqs={prot_id: 'AAP'},
                                                                prot_id_point_mutations_to_make={prot_id: {4: 'D'}}))

        self.assertFalse(mutator._are_valid_positions_to_mutate(prot_id_seqs={prot_id: 'AAP'},
                                                                prot_id_point_mutations_to_make={prot_id: {0: 'D'}}))

    def test__make_point_mutant_simple1(self):
        actual = mutator._make_point_mutants(prot_id_seq={'simple': 'AAP'}, point_mutants_to_make={3: ['D', 'T']})
        expected = {'simple': 'AAP', 'simple(P3D)': 'AAD', 'simple(P3T)': 'AAT'}
        self.assertEqual(expected, actual)

    def test__make_point_mutant_simple2(self):
        actual = mutator._make_point_mutants(prot_id_seq={'simple': 'AAP'}, point_mutants_to_make={3: ['P']})
        expected = {'simple': 'AAP'}
        self.assertEqual(expected, actual)

    @patch.object(read_seqs, 'get_sequences_by_uniprot_accession_nums_or_names')
    def test_make_point_mutants(self, mock_get_sequences_by_uniprot_accession_nums_or_names):
        # mock_get_sequences_by_uniprot_accession_nums_or_names.return_value =
        mock_get_sequences_by_uniprot_accession_nums_or_names.side_effect = [{'prot': 'AAP'}, {'prot2': 'KRI'}]
        prot_id_mutants_to_make = {'prot': {1: ['H'], 3: ['D', 'F']}, 'prot2': {2 : ['G']}}
        actual = mutator.make_point_mutants(prot_id_mutants_to_make)
        expected = {'prot': {'prot': 'AAP', 'prot(A1H)': 'HAP', 'prot(P3D)': 'AAD', 'prot(P3F)': 'AAF'},
                    'prot2':  {'prot2': 'KRI', 'prot2(R2G)': 'KGI'}}
        self.assertDictEqual(expected, actual)

    def test_mutate_protein(self):
        actual = mutator.mutate(prot_seq='ACDEFG', pos_aa={4: 'Y', 6: 'Q'})
        expected = 'ACDYFQ'
        self.assertEqual(expected, actual)

    def test_mutate_protein_invalid(self):
        actual = mutator.mutate(prot_seq='ACDEFG', pos_aa={4: 'Y', 6: 'X'})
        expected = 'ACDYFG'
        self.assertEqual(expected, actual)

    def test_mutate_protein_invalid2(self):
        actual = mutator.mutate(prot_seq='ACDEFG', pos_aa={4: 'Y', 7: 'Q'})
        expected = 'ACDYFG'
        self.assertEqual(expected, actual)
