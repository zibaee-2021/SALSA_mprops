from unittest import TestCase
from data.protein_sequences import read_seqs


class TestReadSequences(TestCase):

    def test_get_sequences_by_uniprot_accession_nums_or_names(self):
        frags_ids = {'672-713': 'P05067', '672-711': 'P05067'}
        actual = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(fragments_ids=frags_ids)
        expected = {'P05067(672-713)': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
                    'P05067(672-711)': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV'}
        self.assertEqual(expected, actual)

