from unittest import TestCase
from data.protein_sequences import read_seqs


class TestReadSequences(TestCase):

    def test_get_sequences_by_uniprot_accession_nums_or_names(self):
        prot_ids = ['E9P8Q3', 'SYUA_HUMAN', 'P05067(672-713)', 'P05067']
        actual = read_seqs.get_sequences_by_uniprot_accession_nums_or_names(prot_ids=prot_ids)
        expected = {'P05067(672-713)': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
                    'E9P8Q3': 'DSNQGNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQGGYQQYNPDAGYQQQYNPQGGYQQYN'
                              'PQGGYQQQFNPQGGRGNYKNFNYNNNLQGYQAGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTK'
                              'PAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEE'
                              'VDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTI'
                              'EVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNK'
                              'MDDPTVNWSKERYDQCVSNVSNFL',
                    'SYUA_HUMAN': 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAV'
                                  'AQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA',
                    'P05067': 'MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQ' \
                              'ITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKS' \
                              'TNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEAD' \
                              'DDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGC' \
                              'GGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRER' \
                              'MSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPR' \
                              'PRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQ' \
                              'KEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADR' \
                              'GLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYT' \
                              'SIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN'}

        self.assertCountEqual(expected, actual)

    def test__has_expected_fragment_id_format(self):
        self.assertTrue(read_seqs._has_expected_fragment_id_format('alphanumeric(1-2)'))
        self.assertTrue(read_seqs._has_expected_fragment_id_format('alphanumeric(10000-200000)'))
        self.assertTrue(read_seqs._has_expected_fragment_id_format('P05067(672-713)'))

        self.assertFalse(read_seqs._has_expected_fragment_id_format('(1-2)alphanumeric'))
        self.assertFalse(read_seqs._has_expected_fragment_id_format('P05067'))
        self.assertFalse(read_seqs._has_expected_fragment_id_format('SYUA_HUMAN'))
        self.assertFalse(read_seqs._has_expected_fragment_id_format('(672-713)'))
        self.assertFalse(read_seqs._has_expected_fragment_id_format('672-713'))

    def test__separate_id_and_fragment(self):
        prot_frag_id = 'P05067(672-713)'
        expected = 'P05067', '672-713'
        actual = read_seqs._separate_id_and_fragment(prot_frag_id)
        self.assertTupleEqual(expected, actual)