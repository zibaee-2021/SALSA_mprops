from unittest import TestCase
from src.mean_properties import mprops
from collections import namedtuple


class TestMprops(TestCase):

    def setUp(self) -> None:
        self.asyn = 'MDVFMKGLSK AKEGVVAAAE KTKQGVAEAA GKTKEGVLYV GSKTKEGVVH GVATVAEKTK EQVTNVGGAV' \
                    'VTGVTAVAQK TVEGAGSIAA ATGFVKKDQL GKNEEGAPQE GILEDMPVDP DNEAYEMPSE EGYQDYEPEA'

    def test__normalise_mbp_mh_mnc_mtc_asyn(self):
        actual = mprops._normalise_mbp_mh_mnc_mtc(mbp=0.9067751636299309, mh=-5.713357142857143,
                                                  mnc=0.06428571428571428, mtc=0.2785714285714286)
        # (Note: the test passes when the name of the namedtuple is different hence `expected`, but fails if the name
        # of the variables nmbp, nmh, nmnc, nmtc are different.)
        expected = namedtuple('expected', 'nmbp, nmh, nmnc, nmtc')
        expected = expected(nmbp=35.59172120997698, nmh=44.58482142857143, nmnc=32.14285714285714,
                            nmtc=65.47619047619047)
        self.assertEqual(expected, actual)

    def test_calculate_weights_for_each_4_props(self):
        _4coefs, _4intcpts, _4rsq = mprops.calculate_relative_weights_for_each_4_props()
        expected_4coefs = {'nmbp': -1.7, 'nmh': -2.8, 'nmnc': 2.768, 'nmtc': 2.958}
        expected_4intcpts = {'nmbp': 3.741, 'nmh': 4.452, 'nmnc': 1.878, 'nmtc': 1.177}
        expected_4rsq = {'nmbp': 0.103, 'nmh': 0.317, 'nmnc': 0.222, 'nmtc': 0.304}
        self.assertDictEqual(expected_4coefs, _4coefs)
        self.assertDictEqual(expected_4intcpts, _4intcpts)
        self.assertDictEqual(expected_4rsq, _4rsq)

    def test__make_fragments1(self):
        a68_71Del = 'MDVFMKGLSK AKEGVVAAAE KTKQGVAEAA GKTKEGVLYV GSKTKEGVVH GVATVAEKTK' \
                    'EQVTNVG     TGVTAVAQK TVEGAGSIAA ATGFVKKDQL GKNEEGAPQE GILEDMPVDP' \
                    'DNEAYEMPSE EGYQDYEPEA'
        syn_name = 'a68_71Del'
        actual = mprops._make_fragment(syn_name)
        self.assertEqual(a68_71Del.replace(' ', ''), actual)

    def test__make_fragments2(self):
        syn_name = 'a11_140'
        actual = mprops._make_fragment(syn_name)
        a11_140 =            'AKEGVVAAAE KTKQGVAEAA GKTKEGVLYV GSKTKEGVVH GVATVAEKTK EQVTNVGGAV ' \
                  'VTGVTAVAQK TVEGAGSIAA ATGFVKKDQL GKNEEGAPQE GILEDMPVDP DNEAYEMPSE EGYQDYEPEA'
        self.assertEqual(a11_140.replace(' ', ''), actual)

    def test__make_fragments3(self):
        a1_80 = 'MDVFMKGLSK AKEGVVAAAE KTKQGVAEAA GKTKEGVLYV ' \
                'GSKTKEGVVH GVATVAEKTK EQVTNVGGAV VTGVTAVAQK'
        syn_name = 'a1_80'
        actual = mprops._make_fragment(syn_name)
        self.assertEqual(a1_80.replace(' ', ''), actual)
