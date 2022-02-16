from unittest import TestCase
from src.mean_properties import mprops
from collections import namedtuple


class TestMprops(TestCase):

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
        expected_4coeffs = -0.059673073944582725, 0.033584104436484644, 0.05566325802064931, 0.03600378546226167
        expected_4rsq = 0.14157907636054157, 0.13181468959476017, 0.28654265816715596, 0.06469494011742483
        self.assertEqual(expected_4coeffs, _4coefs)
        self.assertEqual(expected_4rsq, _4rsq)
