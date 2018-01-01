import unittest
import pandas as pd
import pandas.util.testing as pdtest
import tia.analysis.ta as ta
import numpy as np

class TATest(unittest.TestCase):
    def test_cross(self):
        s = pd.Series([np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 4, 3, 2, 1])

        res = ta.cross_signal(s, 3)
        exp = pd.Series([np.nan, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1])
        pdtest.assert_series_equal(res, exp)
        # continuous should produce same effect
        res = ta.cross_signal(s, 3, continuous=1)
        pdtest.assert_series_equal(res, exp)

        # Check when crossing a range
        res = ta.cross_signal(s, [2, 3])
        exp = pd.Series([np.nan, -1, -1, 0, 0, 1, 1, 1, 1, 1, 0, -1])
        pdtest.assert_series_equal(res, exp)

    def test_close_to_close_signal(self):
        sig = pd.Series([0, 1, 0, -1, 0, 1, -1, 1], index=pd.date_range('12/1/2014', periods=8, freq='B'))
        pxs = pd.Series(list(range(1, len(sig)+1)), index=sig.index)
        px_getter = lambda ts: ts.day
        trds = ta.Signal(sig).close_to_close(pxs)
        self.assertEqual(9, len(trds))
        self.assertEqual(trds[0].qty, 1)
        self.assertEqual(trds[0].px, sig.index[1].day)
        self.assertEqual(trds[0].ts, sig.index[1])
        self.assertEqual(trds[1].qty, -1)
        self.assertEqual(trds[1].ts, sig.index[2])
        self.assertEqual(trds[2].qty, -1)
        self.assertEqual(trds[2].ts, sig.index[3])
        self.assertEqual(trds[3].qty, 1)
        self.assertEqual(trds[3].ts, sig.index[4])




