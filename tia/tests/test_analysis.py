import unittest
import pandas as pd
import pandas.util.testing as pdtest
import numpy as np
from tia.analysis import *


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.closing_pxs = pd.Series(np.arange(10, 19, dtype=float), pd.date_range('12/5/2014', '12/17/2014', freq='B'))
        self.dvds = pd.Series([1.25, 1.], index=[pd.Period('12/8/2014'), pd.Period('12/16/2014')])

    def test_trade_split(self):
        trd = Trade(1, '12/1/2014', 10., 10., -1.)
        t1, t2 = trd.split(4)
        self.assertEqual(t1.qty, 4)
        self.assertEqual(t1.fees, -.4)
        self.assertEqual(t1.ts, trd.ts)
        self.assertEqual(t2.qty, 6)
        self.assertEqual(t2.fees, -.6)
        self.assertEqual(t2.ts, trd.ts)

    def test_txn_details(self):
        t1 = Trade(1, '12/8/2014', 5., 10., -1.)
        t2 = Trade(2, '12/8/2014', 2., 15., -1.)
        t3 = Trade(3, '12/10/2014', -3., 5., -1.)
        t4 = Trade(4, '12/12/2014', -4., 20., -1.)
        t5 = Trade(5, '12/16/2014', -4., 10., 0)
        t6 = Trade(6, '12/17/2014', 4., 15., 0)
        sec = PortfolioPricer(multiplier=2., closing_pxs=self.closing_pxs, dvds=self.dvds)

        # Test txn frame
        port = SingleAssetPortfolio(sec, [t1, t2, t3, t4, t5, t6])
        txns = port.txn_frame
        index = range(len(port.trades))
        pdtest.assert_series_equal(txns.txn_qty, pd.Series([5., 2., -3., -4., -4, 4], index=index))
        pdtest.assert_series_equal(txns.open_val, pd.Series([-100., -160., -160. * 4./7., 0, 80, 0], index=index))
        pdtest.assert_series_equal(txns.txn_fees, pd.Series([-1., -1., -1., -1., 0, 0], index=index))
        pdtest.assert_series_equal(txns.txn_intent, pd.Series([INTENT_OPEN, INTENT_INCREASE, INTENT_DECREASE,
                                                               INTENT_CLOSE, INTENT_OPEN, INTENT_CLOSE],
                                                               index=index))
        pdtest.assert_series_equal(txns.txn_side, pd.Series([SIDE_BUY, SIDE_BUY, SIDE_SELL, SIDE_SELL,
                                                             SIDE_SELL_SHORT, SIDE_COVER], index=index))
        # CHECK PL
        ltd = port.ltd_txn_pl
        dly = port.dly_txn_pl
        # Load the dataset
        import tia, os
        xl = os.path.join(tia.__path__[0], 'tests', 'test_analysis.xlsx')
        expected = pd.read_excel(xl)
        expected = expected.reset_index()
        # check ltd
        pdtest.assert_series_equal(expected.pos.astype(float), ltd.pos)
        pdtest.assert_series_equal(expected.ltd_pl, ltd.pl)
        pdtest.assert_series_equal(expected.ltd_upl, ltd.upl)
        pdtest.assert_series_equal(expected.ltd_rpl, ltd.rpl)
        pdtest.assert_series_equal(expected.ltd_dvds, ltd.dvds)
        pdtest.assert_series_equal(expected.ltd_fees.astype(float), ltd.fees)
        pdtest.assert_series_equal(expected.ltd_rpl_gross, ltd.rpl_gross)
        # check dly
        pdtest.assert_series_equal(expected.pos.astype(float), dly.pos)
        pdtest.assert_series_equal(expected.dly_pl, dly.pl)
        pdtest.assert_series_equal(expected.dly_upl, dly.upl)
        pdtest.assert_series_equal(expected.dly_rpl, dly.rpl)
        pdtest.assert_series_equal(expected.dly_dvds, dly.dvds)
        pdtest.assert_series_equal(expected.dly_fees.astype(float), dly.fees)
        pdtest.assert_series_equal(expected.dly_rpl_gross, dly.rpl_gross)

        # few sanity checks on dly (non-txn level)
        for col in ['pl', 'rpl', 'upl', 'dvds', 'fees']:
            pdtest.assert_series_equal(dly.set_index('dt')[col].resample('B', how='sum'), port.dly_pl[col])

        # Double check the long / short add up to the total
        l, s = port.perf.long_only.ltd_pl, port.perf.short_only.ltd_pl
        ls = port.perf.ltd_pl
        pdtest.assert_frame_equal(ls, l + s)

        l, s = port.perf.long_only.dly_pl, port.perf.short_only.dly_pl
        ls = port.perf.dly_pl
        pdtest.assert_frame_equal(ls, l + s)


        srets = (port.perf.long_only.dly_roii + port.perf.short_only.dly_roii)
        pdtest.assert_series_equal(srets, port.perf.dly_roii)





