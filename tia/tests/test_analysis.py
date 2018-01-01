import unittest
import pandas as pd
import pandas.util.testing as pdtest
import numpy as np
from tia.analysis.model import *


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.closing_pxs = pd.Series(np.arange(10, 19, dtype=float), pd.date_range('12/5/2014', '12/17/2014', freq='B'))
        self.dvds = pd.Series([1.25, 1.], index=[pd.to_datetime('12/8/2014'), pd.to_datetime('12/16/2014')])

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
        txns = port.txns.frame
        index = list(range(len(port.trades)))
        pdtest.assert_series_equal(txns.txn_qty, pd.Series([5., 2., -3., -4., -4, 4], index=index))
        pdtest.assert_series_equal(txns.open_val, pd.Series([-100., -160., -160. * 4./7., 0, 80, 0], index=index))
        pdtest.assert_series_equal(txns.txn_fees, pd.Series([-1., -1., -1., -1., 0, 0], index=index))
        pdtest.assert_series_equal(txns.txn_intent, pd.Series([Intent.Open, Intent.Increase, Intent.Decrease,
                                                               Intent.Close, Intent.Open, Intent.Close],
                                                               index=index))
        pdtest.assert_series_equal(txns.txn_action, pd.Series([Action.Buy, Action.Buy, Action.Sell, Action.Sell,
                                                             Action.SellShort, Action.Cover], index=index))
        # CHECK PL
        pl = port.pl
        # Load the dataset
        import tia, os
        xl = os.path.join(tia.__path__[0], 'tests', 'test_analysis.xlsx')
        expected = pd.read_excel(xl)
        expected = expected.reset_index()
        # check ltd txn level
        ltd = pl.ltd_txn_frame
        pdtest.assert_series_equal(expected.pos.astype(float), ltd.pos)
        pdtest.assert_series_equal(expected.ltd_pl, ltd.pl)
        pdtest.assert_series_equal(expected.ltd_upl, ltd.upl)
        pdtest.assert_series_equal(expected.ltd_rpl, ltd.rpl)
        pdtest.assert_series_equal(expected.ltd_dvds, ltd.dvds)
        pdtest.assert_series_equal(expected.ltd_fees.astype(float), ltd.fees)
        pdtest.assert_series_equal(expected.ltd_rpl_gross, ltd.rpl_gross)
        # check txn level
        txnlvl = pl.txn_frame
        pdtest.assert_series_equal(expected.pos.astype(float), txnlvl.pos)
        pdtest.assert_series_equal(expected.dly_pl, txnlvl.pl)
        pdtest.assert_series_equal(expected.dly_upl, txnlvl.upl)
        pdtest.assert_series_equal(expected.dly_rpl, txnlvl.rpl)
        pdtest.assert_series_equal(expected.dly_dvds, txnlvl.dvds)
        pdtest.assert_series_equal(expected.dly_fees.astype(float), txnlvl.fees)
        pdtest.assert_series_equal(expected.dly_rpl_gross, txnlvl.rpl_gross)

        # few sanity checks on dly (non-txn level)
        for col in ['pl', 'rpl', 'upl', 'dvds', 'fees']:
            pdtest.assert_series_equal(pl.txn_frame.set_index('date')[col].resample('B', how='sum', kind='period'),
                                       pl.dly_frame[col].to_period('B'))

        # Double check the long / short add up to the total
        l, s = port.long.pl.dly_frame, port.short.pl.dly_frame
        ls = port.pl.dly_frame
        pdtest.assert_frame_equal(ls, l + s)


    def test_blotter(self):
        blotter = TradeBlotter()
        blotter.ts = pd.datetime.now()  # all trades have same timestamp for testing
        blotter.open(qty=2, px=10)
        self.assertEqual(2., blotter._live_qty)
        blotter.close(px=10)
        self.assertEqual(0, blotter._live_qty)
        self.assertEqual(2, len(blotter.trades))
        # should be able to call without issue
        blotter.try_close(px=10)
        self.assertEqual(2, len(blotter.trades))

        blotter.open(qty=2, px=10)
        blotter.increase(qty=2, px=10)
        self.assertEqual(4, blotter._live_qty)
        blotter.decrease(qty=-2, px=10)
        self.assertEqual(2, blotter._live_qty)

    def test_blotter_exceptions(self):
        blotter = TradeBlotter()
        blotter.ts = pd.datetime.now()  # all trades have same timestamp for testing
        self.assertRaises(Exception, lambda: blotter.close(2, 10))
        blotter.open(2, 10.)
        self.assertRaises(Exception, lambda: blotter.open(2, 10))
        self.assertRaises(Exception, lambda: blotter.increase(-2, 10))
        self.assertRaises(Exception, lambda: blotter.decrease(2, 10))


    def test_port(self):
        t1 = Trade(1, pd.to_datetime('3/23/2015 10:00'), 1, 10.)
        t2 = Trade(2, pd.to_datetime('3/23/2015 11:00'), -1, 11.)
        t3 = Trade(3, pd.to_datetime('3/23/2015 12:00'), -1, 12.)
        t4 = Trade(4, pd.to_datetime('3/23/2015 13:00'), 1, 13.)
        pp = PortfolioPricer(1., closing_pxs=pd.Series(10., index=[t1.ts]).asfreq('B', normalize=1))
        port = SingleAssetPortfolio(pp, [t1, t2, t3, t4])
        # make sure long/short is correct
        pdtest.assert_frame_equal(port.positions.frame.ix[1:1], port.long.positions.frame)
        pdtest.assert_frame_equal(port.positions.frame.ix[2:2], port.short.positions.frame)
        # some sanity checks
        pdtest.assert_series_equal(port.pl.dly, port.long.pl.dly + port.short.pl.dly)
        pdtest.assert_series_equal(port.pl.ltd_dly, port.long.pl.ltd_dly + port.short.pl.ltd_dly)
        pdtest.assert_series_equal(port.pl.monthly, port.long.pl.monthly + port.short.pl.monthly)
        pdtest.assert_series_equal(port.pl.ltd_monthly, port.long.pl.ltd_monthly + port.short.pl.ltd_monthly)
