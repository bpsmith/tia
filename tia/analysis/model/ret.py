from collections import OrderedDict

import pandas as pd

from tia.util.decorator import lazy_property
from tia.analysis.model.interface import PlColumns as PL
from tia.analysis.perf import drawdown_info, drawdowns, returns_cumulative, sharpe_annualized, sharpe, std_annualized, \
    returns_annualized


def return_on_initial_capital(capital, period_pl):
    if capital <= 0:
        raise ValueError('cost must be a positive number not %s' % capital)
    eod = capital + period_pl.cumsum()
    ltd_rets = (eod / capital) - 1.
    dly_rets = ltd_rets
    dly_rets.iloc[1:] = (1. + ltd_rets).pct_change().iloc[1:]
    return dly_rets


class Rets(object):
    def __init__(self):
        self._txns = None

    @property
    def txns(self):
        return self._txns

    @txns.setter
    def txns(self, val):
        if self._txns is not None and self._txns != val:
            raise Exception('txns can only be assigned once to a return object')
        self._txns = val
        # remove any cached items
        # for attr in ['_ltd_txn', '_dly_txn', '_ltd', '_dly']:
        # if hasattr(self, attr):
        # delattr(self, attr)

    @lazy_property
    def ltd_txn(self):
        return (1. + self.dly_txn).cumprod() - 1.

    @lazy_property
    def ltd(self):
        rets = self.ltd_txn
        rets.index = self.txns.pl.ltd_txn_frame[PL.DT]
        return rets.groupby(lambda x: x).apply(lambda x: x[-1])

    @lazy_property
    def dly(self):
        ltd = self.ltd
        dly = (1. + ltd).pct_change()
        dly.iloc[0] = ltd.iloc[0]
        return dly

    weekly = lazy_property(lambda self: (1. + self.dly).resample('W', how='prod') - 1., 'weekly')
    monthly = lazy_property(lambda self: (1. + self.dly).resample('M', how='prod') - 1., 'monthly')
    quarterly = lazy_property(lambda self: (1. + self.dly).resample('Q', how='prod') - 1., 'quarterly')
    annual = lazy_property(lambda self: (1. + self.dly).resample('A', how='prod') - 1., 'annual')

    def new_stats(self, rets, lbl):
        return RetStats(rets, lbl)

    dly_stats = lazy_property(lambda self: self.new_stats(self.dly, 'dly'), 'dly_stats')
    weekly_stats = property(lambda self: self.new_stats(self.weekly, 'weekly'), 'weekly_stats')
    monthly_stats = property(lambda self: self.new_stats(self.monthly, 'monthly'), 'monthly_stats')
    quarterly_stats = property(lambda self: self.new_stats(self.quarterly, 'quarterly'), 'quarterly_stats')
    annual_stats = property(lambda self: self.new_stats(self.annual, 'annual'), 'annual_stats')


class RoiiRets(Rets):
    def __init__(self):
        Rets.__init__(self)

    @lazy_property
    def dly_txn(self):
        txnpl = self.txns.pl.dly_txn_frame
        rets = pd.Series(0, index=txnpl.index, name='ret')
        for pid, pframe in txnpl[[PL.OPEN_VAL, PL.PID, PL.PL]].groupby(PL.PID):
            if pid != 0:
                cost = abs(pframe[PL.OPEN_VAL].iloc[0])
                ppl = pframe[PL.PL]
                rets[ppl.index] = return_on_initial_capital(cost, ppl)
        return rets

    def subset(self, sub_txns):
        # TODO - think about reusing dly_txn
        rr = RoiiRets()
        rr.txns = sub_txns
        return rr


class SimpleRets(Rets):
    def __init__(self, cash_start):
        """
        :param txns:
        :param cash_start: starting cash balance
        """
        Rets.__init__(self)
        self.cash_start = cash_start

    @lazy_property
    def dly_txn(self):
        txnpl = self.txns.pl.dly_txn
        rets = return_on_initial_capital(self.cash_start, txnpl)
        rets.name = 'ret'
        return rets

    def subset(self, sub_txns):
        # TODO - think about reusing dly_txn
        rets = SimpleRets(self.cash_start)
        rets.txns = sub_txns
        return rets


class RetStats(object):
    def __init__(self, rets, label=None):
        self.rets = rets
        self.label = label

    @lazy_property
    def drawdown_info(self):
        """Drawdown information"""
        return drawdown_info(self.rets)

    # series data
    drawdowns = lazy_property(lambda self: drawdowns(self.rets), 'drawdowns')

    # scalar data
    cnt = property(lambda self: len(self.rets.index))
    total = lazy_property(lambda self: returns_cumulative(self.rets), name='total')
    total_ann = lazy_property(lambda self: returns_annualized(self.rets), name='total_ann')
    ret_avg = lazy_property(lambda self: self.rets.mean(), 'ret_avg')
    ret_avg_ann = lazy_property(lambda self: returns_annualized(self.rets, geometric=0), 'ret_avg_ann')
    std = lazy_property(lambda self: self.rets.std(), 'std')
    std_ann = lazy_property(lambda self: std_annualized(self.rets), 'std_ann')
    sharpe = lazy_property(lambda self: sharpe(self.rets), 'sharpe')
    sharpe_ann = lazy_property(lambda self: sharpe_annualized(self.rets), 'sharpe_ann')
    maxdd = lazy_property(lambda self: self.drawdown_info['maxdd'].min(), 'maxdd')
    maxdd_dt = lazy_property(lambda self: None if self.drawdown_info.empty else self.drawdown_info['maxdd dt'].ix[
        self.drawdown_info['maxdd'].idxmin()], 'maxdd_dt')
    dd_avg = lazy_property(lambda self: self.drawdown_info['maxdd'].mean(), 'dd_avg')

    @lazy_property
    def series(self):
        d = OrderedDict()
        d['total'] = self.total
        d['total ann'] = self.total_ann
        d['ret avg'] = self.ret_avg
        d['ret avg ann'] = self.ret_avg_ann
        d['std'] = self.std
        d['std ann'] = self.std_ann
        d['sharpe ann'] = self.sharpe_ann
        d['maxdd'] = self.maxdd
        d['maxdd dt'] = self.maxdd_dt
        d['dd avg'] = self.dd_avg
        d['cnt'] = self.rets.notnull().astype(int).sum()
        return pd.Series(d, name=self.label or self.rets.index.freq)

    def _repr_html_(self):
        from tia.util.fmt import new_dynamic_formatter

        fmt = new_dynamic_formatter(method='row', precision=2, pcts=1, trunc_dot_zeros=1, parens=1)
        return fmt(self.series.to_frame())._repr_html_()





