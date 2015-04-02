from collections import OrderedDict

import pandas as pd

from tia.util.decorator import lazy_property
from tia.analysis.model.interface import PlColumns as PL
from tia.analysis.perf import drawdown_info, drawdowns, returns_cumulative, sharpe_annualized, sharpe, std_annualized, \
    returns_annualized, sortino_ratio
from tia.util.mplot import AxesFormat
from tia.util.fmt import PercentFormatter


__all__ = ['RoiiRetCalculator', 'SimpleRetCalculator', 'RetStats', 'Rets']


def return_on_initial_capital(capital, period_pl):
    """Return the daily return series based on the capital"""
    if capital <= 0:
        raise ValueError('cost must be a positive number not %s' % capital)
    eod = capital + period_pl.cumsum()
    ltd_rets = (eod / capital) - 1.
    dly_rets = ltd_rets
    dly_rets.iloc[1:] = (1. + ltd_rets).pct_change().iloc[1:]
    return dly_rets


class RetCalculator(object):
    def compute(self, txns):
        raise NotImplementedError()


class RoiiRetCalculator(RetCalculator):
    def compute(self, txns):
        txnpl = txns.pl.dly_txn_frame
        dly_txn_rets = pd.Series(0, index=txnpl.index, name='ret')
        for pid, pframe in txnpl[[PL.OPEN_VAL, PL.PID, PL.PL]].groupby(PL.PID):
            if pid != 0:
                cost = abs(pframe[PL.OPEN_VAL].iloc[0])
                ppl = pframe[PL.PL]
                dly_txn_rets[ppl.index] = return_on_initial_capital(cost, ppl)

        # Convert to daily series
        return TxnRets.from_dly_txn(txns, dly_txn_rets)


class SimpleRetCalculator(RetCalculator):
    def __init__(self, cash_start):
        """
        :param cash_start: starting cash balance
        """
        self.cash_start = cash_start

    def compute(self, txns):
        txnpl = txns.pl.dly_txn
        dly_txn_rets = return_on_initial_capital(self.cash_start, txnpl)
        dly_txn_rets.name = 'ret'
        return TxnRets.from_dly_txn(txns, dly_txn_rets)


class Rets(object):
    def __init__(self, ltd=None, dly=None):
        if dly is None and ltd is None:
            raise ValueError('dly or ltd must be specified')

        if dly is None:
            if ltd.empty:
                dly = ltd.copy()
            else:
                dly = (1. + ltd).pct_change()
                dly.iloc[0] = ltd.iloc[0]

        if ltd is None:
            if dly.empty:
                ltd = dly.empty
            else:
                ltd = (1. + dly).cumprod() - 1.

        self.dly = dly
        self.ltd = ltd

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


class TxnRets(Rets):
    @staticmethod
    def from_dly_txn(txns, dly_txn_rets):
        ltd_txn_rets = (1. + dly_txn_rets).cumprod() - 1.
        tmp = ltd_txn_rets.copy()
        tmp.index = txns.pl.ltd_txn_frame[PL.DT]
        ltd_rets = tmp.groupby(lambda x: x).apply(lambda x: x[-1])
        return TxnRets(ltd_txn_rets, dly_txn_rets, ltd_rets, None)

    def __init__(self, ltd_txn, dly_txn, ltd, dly):
        self.ltd_txn = ltd_txn
        self.dly_txn = dly_txn
        Rets.__init__(self, ltd=ltd, dly=dly)


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
    sortino = lazy_property(lambda self: sortino_ratio(self.rets), 'sortino')
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
        d['sortino'] = self.sortino
        d['maxdd'] = self.maxdd
        d['maxdd dt'] = self.maxdd_dt
        d['dd avg'] = self.dd_avg
        d['cnt'] = self.rets.notnull().astype(int).sum()
        return pd.Series(d, name=self.label or self.rets.index.freq)

    def _repr_html_(self):
        from tia.util.fmt import new_dynamic_formatter

        fmt = new_dynamic_formatter(method='row', precision=2, pcts=1, trunc_dot_zeros=1, parens=1)
        return fmt(self.series.to_frame())._repr_html_()

    def plot_ltd(self, ax=None, style='k', label='ltd', show_dd=1, title=None, legend=1):
        ltd = returns_cumulative(self.rets, expanding=1)
        ax = ltd.plot(ax=ax, style=style, label=label)
        if show_dd:
            dd = self.drawdowns
            dd.plot(style='r', label='drawdowns', alpha=.5)
            ax.fill_between(dd.index, 0, dd.values, facecolor='red', alpha=.25)
            fmt = PercentFormatter

            AxesFormat().Y.percent().apply(ax)
            legend and ax.legend(loc='upper left', prop={'size': 12})

            # show the actualy date and value
            mdt, mdd = self.maxdd_dt, self.maxdd
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.25)
            try:
                dtstr = '{0}'.format(mdt.to_period())
            except:
                # assume daily
                dtstr = '{0}'.format(hasattr(mdt, 'date') and mdt.date() or mdt)
            ax.text(mdt, dd[mdt], "{1} \n {0}".format(fmt(mdd), dtstr).strip(), ha="center", va="top", size=8,
                    bbox=bbox_props)
        title and ax.set_title(title)
        return ax



