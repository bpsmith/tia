import pandas as pd

from tia.util.decorator import lazy_property
from tia.analysis.model.interface import PlColumns as PL


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
        #    if hasattr(self, attr):
        #        delattr(self, attr)

    @lazy_property
    def ltd_txn(self):
        return (1. + self.dly_txn).cumprod() - 1.

    @property
    def ltd(self):
        rets = self.ltd_txn
        rets.index = self.txns.pl.ltd_txn_frame[PL.DT]
        return rets.groupby(lambda x: x).apply(lambda x: x[-1])

    @property
    def dly(self):
        ltd = self.ltd
        dly = (1. + ltd).pct_change()
        dly.iloc[0] = ltd.iloc[0]
        return dly

    weekly = property(lambda self: (1. + self.dly).resample('W', how='prod') - 1.)
    monthly = property(lambda self: (1. + self.dly).resample('M', how='prod') - 1.)
    quarterly = property(lambda self: (1. + self.dly).resample('Q', how='prod') - 1.)
    annual = property(lambda self: (1. + self.dly).resample('A', how='prod') - 1.)


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


class BasicRets(Rets):
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
        rets = BasicRets(self.cash_start)
        rets.txns = sub_txns
        return rets
