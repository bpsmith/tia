import itertools

import pandas as pd
import numpy as np


__all__ = ['Trade', 'TradeBlotter']


class Trade(object):
    """Simple Trade Model"""

    def __init__(self, tid, ts, qty, px, fees=0., **kwargs):
        self.tid = tid
        self.ts = pd.to_datetime(ts)
        self.qty = qty
        self.px = px
        self.fees = fees
        self.kwargs = kwargs

    def split(self, amt):
        """ return 2 trades, 1 with specific amt and the other with self.quantity - amt """
        ratio = abs(amt / self.qty)
        t1 = Trade(self.tid, self.ts, amt, self.px, fees=ratio * self.fees, **self.kwargs)
        t2 = Trade(self.tid, self.ts, self.qty - amt, self.px, fees=(1. - ratio) * self.fees,
                   **self.kwargs)
        return [t1, t2]

    def __repr__(self):
        return '<%s(%s, qty=%s, px=%s, ts=%s)>' % (self.__class__.__name__, self.tid, self.qty, self.px, self.ts)


class TradeBlotter(object):
    """TradeBlotter class provides a way to ensure that a trade never causes a position to cross zero (and later need
    split for long/short accouting). The methods provide the intent so the blotter can verify the expected state."""

    def __init__(self, tidgen=None):
        self.ts = None
        self._live_qty = 0
        self.trades = []
        if tidgen is None or isinstance(tidgen, int):
            tidgen = itertools.count(tidgen or 1, 1)
        self.tidgen = tidgen

    def is_open(self):
        return self._live_qty != 0

    def next_tid(self):
        return next(self.tidgen)

    def _order(self, qty, px, fees=0, **kwargs):
        if not self.ts:
            raise Exception('no timestamp has been set in the blotter')
        trd = Trade(self.next_tid(), self.ts, qty, px, fees=fees, **kwargs)
        self.trades.append(trd)
        self._live_qty += qty

    def open(self, qty, px, fees=0, **kwargs):
        if self.is_open():
            raise Exception('open position failed: position already open')
        if qty == 0:
            raise Exception('open position failed: qty is 0')
        self._order(qty, px, fees, **kwargs)

    def close(self, px, fees=0, **kwargs):
        if not self.is_open():
            raise Exception('close position failed: no position currently open')
        qty = -self.trades[-1].qty
        self._order(qty, px, fees, **kwargs)

    def try_close(self, px, fees=0, **kwargs):
        if not self.is_open():
            return
        else:
            return self.close(px, fees=fees, **kwargs)

    def decrease(self, qty, px, fees=0, **kwargs):
        if not self.is_open():
            raise Exception('decrease position failed: no position currently open')
        if np.sign(self._live_qty) != -np.sign(qty):
            msg = 'decrease position failed: trade quantity {0} is same sign as live quantity {1}'
            raise Exception(msg.format(qty, self._live_qty))
        self._order(qty, px, fees, **kwargs)

    def increase(self, qty, px, fees=0, **kwargs):
        if not self.is_open():
            raise Exception('increase position failed: no position currently open')
        if np.sign(self._live_qty) != np.sign(qty):
            msg = 'increase position failed: trade quantity {0} is different sign as live quantity {1}'
            raise Exception(msg.format(qty, self._live_qty))
        self._order(qty, px, fees, **kwargs)

