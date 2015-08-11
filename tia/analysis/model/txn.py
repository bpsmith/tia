import pandas as pd

from tia.util.decorator import lazy_property
from tia.analysis.model.interface import TxnColumns as TC
from tia.analysis.model.pl import TxnProfitAndLoss
from tia.analysis.model.ret import RoiiRetCalculator
from tia.analysis.util import is_decrease, is_increase, crosses_zero


__all__ = ['Intent', 'Action', 'iter_txns', 'Txns']


class Intent(object):
    Open = 1
    Close = 2
    Increase = 3
    Decrease = 4

    Labels = {
        Open: 'Open',
        Close: 'Close',
        Increase: 'Increase',
        Decrease: 'Decrease',
    }


class Action(object):
    Buy = 1
    Sell = 2
    SellShort = 3
    Cover = 4

    Labels = {
        Buy: 'Buy',
        Sell: 'Sell',
        SellShort: 'SellShort',
        Cover: 'Cover',
    }


def iter_txns(trds):
    """ iterator of trades which splits trades to ensure proper long/short accounting"""
    pos = 0
    for trd in trds:
        if pos != 0 and is_decrease(pos, trd.qty) and crosses_zero(pos, trd.qty):
            # Split to make accounting for long/short possible
            closing_trd, opening_trd = trd.split(-pos)
            # setattr(closing_trd, '_txn_id', 1)
            # setattr(opening_trd, '_txn_id', 2)
            yield closing_trd
            pos = opening_trd.qty
            yield opening_trd
        else:
            pos += trd.qty
            yield trd


class Txns(object):
    def __init__(self, trades, pricer, ret_calc=None):
        """
        #TODO - rethink if user should split trades prior to calling this method...
        :param trades: list of Trade objects
        :param pricer:
        """
        # split into l/s positions
        self.trades = tuple(iter_txns(trades))
        self.pricer = pricer
        self._ret_calc = ret_calc or RoiiRetCalculator()

    pids = property(lambda self: self.frame[TC.PID].unique())
    pl = lazy_property(lambda self: TxnProfitAndLoss(self), 'pl')
    performance = lazy_property(lambda self: self.ret_calc.compute(self), 'performance')

    @property
    def ret_calc(self):
        return self._ret_calc

    @ret_calc.setter
    def ret_calc(self, calc):
        self._ret_calc = calc
        if hasattr(self, '_performance'):
            delattr(self, '_performance')

    @lazy_property
    def frame(self):
        """Convert the trades to transaction level details necessary for long/short accouting.

        :param trades:
        :param pricer: provides the interface to get premium for a specified quanity, price, and timestamp.
        :return:
        """
        rows = []
        pricer = self.pricer
        pos = open_val = pid = 0
        for txn in self.trades:
            # These values always get copied
            qty = txn.qty
            premium = pricer.get_premium(qty, txn.px, ts=txn.ts)
            if pos == 0:  # Open position
                side = qty > 0 and Action.Buy or Action.SellShort
                open_val = premium
                pid += 1
                side = side
                intent = Intent.Open
                pos = qty
            elif pos + qty == 0:  # close position
                side = qty > 0 and Action.Cover or Action.Sell
                open_val = 0
                side = side
                intent = Intent.Close
                pos = 0
            elif is_increase(pos, qty):
                side = txn.qty > 0 and Action.Buy or Action.SellShort
                open_val += premium
                pos += qty
                intent = Intent.Increase
                side = side
            else:  # decrease - no worries about split since iterator takes care of it
                side = txn.qty > 0 and Action.Cover or Action.Sell
                open_val *= ((pos + qty) / pos)
                pos += qty
                intent = Intent.Decrease
                side = side

            # Get rid of anything but the date
            dt = txn.ts.to_period('B').to_timestamp()
            rows.append([dt, txn.ts, pid, txn.tid, txn.qty, txn.px, txn.fees, premium, open_val, pos, intent, side])

        df = pd.DataFrame.from_records(rows, columns=[TC.DT, TC.TS, TC.PID, TC.TID, TC.QTY, TC.PX, TC.FEES, TC.PREMIUM,
                                                      TC.OPEN_VAL, TC.POS, TC.INTENT, TC.ACTION])
        df.index.name = 'seq'
        return df

    def get_pid_txns(self, pid):
        pmask = self.frame[TC.PID] == pid
        assert len(pmask.index) == len(self.trades), 'assume 1-1 ratio of trade to row in frame'
        return tuple(pd.Series(self.trades)[pmask.values])

    def subset(self, pids):
        pmask = self.frame[TC.PID].isin(pids)
        if pmask.all():
            return self
        else:
            # 1 to 1 mapping of txn to row (so can figure out trades from mask)
            trds = tuple(pd.Series(self.trades)[pmask.values])
            # build the object
            result = Txns(trds, self.pricer, self.ret_calc)
            result._frame = self.frame.ix[pmask]
            if hasattr(self, '_profit_and_loss'):
                pl = self.profit_and_loss
                result._profit_and_loss = pl.subset(result)
            return result



