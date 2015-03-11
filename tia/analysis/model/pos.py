import pandas as pd
import numpy as np

from tia.analysis.model.txn import Intent, Action

from tia.analysis.model.interface import PlColumns as PL, PositionColumns as PC
from tia.util.decorator import lazy_property


__all__ = ['Positions', 'Position', 'State', 'Side']


class Side(object):
    Long = 'Long'
    Short = 'Short'


class State(object):
    Open = 'Open'
    Closed = 'Closed'


class Position(object):
    def __init__(self, pid, dly_txn_pl, trades):
        """
        :param pid: position id
        :param dly_txn_pl: daily transaction level DataFrame
        :param trades: trade array
        """
        self.pid = pid
        self.dly_txn_pl = dly_txn_pl
        self.trades = trades
        self.first = first = dly_txn_pl.iloc[0]
        self.last = last = dly_txn_pl.iloc[-1]
        self.is_closed = is_closed = last[PL.TXN_INTENT] == Intent.Close
        self.is_long = is_long = first[PL.TXN_ACTION] == Action.Buy
        self.is_short = not is_long
        self.pid = first[PL.PID]
        self.open_dt = first[PL.DT]
        self.open_premium = first[PL.TXN_PREMIUM]
        self.open_qty = first[PL.TXN_QTY]
        self.open_px = first[PL.TXN_PX]
        self.close_px = last[PL.TXN_PX]
        self.close_dt = last[PL.DT]
        cumpl = dly_txn_pl[PL.PL].cumsum()
        self.pl = cumpl.iloc[-1]
        self.pl_min = cumpl.min()
        self.pl_max = cumpl.max()
        self.duration = len(dly_txn_pl[PL.DT].drop_duplicates())
        self.ntxns = len(trades)

    state = property(lambda self: self.is_closed and State.Closed or State.Open)
    side = property(lambda self: self.is_long and Side.Long or Side.Short)

    @property
    def roii(self):
        '''return on intial investment'''
        return self.pl / abs(self.open_premium)

    @property
    def dly_pl(self):
        return self.dly_txn_pl[[PL.DT, PL.PL]].set_index(PL.DT).resample('B', how='sum', kind='period')[PL.PL].dropna()

    @property
    def dly_pl_path(self):
        return self.dly_pl.cumsum()

    @property
    def dly_roii(self):
        '''Daily return on initial investment
        :return: the dly return series
        '''
        # TODO - handle increase/decrease logic
        cost = abs(self.open_premium)
        rets = (cost + self.dly_pl_path).pct_change()
        rets.iloc[0] = 0
        return rets

    def __repr__(self):
        kwargs = {
            'class': self.__class__.__name__,
            'pid': self.pid,
            'side': self.side,
            'open_dt': self.open_dt,
            'close_dt': self.close_dt
        }
        return '<{class}({pid}, {side}, open_dt={open_dt}, close_dt={close_dt})>'.format(**kwargs)


class Positions(object):
    def __init__(self, txns):
        """
        TODO: possibly cache positions and share with subset
        :param txns: Txns object
        """
        self.txns = txns

    pids = property(lambda self: self.txns.pids)
    sides = property(lambda self: self.summary[PC.SIDE])
    long_pids = property(lambda self: self.pids[self.sides == Side.Long])
    short_pids = property(lambda self: self.pids[self.sides == Side.Short])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, pid):
        if pid == 0:
            raise ValueError('pid must be non-zero')
        trds = self.txns.pid_to_trades[pid]
        dly = self.txns.dly_txn_pl
        sub = dly.ix[dly[PL.PID] == pid]
        return Position(pid, sub, trds)

    def __iter__(self):
        for pid in self.pids:
            yield [self[pid]]

    def subset(self, txns):
        """Construct a new Positions object from the new Txns object (which is assumed to be a subset)
        of current Txns object"""
        result = Positions(txns)
        if hasattr(self, '_summary'):
            summary = self._summary
            allpids = summary[PC.PID]
            result._summary = self._summary.ix[allpids.isin(txns.pids)]
        return result

    @lazy_property
    def summary(self):
        vals = []
        for pos in iter(self):
            vals.append([
                pos.pid,
                pos.side,
                pos.open_dt,
                pos.close_dt,
                pos.open_qty,
                pos.open_px,
                pos.close_px,
                pos.open_premium,
                pos.pl_min,
                pos.pl_max,
                pos.pl,
                pos.duration,
                pos.ntxns,
                pos.state
            ])
        cols = [PC.PID, PC.SIDE, PC.OPEN_DT, PC.CLOSE_DT, PC.OPEN_QTY, PC.OPEN_PX, PC.CLOSE_PX, PC.OPEN_PREMIUM, \
                PC.PL_MIN, PC.PL_MAX, PC.PL, PC.DURATION, PC.NUM_TXNS, PC.STATE]
        return pd.DataFrame.from_records(vals, columns=cols)


class PositionsAnalyzer(object):
    # TODO - fix this
    def __init__(self, frame, zero_is_win=1):
        winmask = frame.pl >= 0 if zero_is_win else frame.pl > 0
        self.zero_is_win = zero_is_win
        self.winner_frame = frame.ix[winmask]
        self.loser_frame = frame.ix[~winmask]
        self.frame = frame

    win_cnt = property(lambda self: len(self.winner_frame.index))
    lose_cnt = property(lambda self: len(self.loser_frame.index))
    cnt = property(lambda self: len(self.frame.index))
    win_pct = property(lambda self: np.divide(float(self.win_cnt), float(self.cnt)))

    def describe(self, pl=True, duration=True, roii=True, percentiles=None):
        def _doit(col, alias):
            cols = ['Winners', 'Losers', 'Total']
            arrs = [self.winner_frame[col], self.loser_frame[col], self.frame[col]]
            f = pd.DataFrame(dict(zip(cols, arrs)), columns=cols).describe()
            f.columns = pd.MultiIndex.from_arrays([[alias] * 3, f.columns.values])
            f.columns.names = ['stat', 'side']
            return f.T

        pieces = []
        pl and pieces.append(_doit('pl', 'pl'))
        roii and pieces.append(_doit('roii', 'roii'))
        duration and pieces.append(_doit('duration', 'duration'))
        return pd.concat(pieces)

    def describe_pl(self, percentiles=None):
        return self.describe(pl=True, duration=False, roii=False, percentiles=percentiles)

    def describe_duration(self, percentiles=None):
        return self.describe(pl=False, duration=True, roii=False, percentiles=percentiles)

    def describe_roii(self, percentiles=None):
        return self.describe(pl=False, duration=False, roii=True, percentiles=percentiles)

    def iter_by_year(self):
        """Iterate the tuple of (year, PositionFrame) by breaking up the positions in the year a position was open"""
        years = self.frame.open_dt.apply(lambda x: x.year)
        for yr in years.unique():
            pass
            #yield yr, PositionFrame(self.frame.ix[years == yr], zero_is_win=self.zero_is_win)

    def describe_by_year(self, pl=True, duration=True, roii=True, percentiles=None, inc_ltd=True):
        years = self.frame.open_dt.apply(lambda x: x.year)
        # fill in missing years?
        def get_desc(yr, pf):
            desc = pf.describe(pl=pl, duration=duration, roii=roii, percentiles=percentiles)
            desc['year'] = yr
            desc.set_index('year', append=1, inplace=1)
            return desc

        pieces = [get_desc(yr, pf) for yr, pf in self.iter_by_year()]
        inc_ltd and pieces.append(get_desc('ltd', self))
        return pd.concat(pieces)




