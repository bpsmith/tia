from collections import OrderedDict

import pandas as pd
import numpy as np

from tia.analysis.model.txn import Intent, Action
from tia.analysis.model.interface import TxnPlColumns as TPL, PositionColumns as PC
from tia.util.decorator import lazy_property


__all__ = ['Positions', 'Position', 'State', 'Side']


class Side(object):
    Long = 'Long'
    Short = 'Short'


class State(object):
    Open = 'Open'
    Closed = 'Closed'


class Position(object):
    def __init__(self, pid, pl, trades, performance):
        """
        :param pid: position id
        :param dly_txn_pl: daily transaction level DataFrame
        :param trades: trade array
        """
        self.pid = pid
        self.pl = pl
        self.performance = performance
        self.trades = trades
        self.first = first = pl.txn_frame.iloc[0]
        self.last = last = pl.txn_frame.iloc[-1]
        self.is_closed = last[TPL.TXN_INTENT] == Intent.Close
        self.is_long = is_long = first[TPL.TXN_ACTION] == Action.Buy
        self.is_short = not is_long
        self.pid = first[TPL.PID]
        self.open_dt = first[TPL.DT]
        self.open_premium = first[TPL.TXN_PREMIUM]
        self.open_qty = first[TPL.TXN_QTY]
        self.open_px = first[TPL.TXN_PX]
        self.close_px = last[TPL.TXN_PX]
        self.close_dt = last[TPL.DT]
        # TODO - should use a time delta and determine the number of days
        self.duration = len(pl.ltd_txn_frame[TPL.DT].drop_duplicates())
        self.ntxns = len(trades)

    state = property(lambda self: self.is_closed and State.Closed or State.Open)
    side = property(lambda self: self.is_long and Side.Long or Side.Short)

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
    sides = property(lambda self: self.frame[PC.SIDE])
    long_pids = property(lambda self: self.frame[self.frame[PC.SIDE] == Side.Long].index)
    short_pids = property(lambda self: self.frame[self.frame[PC.SIDE] == Side.Short].index)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, pid):
        if pid == 0:
            raise ValueError('pid must be non-zero')
        txns = self.txns
        trds = txns.get_pid_txns(pid)
        pl = txns.pl.truncate(pid=pid)
        mask = txns.pl.txn_details.get_pid_mask(pid)
        performance = txns.performance.filter(mask)
        return Position(pid, pl, trds, performance)

    def __iter__(self):
        for pid in self.pids:
            yield self[pid]

    def subset(self, subtxns):
        """Construct a new Positions object from the new Txns object (which is assumed to be a subset)
        of current Txns object"""
        result = Positions(subtxns)
        if hasattr(self, '_frame'):
            result._frame = self._frame.ix[subtxns.pids]
            # passing in array results in index name being removed for some reason???
            if result._frame.index.name != self._frame.index.name:
                result._frame.index.name = self._frame.index.name
        return result

    @lazy_property
    def frame(self):
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
                pos.pl.ltd_txn.iloc[-1],
                pos.performance.ltd_txn.iloc[-1],
                pos.duration,
                pos.ntxns,
                pos.state
            ])
        cols = [PC.PID, PC.SIDE, PC.OPEN_DT, PC.CLOSE_DT, PC.OPEN_QTY, PC.OPEN_PX, PC.CLOSE_PX, PC.OPEN_PREMIUM, \
                PC.PL, PC.RET, PC.DURATION, PC.NUM_TXNS, PC.STATE]
        f = pd.DataFrame.from_records(vals, columns=cols)
        f[PC.PID] = f[PC.PID].astype(int)
        return f.set_index(PC.PID)

    @lazy_property
    def stats(self):
        return PositionsStats(self)

    def _repr_html_(self):
        return self.frame._repr_html_()

    def plot_rets(self, ls=1, ax=None):
        """Plot each of the position returns

        :param ls: True, if positions should be broken into long/short
        :param ax: Axes
        :param regr: True, if regression line is shown
        """
        import matplotlib.pyplot as plt
        from tia.util.mplot import AxesFormat

        if ax is None:
            ax = plt.gca()

        frame = self.frame

        if not ls:
            ax.scatter(frame.index, frame.ret, c='k', marker='o', label='All')
        else:
            if len(self.long_pids) > 0:
                lframe = frame.ix[frame.index.isin(self.long_pids)]
                ax.scatter(lframe.index, lframe.ret, c='k', marker='o', label='Long')
            if len(self.short_pids) > 0:
                sframe = frame.ix[frame.index.isin(self.short_pids)]
                ax.scatter(sframe.index, sframe.ret, c='r', marker='o', label='Short')

        # set some boundaries
        AxesFormat().Y.percent().apply()
        ax.set_xlim(0, frame.index.max() + 3)
        ax.set_xlabel('pid')
        ax.set_ylabel('return')
        ax.legend(loc='upper left')
        return ax

    def plot_ret_range(self, ax=None, ls=0, dur=0):
        """Plot the return range for each position

        :param ax: Axes
        """
        import matplotlib.pyplot as plt
        from tia.util.mplot import AxesFormat

        if ax is None:
            ax = plt.gca()

        frame = self.frame
        pids = frame.index

        min_rets = pd.Series([self[pid].performance.ltd_txn.min() for pid in pids], index=pids)
        max_rets = pd.Series([self[pid].performance.ltd_txn.max() for pid in pids], index=pids)

        if not ls:
            s = frame.duration + 20 if dur else 20
            ax.scatter(frame.index, frame.ret, s=s, c='k', marker='o', label='All')
            ax.vlines(pids, min_rets, max_rets)
        else:
            if len(self.long_pids) > 0:
                lframe = frame.ix[frame.index.isin(self.long_pids)]
                s = lframe.duration + 20 if dur else 20
                ax.scatter(lframe.index, lframe.ret, s=s, c='k', marker='o', label='Long')
                ax.vlines(lframe.index, min_rets[lframe.index], max_rets[frame.index])
            if len(self.short_pids) > 0:
                sframe = frame.ix[frame.index.isin(self.short_pids)]
                s = sframe.duration + 20 if dur else 20
                ax.scatter(sframe.index, sframe.ret, s=s, c='r', marker='o', label='Short')
                ax.vlines(sframe.index, min_rets[sframe.index], max_rets[sframe.index])

        AxesFormat().Y.percent().apply()
        ax.axhline(color='k', linestyle='--')
        ax.set_xlim(0, frame.index.max() + 3)
        ax.set_xlabel('pid')
        ax.set_ylabel('return')
        ax.legend(loc='upper left')
        return ax


class PositionsStats(object):
    def __init__(self, positions):
        self.positions = positions

    _frame = property(lambda self: self.positions.frame)
    _loser_frame = property(lambda self: self._frame.ix[self._frame[PC.RET] < 0])
    _winner_frame = property(lambda self: self._frame.ix[self._frame[PC.RET] >= 0])

    win_cnt = property(lambda self: len(self._winner_frame.index))
    lose_cnt = property(lambda self: len(self._loser_frame.index))
    cnt = property(lambda self: len(self._frame.index))
    win_pct = property(lambda self: np.divide(float(self.win_cnt), float(self.cnt)))

    @lazy_property
    def pl_summary(self):
        return self._frame[PC.PL].describe()

    @lazy_property
    def ret_summary(self):
        return self._frame[PC.RET].describe()

    @lazy_property
    def duration_summary(self):
        return self._frame[PC.DURATION].describe()

    pl_mean = property(lambda self: self.pl_summary['mean'])
    pl_avg = pl_mean
    pl_min = property(lambda self: self.pl_summary['min'])
    pl_max = property(lambda self: self.pl_summary['max'])
    pl_std = property(lambda self: self.pl_summary['std'])

    ret_mean = property(lambda self: self.ret_summary['mean'])
    ret_avg = ret_mean
    ret_min = property(lambda self: self.ret_summary['min'])
    ret_max = property(lambda self: self.ret_summary['max'])
    ret_std = property(lambda self: self.ret_summary['std'])

    duration_mean = property(lambda self: self.duration_summary['mean'])
    duration_avg = duration_mean
    duration_min = property(lambda self: self.duration_summary['min'])
    duration_max = property(lambda self: self.duration_summary['max'])
    duration_std = property(lambda self: self.duration_summary['std'])

    # consecutive winners/losers
    @lazy_property
    def consecutive_frame(self):
        """Return a DataFrame with columns cnt, pids, pl. cnt is the number of pids in the sequence. pl is the pl sum"""
        if self._frame.empty:
            return pd.DataFrame(columns=['pids', 'pl', 'cnt', 'is_win'])
        else:
            vals = (self._frame[PC.RET] >= 0).astype(int)
            seq = (vals.shift(1) != vals).astype(int).cumsum()

            def _do_apply(sub):
                return pd.Series({
                    'pids': sub.index.values,
                    'pl': sub[PC.PL].sum(),
                    'cnt': len(sub.index),
                    'is_win': sub[PC.RET].iloc[0] >= 0,
                })

            return self._frame.groupby(seq).apply(_do_apply)

    @property
    def consecutive_win_frame(self):
        cf = self.consecutive_frame
        return cf.ix[cf.is_win]

    @property
    def consecutive_loss_frame(self):
        cf = self.consecutive_frame
        return cf.ix[~cf.is_win]

    consecutive_wins_max = property(lambda self: self.consecutive_win_frame.cnt.max())
    consecutive_wins_avg = property(lambda self: self.consecutive_win_frame.cnt.mean())
    consecutive_losses_max = property(lambda self: self.consecutive_loss_frame.cnt.max())
    consecutive_losses_avg = property(lambda self: self.consecutive_loss_frame.cnt.mean())

    @property
    def series(self):
        data = OrderedDict()
        data['cnt'] = self.cnt
        data['win_pct'] = self.win_pct
        data['ret_avg'] = self.ret_avg
        data['ret_std'] = self.ret_std
        data['ret_min'] = self.ret_min
        data['ret_max'] = self.ret_max
        data['pl_avg'] = self.pl_avg
        data['pl_std'] = self.pl_std
        data['pl_min'] = self.pl_min
        data['pl_max'] = self.pl_max
        data['duration_avg'] = self.duration_avg
        data['duration_std'] = self.duration_std
        data['consecutive_win_cnt_max'] = self.consecutive_wins_max
        data['consecutive_loss_cnt_max'] = self.consecutive_losses_max
        return pd.Series(data, name='positions')

    def _repr_html_(self):
        from tia.util.fmt import new_dynamic_formatter

        fmt = new_dynamic_formatter(method='row', precision=2, pcts=1, trunc_dot_zeros=1, parens=1)
        return fmt(self.series.to_frame())._repr_html_()


