from collections import OrderedDict

import pandas as pd

from tia.analysis.plots import plot_return_on_dollar
from tia.analysis.model.interface import CostCalculator, EodMarketData, PositionColumns as PC
from tia.analysis.model.pos import Positions
from tia.analysis.model.ret import RoiiRetCalculator
from tia.analysis.model.txn import Txns
from tia.util.decorator import lazy_property


__all__ = ['SingleAssetPortfolio', 'PortfolioPricer', 'PortfolioSubset', 'PortfolioSummary']


class PortfolioPricer(CostCalculator, EodMarketData):
    def __init__(self, multiplier=1., closing_pxs=None, dvds=None):
        if not isinstance(closing_pxs, pd.Series):
            raise ValueError('closing_pxs must be a Series not {0}'.format(type(closing_pxs)))
        if dvds is not None and not isinstance(dvds, pd.Series):
            raise ValueError('dvds be a Series not {0}'.format(type(dvds)))

        self._multiplier = multiplier
        self._dvds = dvds
        self._closing_pxs = closing_pxs

    multiplier = property(lambda self: self._multiplier)
    dvds = property(lambda self: self._dvds)

    def truncate(self, before=None, after=None):
        return PortfolioPricer(self.multiplier, self._closing_pxs.truncate(before=before, after=after), dvds=self._dvds)

    def get_closing_pxs(self, start=None, end=None):
        pxs = self._closing_pxs
        if start or end:
            start = start or pxs.index[0]
            end = end or pxs.index[-1]
            pxs = pxs.ix[start:end]
        return pxs

    def get_mkt_val(self, pxs=None):
        """  return the market value series for the specified Series of pxs """
        pxs = self._closing_pxs if pxs is None else pxs
        return pxs * self.multiplier

    def get_premium(self, qty, px, ts=None):
        return -qty * px * self.multiplier

    def get_eod_frame(self):
        close = self.get_closing_pxs()
        mktval = self.get_mkt_val(close)
        dvds = self.dvds
        df = pd.DataFrame({'close': close, 'mkt_val': mktval, 'dvds': dvds})
        df.index.name = 'date'
        # drop dvds outside the px range
        return df.truncate(before=close.index[0], after=close.index[-1])


class SingleAssetPortfolio(object):
    def __init__(self, pricer, trades, ret_calc=None):
        """
        :param pricer: PortfolioPricer
        :param trades: list of Trade objects
        """
        self.trades = tuple(trades)
        self.pricer = pricer
        self.ret_calc = ret_calc or RoiiRetCalculator()

    txns = lazy_property(lambda self: Txns(self.trades, self.pricer, self.ret_calc), 'txns')
    positions = lazy_property(lambda self: Positions(self.txns), 'positions')

    def clear_cache(self):
        for attr in ['_txns', '_positions', '_long', '_short']:
            if hasattr(self, attr):
                delattr(self, attr)

    # Direct access to the series
    ltd_pl = property(lambda self: self.txns.pl.ltd)
    weekly_pl = property(lambda self: self.txns.pl.weekly)
    monthly_pl = property(lambda self: self.txns.pl.monthly)
    quarterly_pl = property(lambda self: self.txns.pl.quarterly)
    annual_pl = property(lambda self: self.txns.pl.annual)
    dly_pl = property(lambda self: self.txns.pl.dly)
    ltd_rets = property(lambda self: self.txns.rets.ltd)
    weekly_rets = property(lambda self: self.txns.rets.weekly)
    monthly_rets = property(lambda self: self.txns.rets.monthly)
    quarterly_rets = property(lambda self: self.txns.rets.quarterly)
    annual_rets = property(lambda self: self.txns.rets.annual)
    dly_rets = property(lambda self: self.txns.rets.dly)
    # direct access to details
    ltd_txn_pl_frame = property(lambda self: self.txns.pl.ltd_txn_frame)
    dly_txn_pl_frame = property(lambda self: self.txns.pl.dly_txn_frame)
    ltd_pl_frame = property(lambda self: self.txns.pl.ltd_frame)
    dly_pl_frame = property(lambda self: self.txns.pl.dly_frame)
    # direct access to stats
    weekly_ret_stats = property(lambda self: self.txns.rets.weekly_stats)
    monthly_ret_stats = property(lambda self: self.txns.rets.monthly_stats)
    quarterly_ret_stats = property(lambda self: self.txns.rets.quarterly_stats)
    annual_ret_stats = property(lambda self: self.txns.rets.annual_stats)
    dly_ret_stats = property(lambda self: self.txns.rets.dly_stats)
    weekly_pl_stats = property(lambda self: self.txns.pl.weekly_stats)
    monthly_pl_stats = property(lambda self: self.txns.pl.monthly_stats)
    quarterly_pl_stats = property(lambda self: self.txns.pl.quarterly_stats)
    annual_pl_stats = property(lambda self: self.txns.pl.annual_stats)
    dly_pl_stats = property(lambda self: self.txns.pl.dly_stats)

    position_frame = property(lambda self: self.positions.frame)

    def plot_ret_on_dollar(self, freq='M', title=None, show_maxdd=1, figsize=None, ax=None, append=0, label=None,
                           **plot_args):
        freq = freq.lower()
        if freq == 'a':
            rets = self.annual_rets
        elif freq == 'q':
            rets = self.quarterly_rets
        elif freq == 'm':
            rets = self.monthly_rets
        elif freq == 'w':
            rets = self.weekly_rets
        else:
            rets = self.dly_rets.asfreq('B')
        plot_return_on_dollar(rets, title=title, show_maxdd=show_maxdd, figsize=figsize, ax=ax, append=append,
                              label=label, **plot_args)

    def subset(self, pids):
        txns = self.txns
        stxns = txns.subset(pids)
        if stxns == txns:  # return same object
            return self
        else:
            # TODO: rethink logic - maybe split trades (l/s) in Portfolio constructor as now
            # passing split trades back to portfolio subset
            port = SingleAssetPortfolio(self.pricer, stxns.trades, ret_calc=self.ret_calc)
            port._txns = stxns
            if hasattr(self, '_positions'):
                port._positions = self.positions.subset(stxns)
            return port

    @lazy_property
    def long(self):
        return PortfolioSubset.longs(self)

    @lazy_property
    def short(self):
        return PortfolioSubset.shorts(self)

    winner = property(lambda self: PortfolioSubset.winners(self))
    loser = property(lambda self: PortfolioSubset.losers(self))

    def buy_and_hold(self, qty=1., start=None, end=None, start_px=None, end_px=None):
        """Construct a portfolio which opens a position with size qty at start (or first data in pricer) and
        continues to the specified end date. It uses the end of day market prices defined by the pricer
        (or prices supplied)

        :param qty:
        :param start: datetime
        :param end: datetime
        :param which: which price series to use for inital trade px
        :param ret_cacls: portfolio return calculator
        :return: SingleAssetPortfolio
        """
        from tia.analysis.model.trd import TradeBlotter

        pricer = self.pricer
        eod = pricer.get_eod_frame().close
        eod_start, eod_end = eod.index[0], eod.index[-1]

        start = start and pd.to_datetime(start) or eod_start
        end = end and pd.to_datetime(end) or eod_end

        if start != eod_start or end != eod_end:
            pricer = pricer.truncate(start, end)

        start_px = start_px or eod[start]
        end_px = end_px or eod[end]

        blotter = TradeBlotter()
        blotter.ts = start
        blotter.open(qty, start_px)
        blotter.ts = end
        blotter.close(end_px)
        trds = blotter.trades
        return SingleAssetPortfolio(pricer, trds, ret_calc=self.ret_calc)


class PortfolioSubset(object):
    @staticmethod
    def longs(port):
        return port.subset(port.positions.long_pids)

    @staticmethod
    def shorts(port):
        return port.subset(port.positions.short_pids)

    @staticmethod
    def winners(port):
        frame = port.positions.frame
        pids = frame[frame[PC.PL] >= 0].index
        return port.subset(pids)

    @staticmethod
    def losers(port):
        frame = port.positions.frame
        pids = frame[frame[PC.PL] < 0].index
        return port.subset(pids)

    @staticmethod
    def top_pl(port, n=10):
        pids = port.positions.frame[PC.PL].order()[-n:].index
        return port.subset(pids)

    @staticmethod
    def top_rets(port, n=10):
        pids = port.positions.frame[PC.RET].order()[-n:].index
        return port.subset(pids)

    @staticmethod
    def bottom_pl(port, n=10):
        pids = port.positions.frame[PC.PL].order()[:n].index
        return port.subset(pids)

    @staticmethod
    def bottom_rets(port, n=10):
        pids = port.positions.frame[PC.RET].order()[:n].index
        return port.subset(pids)

    @staticmethod
    def top_durations(port, n=10):
        pids = port.positions.frame[PC.DURATION].order()[-n:].index
        return port.subset(pids)

    @staticmethod
    def bottom_durations(port, n=10):
        pids = port.positions.frame[PC.DURATION].order()[:n].index
        return port.subset(pids)


class PortfolioSummary(object):
    def __init__(self):
        self.total_key = 'All'
        self.iter_fcts = []

    def __call__(self, port, analyze_fct):
        """
        analyze_fct: fct(port) which can return Series, or map of key to Series. If key to series, then
        the key is used as an additional index value.
        """
        results = []
        iter_fcts = self.iter_fcts
        lvls = len(iter_fcts)

        def _iter_all_lvls(lvl, keys, parent):
            if lvl < (lvls - 1):
                # exhaust combinations
                for key, child in iter_fcts[lvl](parent):
                    _iter_all_lvls(lvl + 1, keys + [key], child)
            else:
                # at the bottom
                for key, child in iter_fcts[lvl](parent):
                    idx_names = ['lvl{0}'.format(i + 1) for i in range(lvls)]
                    idx_vals = [[k] for k in keys + [key]]
                    idx = pd.MultiIndex.from_arrays(idx_vals, names=idx_names)
                    res = analyze_fct(child)
                    if isinstance(res, pd.Series):
                        res = res.to_frame().T
                        res.index = idx
                        results.append(res)
                    else:
                        for k, v in res.iteritems():
                            # prepend current levels to key name
                            v = v.to_frame().T
                            idx = pd.MultiIndex.from_arrays(idx_vals + [k], names=idx_names + ['lvl%s' % lvls])
                            v.index = idx
                            results.append(v)

        if lvls == 0:
            res = analyze_fct(port)
            if isinstance(res, pd.Series):
                res = res.to_frame().T
            results.append(res)
        else:
            _iter_all_lvls(0, [], port)
        return pd.concat(results)

    def add_iter_fct(self, siter):
        self.iter_fcts.append(siter)
        return self

    def include_win_loss(self, total=1):
        def _split_port(port):
            if total:
                yield self.total_key, port
            yield 'Winner', PortfolioSubset.winners(port)
            yield 'Loser', PortfolioSubset.losers(port)

        self.add_iter_fct(_split_port)
        return self

    def include_long_short(self, total=1):
        def _split_port(port):
            if total:
                yield self.total_key, port
            yield 'Long', port.long
            yield 'Short', port.short

        self.add_iter_fct(_split_port)
        return self

    @staticmethod
    def analyze_returns(port):
        mstats = port.monthly_ret_stats
        dstats = port.dly_ret_stats
        stats = port.positions.stats
        data = OrderedDict()
        data[('port', 'cagr')] = mstats.total_ann
        data[('port', 'mret avg')] = mstats.ret_avg
        data[('port', 'mret avg ann')] = mstats.ret_avg_ann
        data[('port', 'mret std ann')] = mstats.std_ann
        data[('port', 'sharpe ann')] = mstats.sharpe_ann
        data[('port', 'sortino')] = mstats.sortino
        data[('port', 'maxdd')] = dstats.maxdd
        data[('port', 'maxdd dt')] = dstats.maxdd_dt
        data[('port', 'avg dd')] = dstats.dd_avg
        data[('port', 'nmonths')] = mstats.cnt
        # pos data
        data[('pos', 'cnt')] = stats.cnt
        data[('pos', 'win cnt')] = stats.win_cnt
        data[('pos', 'lose cnt')] = stats.lose_cnt
        data[('pos', 'winpct')] = stats.win_pct
        data[('pos', 'ret avg')] = stats.ret_avg
        data[('pos', 'ret std')] = stats.ret_std
        data[('pos', 'ret min')] = stats.ret_min
        data[('pos', 'ret max')] = stats.ret_max
        data[('pos', 'dur avg')] = stats.duration_avg
        data[('pos', 'dur max')] = stats.duration_max
        return pd.Series(data, index=pd.MultiIndex.from_tuples(data.keys()))

    @staticmethod
    def analyze_pl(port):
        mstats = port.monthly_pl_stats
        dstats = port.dly_pl_stats
        stats = port.positions.stats
        data = OrderedDict()
        data[('port', 'ltd')] = mstats.pl.sum()
        data[('port', 'mpl avg')] = mstats.avg
        data[('port', 'mpl std')] = mstats.std
        data[('port', 'mpl std ann')] = mstats.std_ann
        data[('port', 'mpl max')] = mstats.pl.max()
        data[('port', 'mpl min')] = mstats.pl.min()
        data[('port', 'maxdd')] = dstats.maxdd
        data[('port', 'maxdd dt')] = dstats.maxdd_dt
        data[('port', 'avg dd')] = dstats.dd_avg
        data[('port', 'nmonths')] = mstats.cnt
        # pos data
        data[('pos', 'cnt')] = stats.cnt
        data[('pos', 'win cnt')] = stats.win_cnt
        data[('pos', 'lose cnt')] = stats.lose_cnt
        data[('pos', 'winpct')] = stats.win_pct
        data[('pos', 'pl avg')] = stats.pl_avg
        data[('pos', 'pl std')] = stats.pl_std
        data[('pos', 'pl min')] = stats.pl_min
        data[('pos', 'pl max')] = stats.pl_max
        return pd.Series(data, index=pd.MultiIndex.from_tuples(data.keys()))






