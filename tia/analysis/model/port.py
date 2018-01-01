from collections import OrderedDict

import pandas as pd

from tia.analysis.model.interface import CostCalculator, EodMarketData, PositionColumns as PC
from tia.analysis.model.pos import Positions
from tia.analysis.model.ret import RoiiRetCalculator
from tia.analysis.model.txn import Txns
from tia.analysis.util import insert_level
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
        self._ret_calc = ret_calc or RoiiRetCalculator()

    txns = lazy_property(lambda self: Txns(self.trades, self.pricer, self.ret_calc), 'txns')
    positions = lazy_property(lambda self: Positions(self.txns), 'positions')
    pl = property(lambda self: self.txns.pl)
    performance = property(lambda self: self.txns.performance)

    # --------------------------------------------------
    # direct access to common attributes
    dly_pl = property(lambda self: self.pl.dly)
    monthly_pl = property(lambda self: self.pl.monthly)
    dly_rets = property(lambda self: self.performance.dly)
    monthly_rets = property(lambda self: self.performance.monthly)

    @property
    def ret_calc(self):
        return self._ret_calc

    @ret_calc.setter
    def ret_calc(self, calc):
        self._ret_calc = calc
        if hasattr(self, '_txns'):
            self.txns.ret_calc = calc

    def clear_cache(self):
        for attr in ['_txns', '_positions', '_long', '_short']:
            if hasattr(self, attr):
                delattr(self, attr)

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

    def buy_and_hold(self, qty=1., start_dt=None, end_dt=None, start_px=None, end_px=None):
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

        eod = self.pricer.get_eod_frame().close

        start_dt = start_dt and pd.to_datetime(start_dt) or eod.index[0]
        start_px = start_px or eod.asof(start_dt)
        end_dt = end_dt and pd.to_datetime(end_dt) or eod.index[-1]
        end_px = end_px or eod.asof(end_dt)

        pricer = self.pricer.trunace(start_dt, end_dt)
        blotter = TradeBlotter()
        blotter.ts = start_dt
        blotter.open(qty, start_px)
        blotter.ts = end_dt
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

    def __call__(self, port, analyze_fct=None):
        """  analyze_fct: fct(port) which can return Series, or map of key to Series. If key to series, then
        the key is used as an additional index value.

        :param port: Portfolio or dict of key->Portfolio
        :param analyze_fct:
        :return:
        """
        iter_fcts = self.iter_fcts
        lvls = len(iter_fcts)

        analyze_fct = self.analyze_returns if analyze_fct is None else analyze_fct

        def _iter_all_lvls(lvl, keys, parent, results):
            if lvl < (lvls - 1):
                # exhaust combinations
                for key, child in iter_fcts[lvl](parent):
                    _iter_all_lvls(lvl + 1, keys + [key], child, results)
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
                        for k, v in res.items():
                            # prepend current levels to key name
                            v = v.to_frame().T
                            idx = pd.MultiIndex.from_arrays(idx_vals + [k], names=idx_names + ['lvl%s' % lvls])
                            v.index = idx
                            results.append(v)

        if lvls == 0:
            def _get_res(p):
                res = analyze_fct(p)
                return res.to_frame().T if isinstance(res, pd.Series) else res

            if hasattr(port, 'iteritems'):
                pieces = []
                for k, p in port.items():
                    res = _get_res(p)
                    defidx = res.index.nlevels == 1 and (res.index == 0).all()
                    res = insert_level(res, k, axis=1, level_name='lvl1')
                    if defidx:
                        res.index = res.index.droplevel(1)
                    pieces.append(res)
                return pd.concat(pieces)
            else:
                return _get_res(port)
        else:
            if hasattr(port, 'iteritems'):
                pieces = []
                for k, p in port.items():
                    results = []
                    _iter_all_lvls(0, [], p, results)
                    tmp = pd.concat(results)
                    tmp.index.names = ['lvl%s' % (i + 2) for i in range(len(tmp.index.names))]
                    tmp = insert_level(tmp, k, level_name='lvl1', axis=1)
                    pieces.append(tmp)
                return pd.concat(pieces)
            else:
                results = []
                _iter_all_lvls(0, [], port, results)
                return pd.concat(results)

    def add_iter_fct(self, siter):
        self.iter_fcts.append(siter)
        return self

    def include_win_loss(self, total=1):
        def _split_port(port):
            if total:
                yield self.total_key, port
            yield 'winner', PortfolioSubset.winners(port)
            yield 'loser', PortfolioSubset.losers(port)

        self.add_iter_fct(_split_port)
        return self

    def include_long_short(self, total=1):
        def _split_port(port):
            if total:
                yield self.total_key, port
            yield 'long', port.long
            yield 'short', port.short

        self.add_iter_fct(_split_port)
        return self

    @staticmethod
    def analyze_returns(port):
        monthly = port.performance.monthly_details
        dly = port.performance.dly_details
        stats = port.positions.stats
        data = OrderedDict()
        data[('port', 'ltd ann')] = monthly.ltd_ann
        data[('port', 'mret avg')] = monthly.mean
        data[('port', 'mret avg ann')] = monthly.mean_ann
        data[('port', 'mret std ann')] = monthly.std_ann
        data[('port', 'sharpe ann')] = monthly.sharpe_ann
        data[('port', 'sortino')] = monthly.sortino
        data[('port', 'maxdd')] = dly.maxdd
        data[('port', 'maxdd dt')] = dly.maxdd_dt
        data[('port', 'avg dd')] = dly.dd_avg
        data[('port', 'nmonths')] = monthly.cnt
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
        return pd.Series(data, index=pd.MultiIndex.from_tuples(list(data.keys())))

    @staticmethod
    def analyze_pl(port):
        monthly = port.pl.monthly_details
        dstats = port.pl.dly_details
        stats = port.positions.stats
        data = OrderedDict()
        data[('port', 'ltd')] = monthly.ltd_frame.pl.iloc[-1]
        data[('port', 'mpl avg')] = monthly.mean
        data[('port', 'mpl std')] = monthly.std
        data[('port', 'mpl std ann')] = monthly.std_ann
        data[('port', 'mpl max')] = monthly.frame.pl.max()
        data[('port', 'mpl min')] = monthly.frame.pl.min()
        data[('port', 'maxdd')] = dstats.maxdd
        data[('port', 'maxdd dt')] = dstats.maxdd_dt
        data[('port', 'avg dd')] = dstats.dd_avg
        data[('port', 'nmonths')] = monthly.cnt
        # pos data
        data[('pos', 'cnt')] = stats.cnt
        data[('pos', 'win cnt')] = stats.win_cnt
        data[('pos', 'lose cnt')] = stats.lose_cnt
        data[('pos', 'winpct')] = stats.win_pct
        data[('pos', 'pl avg')] = stats.pl_avg
        data[('pos', 'pl std')] = stats.pl_std
        data[('pos', 'pl min')] = stats.pl_min
        data[('pos', 'pl max')] = stats.pl_max
        return pd.Series(data, index=pd.MultiIndex.from_tuples(list(data.keys())))
