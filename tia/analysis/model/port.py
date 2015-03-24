import pandas as pd

from tia.analysis.plots import plot_return_on_dollar
from tia.analysis.model.interface import CostCalculator, EodMarketData, PositionColumns as PC
from tia.analysis.model.pos import Positions
from tia.analysis.model.ret import RoiiRets
from tia.analysis.model.txn import Txns
from tia.util.decorator import lazy_property


__all__ = ['SingleAssetPortfolio', 'PortfolioPricer']


class PortfolioPricer(CostCalculator, EodMarketData):
    """
    A Portfolio Pricer allows the user to speciy the paramters/objects necessary to price a Portfolio

    Parameters
    ----------
    multiplier : non-zero integer, optional
                 security multiplier which applied to px results in the market value
    closing_pxs: Series
    dvds: Series
    """

    def __init__(self, multiplier=1., closing_pxs=None, dvds=None):
        if not isinstance(closing_pxs, pd.Series):
            raise ValueError('closing_pxs must be a Series not {0}'.format(type(closing_pxs)))
        if dvds is not None and not isinstance(dvds, pd.Series):
            raise ValueError('dvds be a Series not {0}'.format(type(dvds)))

        self._multiplier = multiplier
        self._dvds = dvds
        self._closing_pxs = closing_pxs

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def dvds(self):
        return self._dvds

    def truncate(self, before=None, after=None):
        return PortfolioPricer(self.multiplier, self._closing_pxs.truncate(before=before, after=after), dvds=self._dvds)

    def get_closing_pxs(self, start=None, end=None):
        pxs = self._closing_pxs
        if start or end:
            start = start or pxs.index[0]
            end = end or pxs.index[-1]
            pxs = pxs.ix[start:end]
        return pxs

    def get_mkt_val(self, pxs=None, ts=None):
        """  return the market value series for the specified Series of pxs """
        if pxs is None:
            pxs = self._closing_pxs
        return pxs * self.multiplier

    def get_premium(self, qty, px, ts=None):
        return -qty * px * self.multiplier

    def get_frame(self, truncate_dvds=True):
        """
        Build a DataFrame with columns ('close', 'mktval', 'dvd')

        Parameters
        ----------
        truncate_dvds : bool, optional
                        If True, then drop any dvd values which are outside the pxs range

        Returns
        -------
        DataFrame
        """
        close = self.get_closing_pxs()
        mktval = self.get_mkt_val(close)
        dvds = self.dvds
        df = pd.DataFrame({'close': close, 'mkt_val': mktval, 'dvds': dvds})
        df.index.name = 'dt'
        # ignore dvds outside the px range
        if truncate_dvds:
            return df.truncate(before=close.index[0], after=close.index[-1])
        else:
            return df


class SingleAssetPortfolio(object):
    def __init__(self, pricer, trades, rets_calc=None):
        """
        :param pricer: PortfolioPricer
        :param trades: list of Trade objects
        """
        self.trades = tuple(trades)
        self.pricer = pricer
        self.rets_calc = rets_calc or RoiiRets()

    def clear_cache(self):
        for attr in ['_txns', '_positions', '_long', '_short']:
            if hasattr(self, attr):
                delattr(self, attr)

    @lazy_property
    def txns(self):
        """Return the Txns object for this portfolio"""
        txns = Txns(self.trades, self.pricer)
        txns.rets = self.rets_calc
        return txns

    @lazy_property
    def positions(self):
        """Return Positions object"""
        return Positions(self.txns)

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

    def plot_ret_on_dollar(self, freq='M', title=None, show_maxdd=1, figsize=None, ax=None, append=0):
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
        plot_return_on_dollar(rets, title=title, show_maxdd=show_maxdd, figsize=figsize, ax=None, append=append)

    def subset(self, pids):
        txns = self.txns
        stxns = txns.subset(pids)
        if stxns == txns:  # return same object
            return self
        else:
            # TODO: rethink logic - maybe split trades (l/s) in Portfolio constructor as now
            # passing split trades back to portfolio subset
            port = SingleAssetPortfolio(self.pricer, stxns.trades)
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


