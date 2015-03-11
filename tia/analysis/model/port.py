import pandas as pd
import numpy as np

from tia.analysis.model.interface import CostCalculator, EodMarketData
from tia.analysis.model.pos import Positions
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
    def __init__(self, pricer, trades):
        """
        :param pricer: PortfolioPricer
        :param trades: list of Trade objects
        """
        self.trades = tuple(trades)
        self.pricer = pricer

    @lazy_property
    def txns(self):
        """Return the Txns object for this portfolio"""
        return Txns(self.trades, self.pricer)

    @lazy_property
    def positions(self):
        """Return Positions object"""
        return Positions(self.txns)

    # Easier access
    ltd_txn_pl = property(lambda self: self.txns.pl.ltd_txn)
    dly_txn_pl = property(lambda self: self.txns.pl.dly_txn)
    ltd_pl = property(lambda self: self.txns.pl.ltd)
    dly_pl = property(lambda self: self.txns.pl.dly)
    position_summary = property(lambda self: self.positions.summary)

    def subset(self, pids):
        txns = self.txns
        stxns = txns.subset(pids)
        if stxns == txns:  # return same object
            return self
        else:
            # TODO: rethink logic - maybe split trades (l/s) in Portfolio constructor as now
            # passing split trades back to portfolio
            port = SingleAssetPortfolio(self.pricer, stxns.trades)
            port._txns = stxns
            if hasattr(self, '_positions'):
                port._positions = self.positions.subset(stxns)
            return port

    @lazy_property
    def long_only(self):
        return self.subset(self.positions.long_pids)

    @lazy_property
    def short_only(self):
        return self.subset(self.positions.short_pids)


