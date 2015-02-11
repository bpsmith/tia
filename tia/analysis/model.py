"""
Transactions
---------
A structure to track transactions for a single asset. Implements a simple open average
pnl calculation.

"""

import pandas as pd
import numpy as np


__all__ = ['SingleAssetPortfolio', 'Trade', 'INTENT_DECREASE', 'INTENT_INCREASE', 'PortfolioPricer',
           'INTENT_CLOSE', 'INTENT_OPEN', 'SIDE_SELL', 'SIDE_SELL_SHORT', 'SIDE_COVER', 'SIDE_BUY']

INTENT_OPEN = 1
INTENT_CLOSE = 2
INTENT_INCREASE = 3
INTENT_DECREASE = 4

INTENT_LABELS = {
    INTENT_OPEN: 'Open',
    INTENT_CLOSE: 'Close',
    INTENT_INCREASE: 'Increase',
    INTENT_DECREASE: 'Decrease'
}

SIDE_BUY = 1
SIDE_SELL = 2
SIDE_SELL_SHORT = 3
SIDE_COVER = 4

SIDE_LABELS = {
    SIDE_BUY: 'Buy',
    SIDE_SELL: 'Sell',
    SIDE_SELL_SHORT: 'SellShort',
    SIDE_COVER: 'Cover',
}

is_decrease = lambda q1, q2: (q1 * q2) < 0
is_increase = lambda q1, q2: (q1 * q2) > 0
crosses_zero = lambda q1, q2: ((q1 + q2) * q1) < 0


def has_weekends(index):
    """ return if this index contains Saturday or Sunday in it"""
    return 5 in index.dayofweek or 6 in index.dayofweek


def force_freq(series_or_ts, freq='B'):
    """ return a series with a PeriodIndex which conforms to the specified frequency"""
    if isinstance(series_or_ts, pd.Period):
        p = series_or_ts
        if p.freq != freq:
            p = p.asfreq(freq)
        return p
    elif isinstance(series_or_ts, pd.Series):
        series = series_or_ts
        if isinstance(series.index, pd.PeriodIndex):
            series = series.asfreq(freq)
        else:
            series = series.to_period(freq)
        return series
    else:
        return pd.to_datetime(series_or_ts).to_period(freq)


def iter_txns(trds):
    """ iterator of trades which splits trades to ensure proper long/short accounting"""
    pos = 0
    for trd in trds:
        if pos != 0 and is_decrease(pos, trd.qty) and crosses_zero(pos, trd.qty):
            # Split to make accounting for long/short possible
            closing_trd, opening_trd = trd.split(-pos)
            yield closing_trd
            pos = opening_trd.qty
            yield opening_trd
        else:
            pos += trd.qty
            yield trd


def iter_txn_grps(trds):
    """ iterator or trade arrays (positions), which are either long or short (splitting may occur)"""
    pos = 0
    txns = []
    for txn in trds:
        txns.append(txn)
        pos += txn.qty
        if pos == 0:
            yield txns
            txns = []
    if len(txns) > 0:
        yield txns


class PortfolioPricer(object):
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

    def get_mkt_val(self, pxs=None):
        """  return the market value series for the specified Series of pxs """
        if pxs is None:
            pxs = self._closing_pxs
        return pxs * self.multiplier

    def get_premium(self, qty, px):
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


class Trade(object):
    """Trade Model

    Parameters
    ----------
    tid: integer
         trade identifier
    ts: DateTime or Timestamp
    qty: float
    px: float
    fees: float
    kwargs: dict
            user attributes
    """
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


class SingleAssetPortfolio(object):
    """ Portfolio for a single asset (security)

    Parameters
    ----------
    pricer : PortfolioPricer
    trades : array of Trade(s), default is None
    """
    def __init__(self, pricer, trades=None):
        self.trades = trades or []
        self.pricer = pricer
        self._perf = None
        self._txn_frame = None
        self._ltd_txn_pl = None
        self._side = None
        self._long_only = None
        self._short_only = None

    def iter_txns(self):
        return iter_txns(self.trades)

    @property
    def side(self):
        """  return ('Long', 'Short', LongShort') depending on content of trades """
        if self._side is None:
            tf = self.txn_frame
            has_longs = (tf.txn_side == SIDE_BUY).any()
            has_shorts = (tf.txn_side == SIDE_SELL_SHORT).any()
            if has_longs and not has_shorts:
                self._side = 'Long'
            elif has_shorts and not has_longs:
                self._side = 'Short'
            else:
                self._side = 'LongShort'
        return self._side

    @property
    def txn_frame(self):
        """ Convert the portfolio Trades to a transaction DataFrame

        Notes
        -----
        DataFrame columns
        'dt' - date of transaction
        'txn_ts' - transaction timestamp
        'pid' - position id
        'tid' - trade id
        'txn_qty' - transaction quantity
        'txn_px' - transaction prices
        'txn_fees' - transaction fee
        'txn_premium' - total premium of transaction
        'open_val' - position open value
        'pos' - position quantity
        'txn_intent' - (INTENT_OPEN, INTENT_CLOSE, INTENT_INCREASE, INTENT_DECREASE)
        'txn_side' - (SIDE_BUY, SIDE_SELL, SIDE_SELL_SHORT, SIDE_COVER)
        """
        if self._txn_frame is None:
            rows = []
            pos = open_val = pid = 0
            for txn in self.iter_txns():
                # These values always get copied
                qty = txn.qty
                premium = self.pricer.get_premium(qty, txn.px)
                if pos == 0:  # Open position
                    side = qty > 0 and SIDE_BUY or SIDE_SELL_SHORT
                    open_val = premium
                    pid += 1
                    side = side
                    intent = INTENT_OPEN
                    pos = qty
                elif pos + qty == 0:  # close position
                    side = qty > 0 and SIDE_COVER or SIDE_SELL
                    open_val = 0
                    side = side
                    intent = INTENT_CLOSE
                    pos = 0
                elif is_increase(pos, qty):
                    side = txn.qty > 0 and SIDE_BUY or SIDE_SELL_SHORT
                    open_val += premium
                    pos += qty
                    intent = INTENT_INCREASE
                    side = side
                else:  # decrease - no worries about split since iterator takes care of it
                    side = txn.qty > 0 and SIDE_COVER or SIDE_SELL
                    open_val *= ((pos + qty) / pos)
                    pos += qty
                    intent = INTENT_DECREASE
                    side = side

                # Get rid of anything but the date
                dt = txn.ts.to_period('B').to_timestamp()
                rows.append([dt, txn.ts, pid, txn.tid, txn.qty, txn.px, txn.fees, premium, open_val, pos, intent, side])
            df = pd.DataFrame.from_records(rows, columns=['dt', 'txn_ts', 'pid', 'tid', 'txn_qty', 'txn_px', 'txn_fees',
                                                          'txn_premium', 'open_val', 'pos', 'txn_intent', 'txn_side'])
            df.index.name = 'seq'
            self._txn_frame = df
        return self._txn_frame

    @property
    def ltd_txn_pl(self):
        """ Using the pricer, return a DataFrame which contains the live-to-date profit and loss at a transaction level

        Notes
        -----
        DataFrame columns
        'dt' - transaction date
        'pos' - position quantity
        'pid' - position id
        'tid' - trade id
        'txn_qty' - transaction quantity
        'txn_px' - transaction prices
        'txn_fees' - transaction fee
        'txn_premium' - total premium of transaction
        'pos' - position quantity
        'txn_intent' - (INTENT_OPEN, INTENT_CLOSE, INTENT_INCREASE, INTENT_DECREASE)
        'txn_side' - (SIDE_BUY, SIDE_SELL, SIDE_SELL_SHORT, SIDE_COVER)
        'close' - closing price
        'open_val' - position open value
        'mkt_val' - market value of the position
        'tot_val' - total value of the positions
        'dvds' - ltd dividend amount
        'fees' - ltd fees
        'rpl_gross' - ltd realized pl gross (tot_val - open_val)
        'rpl' - ltd realized pl (rpl_gross + fees + dvds)
        'upl' - ltd unrealized pl (mkt_val + open_val)
        'pl' - ltd pl (upl + rpl)
        """
        if self._ltd_txn_pl is None:
            txn_frame = self.txn_frame
            mktdata = self.pricer.get_frame()
            if not isinstance(mktdata.index, pd.DatetimeIndex):
                mktdata.to_timestamp(freq='B')

            pl = txn_frame[['dt', 'pos', 'pid', 'tid', 'txn_intent', 'txn_side', 'txn_fees', 'txn_qty', 'txn_px',
                            'txn_premium', 'open_val']]

            # get the set of all txn dts and mkt data dts
            pl = pd.merge(pl, mktdata.reset_index(), how='outer', on='dt')
            # possibly bad assumption
            pl.sort(['dt', 'pid', 'tid'], inplace=1)
            pl.reset_index(inplace=1, drop=1)
            # check that all days can be priced
            missing = pl.dt[(pl.pid > 0) & pl.close.isnull()]
            if len(missing) > 0:
                msg = 'insifficient price data: {0} prices missing for dates {1}'
                mdates = ','.join([_.strftime('%Y-%m-%d') for _ in set(missing[:5])])
                mdates += (len(missing) > 5 and '...' or '')
                raise Exception(msg.format(len(missing), mdates))
            #
            # Data Prep
            #
            # Remove any dvds except for end of day
            pl.ix[(pl.dt == pl.dt.shift(-1)), 'dvds'] = 0
            #
            pl.open_val.ffill(inplace=1)
            pl.open_val.fillna(0, inplace=1)
            pl.pos.ffill(inplace=1)
            pl.pos.fillna(0, inplace=1)
            # pid is the only tricky one, copy only while position is open
            inpos = pl.txn_intent.notnull() | (pl.pos != 0)
            pids = np.where(inpos, pl.pid.ffill(), 0)
            pl['pid'] = pids
            # zero out missing txn lvl data
            pl.dvds.fillna(0, inplace=1)
            pl.tid.fillna(0, inplace=1)
            pl.txn_intent.fillna(0, inplace=1)
            pl.txn_side.fillna(0, inplace=1)
            pl.txn_fees.fillna(0, inplace=1)
            pl.txn_premium.fillna(0, inplace=1)
            # compute the pl
            pl['fees'] = pl.txn_fees.cumsum()
            pl['tot_val'] = pl.txn_premium.cumsum()
            pl['mkt_val'] = pl.mkt_val * pl.pos
            pl['dvds'] = (pl.dvds * pl.pos).cumsum()
            pl['rpl_gross'] = pl.tot_val - pl.open_val
            pl['rpl'] = pl.rpl_gross + pl.fees + pl.dvds
            pl['upl'] = pl.mkt_val + pl.open_val
            pl['pl'] = pl.upl + pl.rpl
            self._ltd_txn_pl = pl
        return self._ltd_txn_pl

    @property
    def ltd_pl(self):
        """ DataFrame which contains the live-to-date pl rolled up to a daily level

        Notes
        -----
        DataFrame columns
        'dt' - transaction date
        'pos' - end of day position quantity
        'open_val' - open value of the position
        'tot_val' - total value of the positions
        'rpl_gross' - ltd realized pl gross (tot_val - open_val)
        'fees' - ltd fees
        'dvds' - ltd dividend amount
        'rpl' - ltd realized pl (rpl_gross + fees + dvds)
        'upl' - ltd unrealized pl (mkt_val + open_val)
        'pl' - ltd pl (upl + rpl)
        """
        txnpl = self.ltd_txn_pl
        # resample
        pl = txnpl[['dt', 'pos', 'open_val', 'tot_val', 'rpl_gross', 'fees', 'dvds', 'rpl', 'upl', 'pl']]
        # when resampling, adds in missing busines days
        return pl.set_index('dt').resample('B', 'last', kind='period').dropna(how='all')

    def _to_dly_pl(self, pl):
        pl = pl.copy()
        ltd_cols = ['fees', 'tot_val', 'rpl_gross', 'dvds', 'rpl', 'upl', 'pl']
        ilocs = [pl.columns.get_loc(_) for _ in ltd_cols]
        diff = pl[ltd_cols].diff()
        pl.iloc[1:, ilocs] = diff.iloc[1:, range(len(ltd_cols))]
        return pl

    @property
    def dly_pl(self):
        """ DataFrame which contains the daily pl """
        return self._to_dly_pl(self.ltd_pl)

    @property
    def dly_txn_pl(self):
        """ DataFrame which contains the transaction level pl reported as daily numbers """
        return self._to_dly_pl(self.ltd_txn_pl)

    @property
    def dly_txn_roii(self):
        """ return the daily return on initial investment Series reported at transaction level"""
        # build pl path and get pct change to get compound dly returns
        dly = self.dly_txn_pl
        rets = pd.Series(0, index=dly.index, name='dly_roii')
        for pid, pframe in dly[['open_val', 'pid', 'pl']].groupby('pid'):
            if pid != 0:
                cost = abs(pframe.open_val.iloc[0])
                eod = cost + pframe.pl.cumsum()
                sod = eod.shift(1)
                sod.iloc[0] = cost
                pos_rets = eod / sod - 1.
                rets[pframe.index] = pos_rets
        return rets

    @property
    def dly_txn_days_open(self):
        """ return Series of the cumulative day count for the number of days a position is open """
        dly = self.dly_txn_pl
        days = pd.Series(0, index=dly.index, name='day')
        for pid, grp in dly.ix[dly.pid != 0].groupby('pid'):
            # ensure date is only counted once
            barr = grp.dt == grp.dt.shift(1)
            barr.iloc[0] = False
            tmp = (~barr).astype(float).cumsum()
            days.iloc[grp.index] = tmp
        return days

    @property
    def dly_roii(self):
        """ return the daily return on initial investment Series reported daily """
        index = self.ltd_txn_pl.dt
        dly = self.dly_txn_roii
        ltd = (1. + dly).cumprod()
        rets = pd.Series(ltd.values, index=index, name='dly_roii').resample('B', how='last',
                                                                            kind='period').dropna().pct_change()
        rets.iloc[0] = 0
        return rets

    def _filter_positions(self, pids):
        """ return portfolio without the specified pids """
        txn_frame = self.txn_frame
        mask = txn_frame.pid.isin(pids)
        keep_tids = txn_frame.tid[mask]
        trds = [t for t in self.trades if t.tid in keep_tids]
        port = SingleAssetPortfolio(self.pricer, trades=trds)
        # save time and pids by using existing calculations
        port._txn_frame = txn_frame.ix[mask]
        # pl is a little trickier - not worth it at the time
        return port

    @property
    def long_only(self):
        """ return portfolio with only the long positions """
        if self._long_only is None:
            if self.side == 'Long':
                self._long_only = self
            elif self.side == 'Short':
                raise Exception('cannot get Long view from Short portfolio')
            else:
                txn_frame = self.txn_frame
                longs = txn_frame.pid[txn_frame.txn_side == SIDE_BUY]
                self._long_only = self._filter_positions(longs)
        return self._long_only

    @property
    def short_only(self):
        """ return portfolio with only the short positions """
        if self._short_only is None:
            if self.side == 'Short':
                self._short_only = self
            elif self.side == 'Long':
                raise Exception('cannot get Short view from Long portfolio')
            else:
                txn_frame = self.txn_frame
                shorts = txn_frame.pid[txn_frame.txn_side == SIDE_SELL_SHORT]
                self._short_only = self._filter_positions(shorts)
        return self._short_only

    @property
    def position_frame(self):
        """ return DataFrame which provides a summary of the positions """
        vals = []
        for pid, txnpl in self.dly_txn_pl.groupby('pid'):
            if pid != 0:
                pos = PositionView(txnpl)
                vals.append([pos.pid, pos.is_long and 'Long' or 'Short', pos.open_dt, pos.close_dt, pos.open_qty,
                             pos.open_px, pos.close_px, pos.open_premium, pos.pl_min, pos.pl_max, pos.pl, pos.duration,
                             pos.ntxns, pos.roii, pos.is_closed and 'Closed' or 'Open'])
        cols = ['pid', 'side', 'open_dt', 'close_dt', 'open_qty', 'open_px', 'close_px', 'open_premium', 'pl_min',
                'pl_max', 'pl', 'duration', 'ntxns', 'roii', 'state']
        return pd.DataFrame.from_records(vals, columns=cols)

    def position_summary(self, win_is_zero=1, percentiles=None, yearly=0, ltd=1):
        #
        # TODO: This method is not 100% correct, it assumes if a position starts in a year then it is assigned to
        # that year. No splitting.
        #
        pframe = self.position_frame

        def _summarize(sub, is_total=0):
            # Rename
            ret_desc = sub.roii.describe(percentiles=percentiles).rename(lambda col: col == 'count' and 'cnt' or
                                                                                     col == 'mean' and 'roii_avg' or
                                                                                     'roii_%s' % col)
            dur_desc = sub.duration.describe(percentiles=percentiles).drop('count')
            dur_desc.rename(lambda col: col == 'mean' and 'duration_avg' or 'duration_%s' % col, inplace=True)
            return ret_desc.append(dur_desc).to_frame().T

        if yearly:
            years = self.ltd_pl.index.year
            tenors = [str(_) for _ in range(years.min(), years.max() + 1)]
            ltd and tenors.append('LTD')
            yridx = pframe.open_dt.apply(lambda c: c.year)
        elif ltd:
            tenors = ['LTD']
            yridx = None
        else:
            raise Exception('yearly and/or ltd need to be set to true')

        def _to_idx(yr, sd, wl):
            vals = [yr, sd, wl] if yearly else [sd, wl]
            names = ['tenor', 'side', 'winloss'] if yearly else ['side', 'winloss']
            return pd.MultiIndex.from_tuples([vals], names=names)

        pieces = []
        for tenor in tenors:
            view = pframe if tenor == 'LTD' else pframe.ix[yridx == int(tenor)]
            s = self.side
            winmask = view.pl >= 0 if win_is_zero else view.pl > 0
            wins = _summarize(view.ix[winmask])
            wins.insert(1, 'winpct', np.nan)
            wins.index = _to_idx(tenor, s, 'Wins')
            pieces.append(wins)
            losses = _summarize(view.ix[~winmask])
            losses.insert(1, 'winpct', np.nan)
            losses.index = _to_idx(tenor, s, 'Losses')
            pieces.append(losses)
            both = _summarize(view)
            both.insert(1, 'winpct', np.divide(wins.cnt.iloc[0], wins.cnt.iloc[0] + losses.cnt.iloc[0]))
            both.index = _to_idx(tenor, s, 'Total')
            pieces.append(both)
        return pd.concat(pieces)

    def position(self, pid):
        dly = self.dly_txn_pl
        tf = dly.ix[dly.pid == pid]
        return PositionView(tf)

    def iter_positions(self):
        dly_txn_pl = self.dly_txn_pl
        for pid in self.txn_frame.pid.unique():
            yield PositionView(dly_txn_pl.ix[dly_txn_pl.pid == pid])


class PositionView(object):
    def __init__(self, dly_txn_pl):
        self.dly_txn_pl = dly_txn_pl
        self.first = first = dly_txn_pl.iloc[0]
        self.last = last = dly_txn_pl.iloc[-1]
        self.is_closed = is_closed = last.txn_intent == INTENT_CLOSE
        self.is_long = is_long = first.txn_side == SIDE_BUY
        self.is_short = not is_long
        self.pid = first.pid
        self.open_dt = first.dt
        self.open_premium = first.txn_premium
        self.open_qty = first.txn_qty
        self.open_px = first.txn_px
        self.close_px = last.txn_px
        self.close_dt = last.dt
        cumpl = dly_txn_pl.pl.cumsum()
        self.pl = cumpl.iloc[-1]
        self.pl_min = cumpl.min()
        self.pl_max = cumpl.max()
        self.duration = len(dly_txn_pl.dt.drop_duplicates())
        self.ntxns = (dly_txn_pl.tid != 0).sum()

    @property
    def roii(self):
        '''
        :return: return on intial investment
        '''
        return self.pl / abs(self.open_premium)

    @property
    def dly_pl(self):
        return self.dly_txn_pl[['dt', 'pl']].set_index('dt').resample('B', how='sum', kind='period')['pl'].dropna()

    @property
    def dly_roii(self):
        '''Daily return on initial investment
        :return: the dly return series
        '''
        # TODO - handle increase/decrease logic
        cost = abs(self.open_premium)
        plpath = self.dly_pl.cumsum()
        rets = (cost + plpath).pct_change()
        rets.iloc[0] = 0
        return rets

    def __repr__(self):
        return '<%s(%s, %s, open=%s, close=%s, ret=%s)>' % (self.__class__.__name__, self.pid,
                                                            self.is_long and 'Long' or 'Short', self.open_dt,
                                                            self.close_dt, self.ret_on_capital)

