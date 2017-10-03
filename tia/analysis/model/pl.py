from collections import OrderedDict

import pandas as pd
import numpy as np

from tia.analysis.model.interface import TxnColumns as TC, MarketDataColumns as MC, PlColumns as PL, TxnPlColumns as TPL
from tia.analysis.perf import periods_in_year, guess_freq
from tia.util.decorator import lazy_property
from tia.util.fmt import new_dynamic_formatter


__all__ = ['ProfitAndLoss']


def _dly_to_ltd(frame, dly_cols):
    frame = frame.copy()
    ilocs = [frame.columns.get_loc(_) for _ in dly_cols]
    sums = frame[dly_cols].cumsum()
    # BUG when copying a single row, oddly
    if len(frame.index) == 1:
        frame.iloc[0, ilocs] = sums.iloc[0, list(range(len(dly_cols)))]
    else:
        frame.iloc[:, ilocs] = sums.iloc[:, list(range(len(dly_cols)))]
    return frame


def _ltd_to_dly(frame, ltd_cols):
    pl = frame.copy()
    ilocs = [frame.columns.get_loc(_) for _ in ltd_cols]
    diff = frame[ltd_cols].diff()
    # not sure why this is failing
    # pl.iloc[1:, ilocs] = diff.iloc[1:]
    for i, cidx in enumerate(ilocs):
        pl.iloc[1:, cidx] = diff.iloc[1:, i]
    return pl


class OpenAverageProfitAndLossCalculator(object):
    def compute(self, txns):
        """Compute the long/short live-to-date transaction level profit and loss. Uses an open average calculation"""
        txndata = txns.frame
        mktdata = txns.pricer.get_eod_frame()
        if not isinstance(mktdata.index, pd.DatetimeIndex):
            mktdata.to_timestamp(freq='B')

        # get the set of all txn dts and mkt data dts
        pl = pd.merge(txndata, mktdata.reset_index(), how='outer', on=TPL.DT)
        if pl[TC.PID].isnull().all():
            ltd_frame = pd.DataFrame(index=pl.index)
            ltd_frame[TPL.DT] = pl[PL.DT]
            ltd_frame[TPL.POS] = 0
            ltd_frame[TPL.PID] = 0
            ltd_frame[TPL.TID] = 0
            ltd_frame[TPL.TXN_QTY] = np.nan
            ltd_frame[TPL.TXN_PX] = np.nan
            ltd_frame[TPL.TXN_FEES] = 0
            ltd_frame[TPL.TXN_PREMIUM] = 0
            ltd_frame[TPL.TXN_INTENT] = 0
            ltd_frame[TPL.TXN_ACTION] = 0
            ltd_frame[TPL.CLOSE_PX] = pl[TPL.CLOSE_PX]
            ltd_frame[TPL.OPEN_VAL] = 0
            ltd_frame[TPL.MKT_VAL] = 0
            ltd_frame[TPL.TOT_VAL] = 0
            ltd_frame[TPL.DVDS] = 0
            ltd_frame[TPL.FEES] = 0
            ltd_frame[TPL.RPL_GROSS] = 0
            ltd_frame[TPL.RPL] = 0
            ltd_frame[TPL.UPL] = 0
            ltd_frame[TPL.PL] = 0
            return ltd_frame
        else:
            pl.sort([TC.DT, TC.PID, TC.TID], inplace=1)
            pl.reset_index(inplace=1, drop=1)
            # check that all days can be priced
            has_position = pl[TC.PID] > 0
            missing_pxs = pl[MC.CLOSE].isnull()
            missing = pl[TC.DT][has_position & missing_pxs]
            if len(missing) > 0:
                msg = 'insufficient price data: {0} prices missing for dates {1}'
                mdates = ','.join([_.strftime('%Y-%m-%d') for _ in set(missing[:5])])
                mdates += (len(missing) > 5 and '...' or '')
                raise Exception(msg.format(len(missing), mdates))

            # Now there is a row for every timestamp. Now compute the pl and fill in where missing data should be
            cols = [TC.DT, TC.POS, TC.PID, TC.TID, TC.INTENT, TC.ACTION, TC.FEES, TC.QTY, TC.PX, TC.PREMIUM,
                    TC.OPEN_VAL]
            dts, pos_qtys, pids, tids, intents, sides, txn_fees, txn_qtys, txn_pxs, premiums, open_vals = [pl[c] for c
                                                                                                           in
                                                                                                           cols]

            dvds, closing_pxs, mkt_vals = [pl[c] for c in [MC.DVDS, MC.CLOSE, MC.MKT_VAL]]
            # Ensure only end of day is kept for dividends (join will match dvd to any transaction during day
            dvds = dvds.where(dts != dts.shift(-1), 0)
            # fill in pl dates
            open_vals.ffill(inplace=1)
            open_vals.fillna(0, inplace=1)
            pos_qtys.ffill(inplace=1)
            pos_qtys.fillna(0, inplace=1)
            # pid is the only tricky one, copy only while position is open
            inpos = intents.notnull() | (pos_qtys != 0)
            pids = np.where(inpos, pids.ffill(), 0)
            pl['pid'] = pids.astype(int)
            # Zero fill missing
            dvds.fillna(0, inplace=1)
            tids.fillna(0, inplace=1)
            tids = tids.astype(int)
            intents.fillna(0, inplace=1)
            intents = intents.astype(int)
            sides.fillna(0, inplace=1)
            sides = sides.astype(int)
            txn_fees.fillna(0, inplace=1)
            premiums.fillna(0, inplace=1)
            # LTD p/l calculation
            fees = txn_fees.cumsum()
            total_vals = premiums.cumsum()
            mkt_vals *= pos_qtys
            dvds = (dvds * pos_qtys).cumsum()
            rpl_gross = total_vals - open_vals
            rpl = rpl_gross + fees + dvds
            upl = mkt_vals + open_vals
            tpl = upl + rpl
            # build the result
            data = OrderedDict()
            data[TPL.DT] = dts
            data[TPL.POS] = pos_qtys
            data[TPL.PID] = pids
            data[TPL.TID] = tids
            data[TPL.TXN_QTY] = txn_qtys
            data[TPL.TXN_PX] = txn_pxs
            data[TPL.TXN_FEES] = txn_fees
            data[TPL.TXN_PREMIUM] = premiums
            data[TPL.TXN_INTENT] = intents
            data[TPL.TXN_ACTION] = sides
            data[TPL.CLOSE_PX] = closing_pxs
            data[TPL.OPEN_VAL] = open_vals
            data[TPL.MKT_VAL] = mkt_vals
            data[TPL.TOT_VAL] = total_vals
            data[TPL.DVDS] = dvds
            data[TPL.FEES] = fees
            data[TPL.RPL_GROSS] = rpl_gross
            data[TPL.RPL] = rpl
            data[TPL.UPL] = upl
            data[TPL.PL] = tpl
            ltd_frame = pd.DataFrame(data, columns=list(data.keys()))
            return ltd_frame


class TxnProfitAndLossDetails(object):
    def __init__(self, txns=None, frame=None, ltd_frame=None):
        """
        :param txns: Txns object
        """
        if txns is None and frame is None and ltd_frame is None:
            raise ValueError('Either {txns, frame, ltd_frame} must be defined')

        self.txns = txns
        self._frame = frame
        self._ltd_frame = ltd_frame
        self.ltd_cols = [TPL.FEES, TPL.TOT_VAL, TPL.RPL_GROSS, TPL.DVDS, TPL.RPL, TPL.RPL, TPL.UPL, TPL.PL]

    @property
    def ltd_frame(self):
        if self._ltd_frame is None:
            if self._frame is not None:
                self._ltd_frame = _dly_to_ltd(self._frame, self.ltd_cols)
            elif self.txns is not None:
                self._ltd_frame = OpenAverageProfitAndLossCalculator().compute(self.txns)
            else:
                raise Exception('either txns or pl frame must be defined')
        return self._ltd_frame

    @property
    def frame(self):
        if self._frame is None:
            ltd = self.ltd_frame
            self._frame = _ltd_to_dly(ltd, self.ltd_cols)
        return self._frame

    def asfreq(self, freq):
        frame = self.frame
        pl = frame[PL.ALL].set_index(PL.DT)
        if freq == 'B':
            resampled = pl.groupby(pl.index.date).apply(lambda f: f.sum())
            resampled.index = pd.DatetimeIndex([i for i in resampled.index])
            return ProfitAndLossDetails(resampled)
        else:
            resampled = pl.resample(freq, how='sum')
            return ProfitAndLossDetails(resampled)

    # -----------------------------------------------------------
    # Resampled data
    dly = lazy_property(lambda self: self.asfreq('B'), 'dly')
    weekly = lazy_property(lambda self: self.asfreq('W'), 'weekly')
    monthly = lazy_property(lambda self: self.asfreq('M'), 'monthly')
    quarterly = lazy_property(lambda self: self.asfreq('Q'), 'quarterly')
    annual = lazy_property(lambda self: self.asfreq('A'), 'annual')

    def get_pid_mask(self, pid):
        return self.frame[TPL.PID] == pid

    def truncate(self, before=None, after=None, pid=None):
        if before is None and after is None and pid is None:
            return self
        elif before or after:
            sub = self.frame.truncate(before, after)
            return TxnProfitAndLossDetails(frame=sub)
        else:
            mask = self.get_pid_mask(pid)
            frame = self.frame
            sub = frame.ix[mask.values]
            return TxnProfitAndLossDetails(frame=sub)

    def iter_by_year(self):
        for key, grp in self.frame.groupby(self.frame[TPL.DT].dt.year):
            yield key, TxnProfitAndLossDetails(frame=grp)

    def subset(self, txns):
        """To perform a subset it is not possible to reuse the frame since it is LTD, so we convert to daily then
        compute ltd from daily
        :param txns: the update Txns object
        :return:
        """
        result = TxnProfitAndLossDetails(txns)
        # TODO - add reusing calcs. Issue is when removing PIDs, then could be multiple entries per dt
        # use daily txn, clear all values where != pid
        # determine which Timestamp columns can be removed as an old position may have multiple txns on same day
        # recreate ltd from dly
        # Need to take care if a dvd occurs at end of day
        return result


class ProfitAndLossDetails(object):
    def __init__(self, frame=None, ltd_frame=None):
        self._frame = frame
        self._ltd_frame = ltd_frame

    @property
    def ltd_frame(self):
        ltd = self._ltd_frame
        if ltd is None:
            if self._frame is None:
                raise Exception('Both frame and ltd frame are None. At least one must be defined.')
            self._ltd_frame = ltd = _dly_to_ltd(self._frame, PL.LTDS)
        return ltd

    @property
    def frame(self):
        obs = self._frame
        if obs is None:
            if self._ltd_frame is None:
                raise Exception('Both frame and ltd frames are None. At least one must be defined.')
            self._frame = obs = _ltd_to_dly(self._ltd_frame, PL.LTDS)
        return obs

    def rolling_frame(self, n):
        return pd.rolling_sum(self.frame, n)

    def asfreq(self, freq):
        """Resample the p&l at the specified frequency

        :param freq:
        :return: Pl object
        """
        frame = self.frame
        if freq == 'B':
            resampled = frame.groupby(frame.index.date).apply(lambda f: f.sum())
            resampled.index = pd.DatetimeIndex([i for i in resampled.index])
            return ProfitAndLossDetails(resampled)
        else:
            resampled = frame.resample(freq, how='sum')
            return ProfitAndLossDetails(resampled)

    @lazy_property
    def drawdown_info(self):
        dd = self.drawdowns.to_frame()
        last = dd.index[-1]
        dd.columns = ['vals']
        dd['nonzero'] = (dd.vals != 0).astype(int)
        dd['gid'] = (dd.nonzero.shift(1) != dd.nonzero).astype(int).cumsum()
        ixs = dd.reset_index().groupby(['nonzero', 'gid'])[dd.index.name or 'index'].apply(lambda x: np.array(x))
        rows = []
        if 1 in ixs:
            for ix in ixs[1]:
                sub = dd.ix[ix]
                # need to get t+1 since actually draw down ends on the 0 value
                end = dd.index[dd.index.get_loc(sub.index[-1]) + (last != sub.index[-1] and 1 or 0)]
                rows.append([sub.index[0], end, sub.vals.min(), sub.vals.idxmin()])
        f = pd.DataFrame.from_records(rows, columns=['dd start', 'dd end', 'maxdd', 'maxdd dt'])
        f['days'] = (f['dd end'] - f['dd start']).astype('timedelta64[D]')
        return f

    @lazy_property
    def drawdowns(self):
        ltd = self.ltd_frame.pl
        maxpl = pd.expanding_max(ltd)
        maxpl[maxpl < 0] = 0
        dd = ltd - maxpl
        return dd

    # scalar data
    cnt = property(lambda self: self.frame.pl.notnull().astype(int).sum())
    mean = lazy_property(lambda self: self.frame.pl.mean(), 'mean')
    avg = mean
    std = lazy_property(lambda self: self.frame.pl.std(), 'std')
    std_ann = lazy_property(lambda self: np.sqrt(periods_in_year(self.frame.pl)) * self.std, 'std_ann')
    maxdd = lazy_property(lambda self: self.drawdown_info['maxdd'].min(), 'maxdd')
    dd_avg = lazy_property(lambda self: self.drawdown_info['maxdd'].mean(), 'dd_avg')
    min = property(lambda self: self.frame.pl.min())
    max = property(lambda self: self.frame.pl.max())

    @lazy_property
    def maxdd_dt(self):
        if self.drawdown_info.empty:
            return None
        else:
            return self.drawdown_info['maxdd dt'].ix[self.drawdown_info['maxdd'].idxmin()]

    @lazy_property
    def summary(self):
        d = OrderedDict()
        d['avg'] = self.avg
        d['std'] = self.std
        d['maxdd'] = self.maxdd
        d['maxdd dt'] = self.maxdd_dt
        d['dd avg'] = self.dd_avg
        d['cnt'] = self.cnt
        return pd.Series(d, name=self.frame.index.freq or guess_freq(self.frame.index))

    def _repr_html_(self):
        from tia.util.fmt import new_dynamic_formatter

        fmt = new_dynamic_formatter(method='row', precision=2, pcts=1, trunc_dot_zeros=1, parens=1)
        return fmt(self.summary.to_frame())._repr_html_()

    def plot_ltd(self, ax=None, style='k', label='ltd', show_dd=1, guess_xlabel=1, title=True):
        ltd = self.ltd_frame.pl
        ax = ltd.plot(ax=ax, style=style, label=label)
        if show_dd:
            dd = self.drawdowns
            dd.plot(style='r', label='drawdowns', alpha=.5)
            ax.fill_between(dd.index, 0, dd.values, facecolor='red', alpha=.25)
            fmt = lambda x: x
            # guess the formatter
            if guess_xlabel:
                from tia.util.fmt import guess_formatter
                from tia.util.mplot import AxesFormat

                fmt = guess_formatter(ltd.abs().max(), precision=1)
                AxesFormat().Y.apply_format(fmt).apply(ax)
                ax.legend(loc='upper left', prop={'size': 12})

            # show the actualy date and value
            mdt, mdd = self.maxdd_dt, self.maxdd
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.25)
            try:
                dtstr = '{0}'.format(mdt.to_period())
            except:
                # assume daily
                dtstr = '{0}'.format(hasattr(mdt, 'date') and mdt.date() or mdt)
            ax.text(mdt, dd[mdt], "{1} \n {0}".format(fmt(mdd), dtstr).strip(), ha="center", va="top", size=8,
                    bbox=bbox_props)

        if title is True:
            df = new_dynamic_formatter(precision=1, parens=False, trunc_dot_zeros=True)
            total = df(ltd.iloc[-1])
            vol = df(self.std)
            mdd = df(self.maxdd)
            title = 'pnl %s     vol %s     maxdd %s' % (total, vol, mdd)

        title and ax.set_title(title, fontdict=dict(fontsize=10, fontweight='bold'))
        return ax

    def truncate(self, before=None, after=None):
        if before is None and after is None:
            return self
        else:
            sub = self.frame.truncate(before, after)
            return ProfitAndLossDetails(frame=sub)


class ProfitAndLoss(object):
    def __init__(self, dly_details):
        self._dly_details = dly_details

    dly_details = property(lambda self: self._dly_details)
    dly_frame = property(lambda self: self.dly_details.frame)
    ltd_dly_frame = property(lambda self: self.dly_details.ltd_frame)
    dly = property(lambda self: self.dly_frame.pl)
    ltd_dly = property(lambda self: self.ltd_dly_frame.pl)

    weekly_details = lazy_property(lambda self: self.txn_details.weekly, 'weekly_details')
    weekly_frame = property(lambda self: self.weekly_details.frame)
    ltd_weekly_frame = property(lambda self: self.weekly_details.ltd_frame)
    weekly = property(lambda self: self.weekly_frame.pl)
    ltd_weekly = property(lambda self: self.ltd_weekly_frame.pl)

    monthly_details = lazy_property(lambda self: self.txn_details.monthly, 'monthly_details')
    monthly_frame = property(lambda self: self.monthly_details.frame)
    ltd_monthly_frame = property(lambda self: self.monthly_details.ltd_frame)
    monthly = property(lambda self: self.monthly_frame.pl)
    ltd_monthly = property(lambda self: self.ltd_monthly_frame.pl)

    quarterly_details = lazy_property(lambda self: self.txn_details.quarterly, 'quarterly_details')
    quarterly_frame = property(lambda self: self.quarterly_details.frame)
    ltd_quarterly_frame = property(lambda self: self.quarterly_details.ltd_frame)
    quarterly = property(lambda self: self.quarterly_frame.pl)
    ltd_quarterly = property(lambda self: self.ltd_quarterly_frame.pl)

    annual_details = lazy_property(lambda self: self.txn_details.annual, 'annual_details')
    annual_frame = property(lambda self: self.annual_details.frame)
    ltd_annual_frame = property(lambda self: self.annual_details.ltd_frame)
    annual = property(lambda self: self.annual_frame.pl)
    ltd_annual = property(lambda self: self.ltd_annual_frame.pl)

    def iter_by_year(self):
        for yr, details in self.dly_details.iter_by_year():
            yield yr, ProfitAndLoss(details)

    def truncate(self, before=None, after=None, pid=None):
        if before is None and after is None and pid is None:
            return self
        else:
            details = self.dly_details.truncate(before, after)
            return ProfitAndLoss(details)

    def report_by_year(self, summary_fct=None, years=None, ltd=1, prior_n_yrs=None, first_n_yrs=None, ranges=None,
                       bm_rets=None):
        """Summarize the profit and loss by year
        :param summary_fct: function(ProfitAndLoss) and returns a dict or Series
        :param years: int, array, boolean or None. If boolean and False, then show no years. If int or array
                      show only those years, else show all years if None
        :param ltd: include live to date summary
        :param prior_n_years: integer or list. Include summary for N years of return data prior to end date
        :param first_n_years: integer or list. Include summary for N years of return data after start date
        :param ranges: list of ranges. The range consists of a year start and year end
        :param dm_dly_rets: daily return series for the benchmark for beta/alpha calcs
        :return: DataFrame
        """
        if years and np.isscalar(years):
            years = [years]

        if summary_fct is None:
            def summary_fct(pl):
                monthly = pl.monthly_details
                dly = pl.dly_details
                data = OrderedDict()
                data['mpl avg'] = monthly.mean
                data['mpl std ann'] = monthly.std_ann
                data['maxdd'] = dly.maxdd
                data['maxdd dt'] = dly.maxdd_dt
                data['avg dd'] = dly.dd_avg
                data['best month'] = monthly.max
                data['worst month'] = monthly.min
                data['best day'] = dly.max
                data['worst day'] = dly.min
                data['nmonths'] = monthly.cnt
                return data

        results = OrderedDict()

        if years is not False:
            for yr, pandl in self.iter_by_year():
                if years is None or yr in years:
                    results[yr] = summary_fct(pandl)

        # First n years
        if first_n_yrs:
            first_n_yrs = first_n_yrs if not np.isscalar(first_n_yrs) else [first_n_yrs]
            for first in first_n_yrs:
                after = '12/31/%s' % (self.dly.index[0].year + first)
                firstN = self.truncate(after=after)
                results['first {0}yrs'.format(first)] = summary_fct(firstN)

        # Ranges
        if ranges:
            for range in ranges:
                yr_start, yr_end = range
                rng_rets = self.truncate('1/1/%s' % yr_start, '12/31/%s' % yr_end)
                results['{0}-{1}'.format(yr_start, yr_end)] = summary_fct(rng_rets)

        # Prior n years
        if prior_n_yrs:
            prior_n_yrs = prior_n_yrs if not np.isscalar(prior_n_yrs) else [prior_n_yrs]
            for prior in prior_n_yrs:
                before = '1/1/%s' % (self.dly.index[-1].year - prior)
                priorN = self.truncate(before)
                results['past {0}yrs'.format(prior)] = summary_fct(priorN)

        # LTD
        if ltd:
            results['ltd'] = summary_fct(self)

        return pd.DataFrame(results, index=list(results.values())[0].keys()).T


class TxnProfitAndLoss(ProfitAndLoss):
    def __init__(self, txns=None, txnpl_details=None):
        if txns is None and txnpl_details is None:
            raise ValueError('txns or txn_details must be specified')
        self.txns = txns
        self._txn_details = txnpl_details
        # Don't set the attribute, wany lazy property to be called
        #ProfitAndLoss.__init__(self, None)

    @property
    def txn_details(self):
        if self._txn_details is None:
            self._txn_details = TxnProfitAndLossDetails(self.txns)
        return self._txn_details

    txn_frame = property(lambda self: self.txn_details.frame)
    ltd_txn_frame = property(lambda self: self.txn_details.ltd_frame)
    txn = property(lambda self: self.txn_frame.set_index(PL.DT).pl)
    ltd_txn = property(lambda self: self.ltd_txn_frame.set_index(PL.DT).pl)

    dly_details = lazy_property(lambda self: self.txn_details.dly, 'dly_details')

    def truncate(self, before=None, after=None, pid=None):
        if before is None and after is None and pid is None:
            return self
        else:
            details = self.txn_details.truncate(before, after, pid)
            return TxnProfitAndLoss(txnpl_details=details)

    def get_pid_mask(self, pid):
        return self.txn_details.get_pid_mask(pid)

