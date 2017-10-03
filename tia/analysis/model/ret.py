from collections import OrderedDict

import pandas as pd
import numpy as np

from tia.util.decorator import lazy_property
from tia.analysis.model.interface import TxnPlColumns as TPL
from tia.analysis.perf import drawdown_info, drawdowns, guess_freq, downside_deviation, periodicity
from tia.analysis.plots import plot_return_on_dollar
from tia.util.mplot import AxesFormat
from tia.util.fmt import PercentFormatter, new_percent_formatter, new_float_formatter


__all__ = ['RoiiRetCalculator', 'AumRetCalculator', 'FixedAumRetCalculator', 'CumulativeRets', 'Performance']


def return_on_initial_capital(capital, period_pl, leverage=None):
    """Return the daily return series based on the capital"""
    if capital <= 0:
        raise ValueError('cost must be a positive number not %s' % capital)
    leverage = leverage or 1.
    eod = capital + (leverage * period_pl.cumsum())
    ltd_rets = (eod / capital) - 1.
    dly_rets = ltd_rets
    dly_rets.iloc[1:] = (1. + ltd_rets).pct_change().iloc[1:]
    return dly_rets


class RetCalculator(object):
    def compute(self, txns):
        raise NotImplementedError()


class RoiiRetCalculator(RetCalculator):
    def __init__(self, leverage=None):
        """
        :param leverage: {None, scalar, Series}, number to scale the position returns
        :return:
        """
        get_lev = None
        if leverage is None:
            pass
        elif np.isscalar(leverage):
            if leverage <= 0:
                raise ValueError('leverage must be a positive non-zero number, not %s' % leverage)
            else:
                get_lev = lambda ts: leverage
        elif isinstance(leverage, pd.Series):
            get_lev = lambda ts: leverage.asof(ts)
        else:
            raise ValueError(
                'leverage must be {None, positive scalar, Datetime/Period indexed Series} not %s' % type(leverage))

        self.leverage = leverage
        self.get_lev = get_lev

    def compute(self, txns):
        txnpl = txns.pl.txn_frame
        txnrets = pd.Series(0, index=txnpl.index, name='ret')
        get_lev = self.get_lev
        for pid, pframe in txnpl[[TPL.OPEN_VAL, TPL.PID, TPL.PL, TPL.DT]].groupby(TPL.PID):
            if pid != 0:
                cost = abs(pframe[TPL.OPEN_VAL].iloc[0])
                ppl = pframe[TPL.PL]
                lev = None if get_lev is None else get_lev(pframe[TPL.DT].iloc[0])
                ret = return_on_initial_capital(cost, ppl, lev)
                txnrets[ppl.index] = ret

        txnrets.index = txnpl[TPL.DT]
        crets = CumulativeRets(txnrets)
        return Performance(crets)


class FixedAumRetCalculator(RetCalculator):
    def __init__(self, aum, reset_freq='M'):
        self.aum = aum
        self.reset_freq = reset_freq
        # capture what cash flows would be needed on reset date to reset the aum
        self.external_cash_flows = None

    def compute(self, txns):
        ltd = txns.pl.ltd_txn
        grouper = pd.TimeGrouper(self.reset_freq)
        period_rets = pd.Series(np.nan, index=ltd.index)
        aum = self.aum
        at = 0
        cf = OrderedDict()
        for key, grp in ltd.groupby(grouper):
            if grp.empty:
                continue
            eod = aum + grp
            sod = eod.shift(1)
            sod.iloc[0] = aum
            period_rets.iloc[at:at + len(grp.index)] = eod / sod - 1.
            at += len(grp.index)
            # get aum back to fixed amount
            cf[key] = eod.iloc[-1] - aum
        self.external_cash_flows = pd.Series(cf)
        crets = CumulativeRets(period_rets)
        return Performance(crets)


class AumRetCalculator(RetCalculator):
    def __init__(self, starting_aum, freq='M'):
        self.starting_aum = starting_aum
        self.freq = freq
        self.txn_aum = None

    def compute(self, txns):
        ltd = txns.pl.ltd_txn
        grouper = pd.TimeGrouper(self.freq)
        period_rets = pd.Series(np.nan, index=ltd.index)
        self.txn_aum = txn_aum = pd.Series(np.nan, index=ltd.index)
        sop = self.starting_aum
        at = 0
        for key, grp in ltd.groupby(grouper):
            if grp.empty:
                continue
            eod = sop + grp
            sod = eod.shift(1)
            sod.iloc[0] = sop
            period_rets.iloc[at:at + len(grp.index)] = eod / sod - 1.
            txn_aum.iloc[at:at + len(grp.index)] = sod
            at += len(grp.index)
            sop = eod.iloc[-1]
        crets = CumulativeRets(period_rets)
        return Performance(crets)


class CumulativeRets(object):
    def __init__(self, rets=None, ltd_rets=None):
        if rets is None and ltd_rets is None:
            raise ValueError('rets or ltd_rets must be specified')

        if rets is None:
            if ltd_rets.empty:
                rets = ltd_rets
            else:
                rets = (1. + ltd_rets).pct_change()
                rets.iloc[0] = ltd_rets.iloc[0]

        if ltd_rets is None:
            if rets.empty:
                ltd_rets = rets
            else:
                ltd_rets = (1. + rets).cumprod() - 1.

        self.rets = rets
        self.ltd_rets = ltd_rets

    pds_per_year = property(lambda self: periodicity(self.rets))

    def asfreq(self, freq):
        other_pds_per_year = periodicity(freq)
        if self.pds_per_year < other_pds_per_year:
            msg = 'Cannot downsample returns. Cannot convert from %s periods/year to %s'
            raise ValueError(msg % (self.pds_per_year, other_pds_per_year))

        if freq == 'B':
            rets = (1. + self.rets).groupby(self.rets.index.date).apply(lambda s: s.prod()) - 1.
            # If you do not do this, it will be an object index
            rets.index = pd.DatetimeIndex([i for i in rets.index])
            return CumulativeRets(rets)
        else:
            rets = (1. + self.rets).resample(freq, how='prod') - 1.
            return CumulativeRets(rets)

    # -----------------------------------------------------------
    # Resampled data
    dly = lazy_property(lambda self: self.asfreq('B'), 'dly')
    weekly = lazy_property(lambda self: self.asfreq('W'), 'weekly')
    monthly = lazy_property(lambda self: self.asfreq('M'), 'monthly')
    quarterly = lazy_property(lambda self: self.asfreq('Q'), 'quarterly')
    annual = lazy_property(lambda self: self.asfreq('A'), 'annual')

    # -----------------------------------------------------------
    # Basic Metrics
    @lazy_property
    def ltd_rets_ann(self):
        return (1. + self.ltd_rets) ** (self.pds_per_year / pd.expanding_count(self.rets)) - 1.

    cnt = property(lambda self: self.rets.notnull().astype(int).sum())
    mean = lazy_property(lambda self: self.rets.mean(), 'avg')
    mean_ann = lazy_property(lambda self: self.mean * self.pds_per_year, 'avg_ann')
    ltd = lazy_property(lambda self: self.ltd_rets.iloc[-1], name='ltd')
    ltd_ann = lazy_property(lambda self: self.ltd_rets_ann.iloc[-1], name='ltd_ann')
    std = lazy_property(lambda self: self.rets.std(), 'std')
    std_ann = lazy_property(lambda self: self.std * np.sqrt(self.pds_per_year), 'std_ann')
    drawdown_info = lazy_property(lambda self: drawdown_info(self.rets), 'drawdown_info')
    drawdowns = lazy_property(lambda self: drawdowns(self.rets), 'drawdowns')
    maxdd = lazy_property(lambda self: self.drawdown_info['maxdd'].min(), 'maxdd')
    dd_avg = lazy_property(lambda self: self.drawdown_info['maxdd'].mean(), 'dd_avg')
    kurtosis = lazy_property(lambda self: self.rets.kurtosis(), 'kurtosis')
    skew = lazy_property(lambda self: self.rets.skew(), 'skew')

    sharpe_ann = lazy_property(lambda self: np.divide(self.ltd_ann, self.std_ann), 'sharpe_ann')
    downside_deviation = lazy_property(lambda self: downside_deviation(self.rets, mar=0, full=0, ann=1),
                                       'downside_deviation')
    sortino = lazy_property(lambda self: self.ltd_ann / self.downside_deviation, 'sortino')

    @lazy_property
    def maxdd_dt(self):
        ddinfo = self.drawdown_info
        if ddinfo.empty:
            return None
        else:
            return self.drawdown_info['maxdd dt'].ix[self.drawdown_info['maxdd'].idxmin()]

    # -----------------------------------------------------------
    # Expanding metrics
    expanding_mean = property(lambda self: pd.expanding_mean(self.rets), 'expanding_avg')
    expanding_mean_ann = property(lambda self: self.expanding_mean * self.pds_per_year, 'expanding_avg_ann')
    expanding_std = lazy_property(lambda self: pd.expanding_std(self.rets), 'expanding_std')
    expanding_std_ann = lazy_property(lambda self: self.expanding_std * np.sqrt(self.pds_per_year), 'expanding_std_ann')
    expanding_sharpe_ann = property(lambda self: np.divide(self.ltd_rets_ann, self.expanding_std_ann))

    # -----------------------------------------------------------
    # Rolling metrics
    rolling_mean = property(lambda self: pd.rolling_mean(self.rets), 'rolling_avg')
    rolling_mean_ann = property(lambda self: self.rolling_mean * self.pds_per_year, 'rolling_avg_ann')

    def rolling_ltd_rets(self, n):
        return pd.rolling_apply(self.rets, n, lambda s: (1. + s).prod() - 1.)

    def rolling_ltd_rets_ann(self, n):
        tot = self.rolling_ltd_rets(n)
        return tot ** (self.pds_per_year / n)

    def rolling_std(self, n):
        return pd.rolling_std(self.rets, n)

    def rolling_std_ann(self, n):
        return self.rolling_std(n) * np.sqrt(self.pds_per_year)

    def rolling_sharpe_ann(self, n):
        return self.rolling_ltd_rets_ann(n) / self.rolling_std_ann(n)

    def iter_by_year(self):
        """Split the return objects by year and iterate"""
        for key, grp in self.rets.groupby(lambda x: x.year):
            yield key, CumulativeRets(rets=grp)

    def truncate(self, before=None, after=None):
        rets = self.rets.truncate(before=before, after=after)
        return CumulativeRets(rets=rets)

    @lazy_property
    def summary(self):
        d = OrderedDict()
        d['ltd'] = self.ltd
        d['ltd ann'] = self.ltd_ann
        d['mean'] = self.mean
        d['mean ann'] = self.mean_ann
        d['std'] = self.std
        d['std ann'] = self.std_ann
        d['sharpe ann'] = self.sharpe_ann
        d['sortino'] = self.sortino
        d['maxdd'] = self.maxdd
        d['maxdd dt'] = self.maxdd_dt
        d['dd avg'] = self.dd_avg
        d['cnt'] = self.cnt
        return pd.Series(d, name=self.rets.index.freq or guess_freq(self.rets.index))

    def _repr_html_(self):
        from tia.util.fmt import new_dynamic_formatter

        fmt = new_dynamic_formatter(method='row', precision=2, pcts=1, trunc_dot_zeros=1, parens=1)
        df = self.summary.to_frame()
        return fmt(df)._repr_html_()

    def get_alpha_beta(self, bm_rets):
        if isinstance(bm_rets, pd.Series):
            bm = CumulativeRets(bm_rets)
        elif isinstance(bm_rets, CumulativeRets):
            bm = bm_rets
        else:
            raise ValueError('bm_rets must be series or CumulativeRetPerformace not %s' % (type(bm_rets)))

        bm_freq = guess_freq(bm_rets)
        if self.pds_per_year != bm.pds_per_year:
            tgt = {'B': 'dly', 'W': 'weekly', 'M': 'monthly', 'Q': 'quarterly', 'A': 'annual'}.get(bm_freq, None)
            if tgt is None:
                raise ValueError('No mapping for handling benchmark with frequency: %s' % bm_freq)
            tmp = getattr(self, tgt)
            y = tmp.rets
            y_ann = tmp.ltd_ann
        else:
            y = self.rets
            y_ann = self.ltd_ann

        x = bm.rets.truncate(y.index[0], y.index[-1])
        x_ann = bm.ltd_ann

        model = pd.ols(x=x, y=y)
        beta = model.beta[0]
        alpha = y_ann - beta * x_ann
        return pd.Series({'alpha': alpha, 'beta': beta}, name=bm_freq)

    def plot_ltd(self, ax=None, style='k', label='ltd', show_dd=1, title=True, legend=1):
        ltd = self.ltd_rets
        ax = ltd.plot(ax=ax, style=style, label=label)
        if show_dd:
            dd = self.drawdowns
            dd.plot(style='r', label='drawdowns', alpha=.5, ax=ax)
            ax.fill_between(dd.index, 0, dd.values, facecolor='red', alpha=.25)
            fmt = PercentFormatter

            AxesFormat().Y.percent().X.label("").apply(ax)
            legend and ax.legend(loc='upper left', prop={'size': 12})

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
            pf = new_percent_formatter(1, parens=False, trunc_dot_zeros=True)
            ff = new_float_formatter(precision=1, parens=False, trunc_dot_zeros=True)
            total = pf(self.ltd_ann)
            vol = pf(self.std_ann)
            sh = ff(self.sharpe_ann)
            mdd = pf(self.maxdd)
            title = 'ret$\mathregular{_{ann}}$ %s     vol$\mathregular{_{ann}}$ %s     sharpe %s     maxdd %s' % (
            total, vol, sh, mdd)

        title and ax.set_title(title, fontdict=dict(fontsize=10, fontweight='bold'))
        return ax

    def plot_ret_on_dollar(self, title=None, show_maxdd=1, figsize=None, ax=None, append=0, label=None, **plot_args):
        plot_return_on_dollar(self.rets, title=title, show_maxdd=show_maxdd, figsize=figsize, ax=ax, append=append,
                              label=label, **plot_args)

    def plot_hist(self, ax=None, **histplot_kwargs):
        pf = new_percent_formatter(precision=1, parens=False, trunc_dot_zeros=1)
        ff = new_float_formatter(precision=1, parens=False, trunc_dot_zeros=1)

        ax = self.rets.hist(ax=ax, **histplot_kwargs)
        AxesFormat().X.percent(1).apply(ax)
        m, s, sk, ku = pf(self.mean), pf(self.std), ff(self.skew), ff(self.kurtosis)
        txt = '$\mathregular{\mu}$=%s   $\mathregular{\sigma}$=%s   skew=%s   kurt=%s' % (m, s, sk, ku)
        bbox = dict(facecolor='white', alpha=0.5)
        ax.text(0, 1, txt, fontdict={'fontweight': 'bold'}, bbox=bbox, ha='left', va='top', transform=ax.transAxes)
        return ax

    def filter(self, mask, keep_ltd=0):
        if isinstance(mask, pd.Series):
            mask = mask.values
        rets = self.rets.ix[mask]
        ltd = None
        if keep_ltd:
            ltd = self.ltd_rets.ix[mask]
        return CumulativeRets(rets=rets, ltd_rets=ltd)


class Performance(object):
    def __init__(self, txn_rets):
        if isinstance(txn_rets, pd.Series):
            txn_rets = CumulativeRets(txn_rets)
        self.txn_details = txn_rets

    txn = property(lambda self: self.txn_details.rets)
    ltd_txn = property(lambda self: self.txn_details.ltd_rets)

    dly_details = lazy_property(lambda self: self.txn_details.dly, 'dly_details')
    dly = property(lambda self: self.dly_details.rets)
    ltd_dly = property(lambda self: self.dly_details.ltd_rets)
    ltd_dly_ann = property(lambda self: self.dly_details.ltd_rets_ann)

    weekly_details = lazy_property(lambda self: self.txn_details.weekly, 'weekly_details')
    weekly = property(lambda self: self.weekly_details.rets)
    ltd_weekly = property(lambda self: self.weekly_details.ltd_rets)
    ltd_weekly_ann = property(lambda self: self.weekly_details.ltd_rets_ann)

    monthly_details = lazy_property(lambda self: self.txn_details.monthly, 'monthly_details')
    monthly = property(lambda self: self.monthly_details.rets)
    ltd_monthly = property(lambda self: self.monthly_details.ltd_rets)
    ltd_monthly_ann = property(lambda self: self.monthly_details.ltd_rets_ann)

    quarterly_details = lazy_property(lambda self: self.txn_details.quarterly, 'quarterly_details')
    quarterly = property(lambda self: self.quarterly_details.rets)
    ltd_quarterly = property(lambda self: self.quarterly_details.ltd_rets)
    ltd_quarterly_ann = property(lambda self: self.quarterly_details.ltd_rets_ann)

    annual_details = lazy_property(lambda self: self.txn_details.annual, 'annual_details')
    annual = property(lambda self: self.annual_details.rets)
    ltd_annual = property(lambda self: self.annual_details.ltd_rets)
    ltd_annual_ann = property(lambda self: self.annual_details.ltd_rets_ann)

    def iter_by_year(self):
        """Split the return objects by year and iterate"""
        for yr, details in self.txn_details.iter_by_year():
            yield yr, Performance(details)

    def filter(self, txn_mask):
        details = self.txn_details.filter(txn_mask)
        return Performance(details)

    def truncate(self, before=None, after=None):
        details = self.txn_details.truncate(before, after)
        return Performance(details)

    def report_by_year(self, summary_fct=None, years=None, ltd=1, prior_n_yrs=None, first_n_yrs=None, ranges=None,
                       bm_rets=None):
        """Summary the returns
        :param summary_fct: function(Rets) and returns a dict or Series
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
            def summary_fct(performance):
                monthly = performance.monthly_details
                dly = performance.dly_details
                data = OrderedDict()
                data['ltd ann'] = monthly.ltd_ann
                data['mret avg'] = monthly.mean
                data['mret std ann'] = monthly.std_ann
                data['sharpe ann'] = monthly.sharpe_ann
                data['sortino'] = monthly.sortino
                data['maxdd'] = dly.maxdd
                data['maxdd dt'] = dly.maxdd_dt
                if bm_rets is not None:
                    abseries = performance.get_alpha_beta(bm_rets)
                    prefix = {'weekly': 'wkly ', 'monthly': 'mret '}.get(abseries.name, abseries.name)
                    data['{0}beta'.format(prefix)] = abseries['beta']
                    data['{0}alpha'.format(prefix)] = abseries['alpha']
                data['avg dd'] = dly.dd_avg
                data['best month'] = monthly.rets.max()
                data['worst month'] = monthly.rets.min()
                data['nmonths'] = monthly.cnt
                return data

        results = OrderedDict()

        if years is not False:
            for yr, robj in self.iter_by_year():
                if years is None or yr in years:
                    results[yr] = summary_fct(robj)

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
