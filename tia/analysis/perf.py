"""
inspiration from R Package - PerformanceAnalytics
"""
from collections import OrderedDict

import pandas as pd
import numpy as np

from tia.analysis.util import per_series


PER_YEAR_MAP = {
    'BA': 1.,
    'BAS': 1.,
    'A': 1.,
    'AS': 1.,
    'BQ': 4.,
    'BQS': 4.,
    'Q': 4.,
    'QS': 4.,
    'D': 365.,
    'B': 252.,
    'BMS': 12.,
    'BM': 12.,
    'MS': 12.,
    'M': 12.,
    'W': 52.,
}


def guess_freq(index):
    # admittedly weak way of doing this...This needs to be abolished
    if isinstance(index, (pd.Series, pd.DataFrame)):
        index = index.index

    if hasattr(index, 'freqstr') and index.freqstr:
        return index.freqstr[0]
    elif len(index) < 3:
        raise Exception('cannot guess frequency with less than 3 items')
    else:
        lb = min(7, len(index))
        idx_zip = lambda: list(zip(index[-lb:-1], index[-(lb-1):]))

        diff = min([t2 - t1 for t1, t2, in idx_zip()])
        if diff.days <= 1:
            if 5 in index.dayofweek or 6 in index.dayofweek:
                return 'D'
            else:
                return 'B'
        elif diff.days == 7:
            return 'W'
        else:
            diff = min([t2.month - t1.month for t1, t2, in idx_zip()])
            if diff == 1:
                return 'M'
            diff = min([t2.year - t1.year for t1, t2, in idx_zip()])
            if diff == 1:
                return 'A'

            strs = ','.join([i.strftime('%Y-%m-%d') for i in index[-lb:]])
            raise Exception('unable to determine frequency, last %s dates %s' % (lb, strs))


def periodicity(freq_or_frame):
    """
    resolve the number of periods per year
    """
    if hasattr(freq_or_frame, 'rule_code'):
        rc = freq_or_frame.rule_code
        rc = rc.split('-')[0]
        factor = PER_YEAR_MAP.get(rc, None)
        if factor is not None:
            return factor / abs(freq_or_frame.n)
        else:
            raise Exception('Failed to determine periodicity. No factor mapping for %s' % freq_or_frame)
    elif isinstance(freq_or_frame, str):
        factor = PER_YEAR_MAP.get(freq_or_frame, None)
        if factor is not None:
            return factor
        else:
            raise Exception('Failed to determine periodicity. No factor mapping for %s' % freq_or_frame)
    elif isinstance(freq_or_frame, (pd.Series, pd.DataFrame, pd.TimeSeries)):
        freq = freq_or_frame.index.freq
        if not freq:
            freq = pd.infer_freq(freq_or_frame.index)
            if freq:
                return periodicity(freq)
            else:
                # Attempt to resolve it
                import warnings

                freq = guess_freq(freq_or_frame.index)
                warnings.warn('frequency not set. guessed it to be %s' % freq)
                return periodicity(freq)
        else:
            return periodicity(freq)
    else:
        raise ValueError("periodicity expects DataFrame, Series, or rule_code property")


periods_in_year = periodicity


def _resolve_periods_in_year(scale, frame):
    """ Convert the scale to an annualzation factor.  If scale is None then attempt to resolve from frame. If scale is a scalar then
        use it. If scale is a string then use it to lookup the annual factor
    """
    if scale is None:
        return periodicity(frame)
    elif isinstance(scale, str):
        return periodicity(scale)
    elif np.isscalar(scale):
        return scale
    else:
        raise ValueError("scale must be None, scalar, or string, not %s" % type(scale))


def excess_returns(returns, bm=0):
    """
    Return the excess amount of returns above the given benchmark bm
    """
    return returns - bm


def returns(prices, method='simple', periods=1, fill_method='pad', limit=None, freq=None):
    """
     compute the returns for the specified prices.
     method: [simple,compound,log], compound is log
    """
    if method not in ('simple', 'compound', 'log'):
        raise ValueError("Invalid method type. Valid values are ('simple', 'compound')")

    if method == 'simple':
        return prices.pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq)
    else:
        if freq is not None:
            raise NotImplementedError("TODO: implement this logic if needed")

        if isinstance(prices, pd.Series):
            if fill_method is None:
                data = prices
            else:
                data = prices.fillna(method=fill_method, limit=limit)

            data = np.log(data / data.shift(periods=periods))
            mask = pd.isnull(prices.values)
            np.putmask(data.values, mask, np.nan)
            return data
        else:
            return pd.DataFrame(
                {name: returns(col, method, periods, fill_method, limit, freq) for name, col in prices.items()},
                columns=prices.columns,
                index=prices.index)


def returns_cumulative(returns, geometric=True, expanding=False):
    """ return the cumulative return

    Parameters
    ----------
    returns : DataFrame or Series
    geometric : bool, default is True
                If True, geometrically link returns
    expanding : bool default is False
                If True, return expanding series/frame of returns
                If False, return the final value(s)
    """
    if expanding:
        if geometric:
            return (1. + returns).cumprod() - 1.
        else:
            return returns.cumsum()
    else:
        if geometric:
            return (1. + returns).prod() - 1.
        else:
            return returns.sum()


def rolling_returns_cumulative(returns, window, min_periods=1, geometric=True):
    """ return the rolling cumulative returns

    Parameters
    ----------
    returns : DataFrame or Series
    window : number of observations
    min_periods : minimum number of observations in a window
    geometric : link the returns geometrically
    """
    if geometric:
        rc = lambda x: (1. + x[np.isfinite(x)]).prod() - 1.
    else:
        rc = lambda x: (x[np.isfinite(x)]).sum()

    return pd.rolling_apply(returns, window, rc, min_periods=min_periods)


def returns_annualized(returns, geometric=True, scale=None, expanding=False):
    """ return the annualized cumulative returns

    Parameters
    ----------
    returns : DataFrame or Series
    geometric : link the returns geometrically
    scale: None or scalar or string (ie 12 for months in year),
           If None, attempt to resolve from returns
           If scalar, then use this as the annualization factor
           If string, then pass this to periodicity function to resolve annualization factor
    expanding: bool, default is False
                If True, return expanding series/frames.
                If False, return final result.
    """
    scale = _resolve_periods_in_year(scale, returns)
    if expanding:
        if geometric:
            n = pd.expanding_count(returns)
            return ((1. + returns).cumprod() ** (scale / n)) - 1.
        else:
            return pd.expanding_mean(returns) * scale
    else:
        if geometric:
            n = returns.count()
            return ((1. + returns).prod() ** (scale / n)) - 1.
        else:
            return returns.mean() * scale


def drawdowns(returns, geometric=True):
    """
    compute the drawdown series for the period return series
    return: periodic return Series or DataFrame
    """
    wealth = 1. + returns_cumulative(returns, geometric=geometric, expanding=True)
    values = wealth.values
    if values.ndim == 2:
        ncols = values.shape[-1]
        values = np.vstack(([1.] * ncols, values))
        maxwealth = pd.expanding_max(values)[1:]
        dds = wealth / maxwealth - 1.
        dds[dds > 0] = 0  # Can happen if first returns are positive
        return dds
    elif values.ndim == 1:
        values = np.hstack(([1.], values))
        maxwealth = pd.expanding_max(values)[1:]
        dds = wealth / maxwealth - 1.
        dds[dds > 0] = 0  # Can happen if first returns are positive
        return dds
    else:
        raise ValueError('unable to process array with %s dimensions' % values.ndim)


def max_drawdown(returns=None, geometric=True, dd=None, inc_date=False):
    """
    compute the max draw down.
    returns: period return Series or DataFrame
    dd: drawdown Series or DataFrame (mutually exclusive with returns)
    """
    if (returns is None and dd is None) or (returns is not None and dd is not None):
        raise ValueError('returns and drawdowns are mutually exclusive')

    if returns is not None:
        dd = drawdowns(returns, geometric=geometric)

    if isinstance(dd, pd.DataFrame):
        vals = [max_drawdown(dd=dd[c], inc_date=inc_date) for c in dd.columns]
        cols = ['maxxdd'] + (inc_date and ['maxdd_dt'] or [])
        res = pd.DataFrame(vals, columns=cols, index=dd.columns)
        return res if inc_date else res.maxdd
    else:
        mddidx = dd.idxmin()
        # if mddidx == dd.index[0]:
        # # no maxff
        #    return 0 if not inc_date else (0, None)
        #else:
        sub = dd[:mddidx]
        start = sub[::-1].idxmax()
        mdd = dd[mddidx]
        # return start, mddidx, mdd
        return mdd if not inc_date else (mdd, mddidx)


@per_series(result_is_frame=1)
def drawdown_info(returns, geometric=True):
    """Return a DataFrame containing information about ALL the drawdowns for the rets. The frame
    contains the columns:
    'dd start': drawdown start date
    'dd end': drawdown end date
    'maxdd': maximium drawdown
    'maxdd dt': maximum drawdown
    'days': duration of drawdown
    """
    dd = drawdowns(returns, geometric=True).to_frame()
    last = dd.index[-1]
    dd.columns = ['vals']
    dd['nonzero'] = (dd.vals != 0).astype(int)
    dd['gid'] = (dd.nonzero.shift(1) != dd.nonzero).astype(int).cumsum()
    idxname = dd.index.name or 'index'
    ixs = dd.reset_index().groupby(['nonzero', 'gid'])[idxname].apply(lambda x: np.array(x))
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


def std_annualized(returns, scale=None, expanding=0):
    scale = _resolve_periods_in_year(scale, returns)
    if expanding:
        return np.sqrt(scale) * pd.expanding_std(returns)
    else:
        return np.sqrt(scale) * returns.std()


def sharpe(returns, rfr=0, expanding=0):
    """
    returns: periodic return string
    rfr: risk free rate
    expanding: bool
    """
    if expanding:
        excess = excess_returns(returns, rfr)
        return pd.expanding_mean(excess) / pd.expanding_std(returns)
    else:
        return excess_returns(returns, rfr).mean() / returns.std()


def sharpe_annualized(returns, rfr_ann=0, scale=None, expanding=False, geometric=False):
    scale = _resolve_periods_in_year(scale, returns)
    stdann = std_annualized(returns, scale=scale, expanding=expanding)
    retsann = returns_annualized(returns, scale=scale, expanding=expanding, geometric=geometric)
    return (retsann - rfr_ann) / stdann


def downside_deviation(rets, mar=0, expanding=0, full=0, ann=0):
    """Compute the downside deviation for the specifed return series
    :param rets: periodic return series
    :param mar: minimum acceptable rate of return (MAR)
    :param full: If True, use the lenght of full series. If False, use only values below MAR
    :param expanding:
    :param ann: True if result should be annualized
    """
    below = rets[rets < mar]
    if expanding:
        n = pd.expanding_count(rets)[below.index] if full else pd.expanding_count(below)
        dd = np.sqrt(((below - mar) ** 2).cumsum() / n)
        if ann:
            dd *= np.sqrt(periods_in_year(rets))
        return dd.reindex(rets.index).ffill()
    else:
        n = rets.count() if full else below.count()
        dd = np.sqrt(((below - mar) ** 2).sum() / n)
        if ann:
            dd *= np.sqrt(periods_in_year(rets))
        return dd


def sortino_ratio(rets, rfr_ann=0, mar=0, full=0, expanding=0):
    """Compute the sortino ratio as (Ann Rets - Risk Free Rate) / Downside Deviation Ann

    :param rets: period return series
    :param rfr_ann: annualized risk free rate
    :param mar: minimum acceptable rate of return (MAR)
    :param full: If True, use the lenght of full series. If False, use only values below MAR
    :param expanding:
    :return:
    """
    annrets = returns_annualized(rets, expanding=expanding) - rfr_ann
    return annrets / downside_deviation(rets, mar=mar, expanding=expanding, full=full, ann=1)


def information_ratio(rets, bm_rets, scale=None, expanding=False):
    """Information ratio, a common measure of manager efficiency, evaluates excess returns over a benchmark
    versus tracking error.

    :param rets: period returns
    :param bm_rets: periodic benchmark returns (not annualized)
    :param scale: None or the scale to be used for annualization
    :param expanding:
    :return:
    """
    scale = _resolve_periods_in_year(scale, rets)
    rets_ann = returns_annualized(rets, scale=scale, expanding=expanding)
    bm_rets_ann = returns_annualized(rets, scale=scale, expanding=expanding)
    tracking_error_ann = std_annualized((rets - bm_rets), scale=scale, expanding=expanding)
    return (rets_ann - bm_rets_ann) / tracking_error_ann


def upside_potential_ratio(rets, mar=0, full=0, expanding=0):
    if isinstance(rets, pd.Series):
        above = rets[rets > mar]
        excess = -mar + above
        if expanding:
            n = pd.expanding_count(rets) if full else pd.expanding_count(above)
            upside = excess.cumsum() / n
            downside = downside_deviation(rets, mar=mar, full=full, expanding=1)
            return (upside / downside).reindex(rets.index).fillna(method='ffill')
        else:
            n = rets.count() if full else above.count()
            upside = excess.sum() / n
            downside = downside_deviation(rets, mar=mar, full=full)
            return upside / downside
    else:
        vals = {c: upside_potential_ratio(rets[c], mar=mar, full=full, expanding=expanding) for c in rets.columns}
        if expanding:
            return pd.DataFrame(vals, columns=rets.columns)
        else:
            return pd.Series(vals)


@per_series()
def rolling_percentileofscore(series, window, min_periods=None):
    """Computue the score percentile for the specified window."""
    import scipy.stats as stats

    def _percentile(arr):
        score = arr[-1]
        vals = arr[:-1]
        return stats.percentileofscore(vals, score)

    notnull = series.dropna()
    min_periods = min_periods or window
    if notnull.empty:
        return pd.Series(np.nan, index=series.index)
    else:
        return pd.rolling_apply(notnull, window, _percentile, min_periods=min_periods).reindex(series.index)


@per_series()
def expanding_percentileofscore(series, min_periods=None):
    import scipy.stats as stats

    def _percentile(arr):
        score = arr[-1]
        vals = arr[:-1]
        return stats.percentileofscore(vals, score)

    notnull = series.dropna()
    if notnull.empty:
        return pd.Series(np.nan, index=series.index)
    else:
        return pd.expanding_apply(notnull, _percentile, min_periods=min_periods).reindex(series.index)


def hurst_exponent(px, lags=list(range(2, 100))):
    """
    describe the prices
    http://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    H = the hurts exponent
    H < 0.5 - mean reverting
    H = 0.5 - geometric brownian motion (aka random)
    H > 0.5 - trending series
    """
    ts = px.reset_index(drop=True).dropna()
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(ts.diff(lag).std()) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def summarize_returns(period_rets, rollup='M', prefix=1, ret_method='compound', yearly=1, ltd=1):
    # TODO - should be able to handle DateTimeIndex
    if not isinstance(period_rets.index, pd.PeriodIndex):
        raise Exception('expected periodic return series')

    def _analyze(rets):
        rets = rets.dropna()
        if rollup != rets.index.freqstr:
            if ret_method == 'simple':
                resampled = rets.resample(rollup, 'sum')
            elif ret_method == 'compound':
                resampled = (1. + rets).resample(rollup, 'prod') - 1.
            else:
                raise Exception('ret_method must be on of [simple, compound]')
        else:
            resampled = rets

        rfreq = rollup.lower().replace('b', 'd')
        pfreq = period_rets.index.freqstr.lower().replace('b', 'd')
        fmap = {'d': 'daily', 'm': 'monthly', 'w': 'weekly', 'q': 'quarterly'}
        rfreq = fmap.get(rfreq, pfreq)
        pfreq = fmap.get(pfreq, pfreq)

        pds_per_yr = periodicity(resampled)
        d = OrderedDict()
        d['{0}_ret_avg'.format(rfreq)] = ret_avg = resampled.mean()
        d['{0}_ret_cum'.format(rfreq)] = returns_cumulative(resampled)
        d['{0}_stdev'.format(rfreq)] = resampled.std()
        d['{0}_stdev_ann'.format(rfreq)] = std_ann = std_annualized(resampled)
        d['{0}_sharpe_ann'.format(rfreq)] = pds_per_yr * ret_avg / std_ann
        d['{0}_sortino'.format(rfreq)] = sortino_ratio(resampled, full=0)
        mdd = max_drawdown(rets, inc_date=1)
        d['{0}_maxdd'.format(pfreq)] = mdd[0]
        d['{0}_maxdd_dt'.format(pfreq)] = mdd[1]
        return d

    if isinstance(period_rets, pd.DataFrame):
        # Multi-indexed data frame needed to show multiple rows per year
        arr = []
        for c in period_rets:
            res = summarize_returns(period_rets[c], rollup=rollup, yearly=yearly, ltd=ltd)
            names = list(res.index.names)
            names.append(period_rets.index.name)
            res.index = pd.MultiIndex.from_arrays([res.index, [c] * len(res.index)], names=names)
            arr.append(res)
        return pd.concat(arr).sortlevel()
    else:
        arrs = []
        if yearly:
            arrs.append(period_rets.groupby(lambda k: k.year).apply(lambda tmp: _analyze(tmp.sort_index())).unstack(1))
        if ltd:
            arrs.append(pd.DataFrame(_analyze(period_rets), index=['LTD']))
        summary = pd.concat(arrs)
        summary.index.name = 'period'
        return summary
