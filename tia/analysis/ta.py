import pandas as pd
import numpy as np

from tia.analysis.model.trd import TradeBlotter
from tia.analysis.util import per_level, per_series


__all__ = ['per_level', 'per_series', 'sma', 'ema', 'wilderma', 'ma', 'macd', 'rsi', 'true_range', 'dmi',
           'cross_signal', 'Signal']


@per_series()
def sma(arg, n):
    """ If n is 0 then return the ltd mean; else return the n day mean """
    if n == 0:
        return pd.expanding_mean(arg)
    else:
        return pd.rolling_mean(arg, n, min_periods=n)


@per_series()
def ema(arg, n):
    if n == 0:
        return pd.ewma(arg, span=len(arg), min_periods=1)
    else:
        return pd.ewma(arg, span=n, min_periods=n)


@per_series()
def wilderma(arg, n):
    converted = arg.dropna()
    values = converted.values

    if len(values) < n:
        return pd.Series(np.nan, index=arg.index)
    else:
        result = np.empty(len(values), dtype=float)
        result[:n - 1] = np.nan
        result[n - 1] = values[:n].mean()
        i, sz = n, len(values)
        pm = 1. / n
        wm = 1. - pm
        while i < sz:
            result[i] = pm * values[i] + wm * result[i - 1]
            i += 1
        return pd.Series(result, index=converted.index).reindex(arg.index)


def ma(arg, n, matype='sma'):
    if matype == 'sma':
        return sma(arg, n)
    elif matype == 'ema':
        return ema(arg, n)
    elif matype == 'wma':
        return wilderma(arg, n)
    else:
        raise ValueError('unknown moving average type %s' % matype)


def _ensure_sorf(arg):
    if not isinstance(arg, (pd.DataFrame, pd.Series)):
        raise Exception('expected Series or DataFrame')


def _ensure_col(arg, **kwds):
    for k, v in kwds.items():
        if v not in arg:
            raise Exception('failed to find column for argument %s=%s' % (k, v))


def true_range(arg, high_col='high', low_col='low', close_col='close', skipna=0):
    """
    http://en.wikipedia.org/wiki/Average_true_range
    The greatest of the following:
    - Current High less the current Low
    - Current High less the previous Close (absolute value)
    - Curre    nt Low less the previous Close (absolute value)
    """
    _ensure_col(arg, high_col=high_col, low_col=low_col, close_col=close_col)
    yclose = arg[close_col].shift(1)
    low, high = arg[low_col], arg[high_col]
    mx = pd.DataFrame({'a': high, 'b': yclose}).max(axis=1, skipna=skipna)
    mn = pd.DataFrame({'a': low, 'b': yclose}).min(axis=1, skipna=skipna)
    result = mx - mn
    return pd.Series(result, index=arg.index, name='true_range')


def dmi(arg, n, high_col='high', low_col='low', close_col='close'):
    """ Return the dmi+, dmi-, Average directional index
    ( http://en.wikipedia.org/wiki/Average_Directional_Index )
        TODO - break up calcuat
    """
    converted = arg[[close_col, high_col, low_col]]
    converted.columns = ['close', 'high', 'low']

    up_mv = converted.high.diff()
    dn_mv = -1 * converted.low.diff()
    up_mv[~((up_mv > 0) & (up_mv > dn_mv))] = 0
    dn_mv[~((dn_mv > 0) & (dn_mv > up_mv))] = 0

    tr = true_range(converted, 'high', 'low', 'close')
    atr = wilderma(tr, n)

    di_pos = 100. * wilderma(up_mv, n) / atr
    di_neg = 100. * wilderma(dn_mv, n) / atr
    dx = 100. * np.abs(di_pos - di_neg) / (di_pos + di_neg)
    adx = wilderma(dx, n)

    data = [
        ('DI+', di_pos),
        ('DI-', di_neg),
        ('DX', dx),
        ('ADX', adx),
    ]
    return pd.DataFrame.from_items(data)


def aroon(arg, n, up_col='close', dn_col='close'):
    """
    TODO - need to verify that the dropna does not take away too many entries (ie maybe set to all? ) This function assumes that
    the length of up_col is always equal to dn_col (ie values not missing in just one series)

    arg: Series or DataFrame
    n: lookback count
    columns: list of up column name, down column name or single column name for both or none
    """
    if isinstance(arg, pd.DataFrame):
        tmp = arg[[up_col, dn_col]].dropna()
        idx = tmp.index
        upvals = tmp[up_col].values
        dnvals = tmp[dn_col].values
    else:
        tmp = arg.dropna()
        idx = tmp.index
        upvals = tmp.values
        dnvals = upvals

    n = int(n)
    up, dn = np.empty(len(upvals), 'd'), np.empty(len(upvals), 'd')
    up[:n] = np.nan
    dn[:n] = np.nan
    for i in range(n, len(upvals)):
        up[i] = 100. * (n - upvals[i - n:i + 1][::-1].argmax()) / n
        dn[i] = 100. * (n - dnvals[i - n:i + 1][::-1].argmin()) / n

    osc = up - dn
    data = [
        ('UP', pd.Series(up, index=idx)),
        ('DOWN', pd.Series(dn, index=idx)),
        ('OSC', pd.Series(osc, index=idx)),
    ]
    return pd.DataFrame.from_items(data).reindex(arg.index)


@per_series(result_is_frame=1)
def macd(arg, nslow=26, nfast=12, nsignal=9):
    nslow, nfast, nsignal = int(nslow), int(nfast), int(nsignal)
    emafast = ema(arg, nfast)
    emaslow = ema(arg, nslow)
    line = emafast - emaslow
    signal = ema(line, nsignal)
    hist = line - signal
    data = [
        ('MACD_FAST', emafast),
        ('MACD_SLOW', emaslow),
        ('MACD', line),
        ('MACD_SIGNAL', signal),
        ('MACD_HIST', hist),
    ]
    return pd.DataFrame.from_items(data)


@per_series()
def rsi(arg, n):
    """ compute RSI for the given arg

    arg: Series or DataFrame
    """
    if isinstance(arg, pd.DataFrame):
        cols = [(name, rsi(arg[name], n)) for name in arg.columns]
        return pd.DataFrame.from_items(cols)
    else:
        assert isinstance(arg, pd.Series)
        n = int(n)
        converted = arg.dropna()
        change = converted.diff()
        gain = change.apply(lambda c: c > 0 and c or 0)
        avgGain = wilderma(gain, n)
        loss = change.apply(lambda c: c < 0 and abs(c) or 0)
        avgLoss = wilderma(loss, n)

        result = avgGain / avgLoss
        result[result == np.inf] = 100.  # divide by zero
        result = 100. - (100. / (1. + result))
        return pd.Series(result, index=converted.index).reindex(arg.index)


def cross_signal(s1, s2, continuous=0):
    """ return a signal with the following
    1 : when all values of s1 cross all values of s2
    -1 : when all values of s2 cross below all values of s2
    0 : if s1 < max(s2) and s1 > min(s2)
    np.nan : if s1 or s2 contains np.nan at position


    s1: Series, DataFrame, float, int, or tuple(float|int)
    s2: Series, DataFrame, float, int, or tuple(float|int)
    continous: bool, if true then once the signal starts it is always 1 or -1
    """

    def _convert(src, other):
        if isinstance(src, pd.DataFrame):
            return src.min(axis=1, skipna=0), src.max(axis=1, skipna=0)
        elif isinstance(src, pd.Series):
            return src, src
        elif isinstance(src, (int, float)):
            s = pd.Series(src, index=other.index)
            return s, s
        elif isinstance(src, (tuple, list)):
            l, u = min(src), max(src)
            assert l <= u, 'lower bound must be less than upper bound'
            lower, upper = pd.Series(l, index=other.index), pd.Series(u, index=other.index)
            return lower, upper
        else:
            raise Exception('unable to handle type %s' % type(src))

    lower1, upper1 = _convert(s1, s2)
    lower2, upper2 = _convert(s2, s1)

    df = pd.DataFrame({'upper1': upper1, 'lower1': lower1, 'upper2': upper2, 'lower2': lower2})
    df.ffill(inplace=True)

    signal = pd.Series(np.nan, index=df.index)
    signal[df.upper1 > df.upper2] = 1
    signal[df.lower1 < df.lower2] = -1

    if continuous:
        # Just roll with 1, -1
        signal = signal.fillna(method='ffill')
        m1, m2 = df.upper1.first_valid_index(), df.upper2.first_valid_index()
        if m1 is not None or m2 is not None:
            m1 = m2 if m1 is None else m1
            m2 = m1 if m2 is None else m2
            fv = max(m1, m2)
            if np.isnan(signal[fv]):
                signal[fv] = 0
                signal.ffill(inplace=1)
    else:
        signal[(df.upper1 < df.upper2) & (df.lower1 > df.lower2)] = 0
        # special handling when equal, determine where it previously was
        eq = (df.upper1 == df.upper2)
        if eq.any():  # Set to prior value
            tmp = signal[eq]
            for i in tmp.index:
                loc = signal.index.get_loc(i)
                if loc != 0:
                    u, l = df.upper2.iloc[loc], df.lower2.iloc[loc]
                    ps = signal.iloc[loc - 1]
                    if u == l or ps == 1.:  # Line coming from above upper bound if ps == 1
                        signal[i] = ps
                    else:
                        signal[i] = 0

        eq = (df.lower1 == df.lower2)
        if eq.any():  # Set to prior value
            tmp = signal[eq]
            for i in tmp.index:
                loc = signal.index.get_loc(i)
                if loc != 0:
                    u, l = df.upper2.iloc[loc], df.lower2.iloc[loc]
                    ps = signal.iloc[loc - 1]
                    if u == l or ps == -1.:  # Line coming from below lower bound if ps == -1
                        signal[i] = ps
                    else:
                        signal[i] = 0

    return signal


class Signal(object):
    def __init__(self, signal, qtys=None):
        self.signal = signal
        self.qtys = qtys

    def _qty_fct(self, index):
        qtys = self.qtys
        if qtys is None:
            qtyfct = lambda ts: 1.
        elif isinstance(qtys, pd.Series):
            qtys = qtys.reindex(index, method='ffill').bfill()
            qtyfct = lambda ts: qtys[ts]
        elif callable(qtys):
            qtyfct = qtys
        else:
            raise ValueError('qtys must be one of (scalar, Series, function)')
        return qtyfct

    def close_to_close(self, pxs):
        """
        :param pxs:
        :return:
        """
        if not isinstance(pxs, pd.Series):
            raise ValueError('pxs expected to be Series')

        blotter = TradeBlotter()
        signal = self.signal.dropna()
        diff = signal.diff()
        diff.iloc[0] = 0
        changes = signal[diff != 0]
        lsig = 0
        qtyfct = self._qty_fct(pxs.index)
        for ts, sig in changes.items():
            blotter.ts = ts
            if sig != lsig:
                px = pxs.get(ts, None)
                if px is None:
                    raise Exception('insufficient price data: no data found at %s' % ts)
                if lsig != 0:  # close open trd
                    blotter.close(px=px)

                if sig != 0:
                    size = abs(qtyfct(ts))
                    qty = sig > 0 and size or -size
                    blotter.open(qty, px)
            lsig = sig
        return blotter.trades

    def open_to_close(self, open_pxs, close_pxs):
        blotter = TradeBlotter()
        signal = self.signal.dropna()
        diff = signal.diff()
        diff.iloc[0] = 0
        changes = signal[diff != 0]
        lsig = 0
        nopen = len(open_pxs)
        qtyfct = self._qty_fct(open_pxs.index)
        for ts, sig in changes.items():
            blotter.ts = ts
            if sig != lsig:
                if lsig != 0:
                    # close open trd with today's closing price
                    px = close_pxs.get(ts, None)
                    if px is None:
                        raise Exception('insufficient close price data: no data found at %s' % ts)
                    blotter.close(px)

                if sig != 0:
                    idx = open_pxs.index.get_loc(ts)
                    if (idx + 1) != nopen:
                        ts_plus1 = open_pxs.index[idx + 1]
                        px = open_pxs.iloc[idx + 1]
                        size = abs(qtyfct(ts))
                        qty = sig > 0 and size or -size
                        blotter.ts = ts_plus1
                        blotter.open(qty, px)
                    else:
                        pass
            lsig = sig
        return blotter.trades


# make upper case available to match ta-lib wrapper
RSI = rsi
MACD = macd
DMI = dmi
SMA = sma
EMA = ema
MA = ma
TRANGE = true_range
