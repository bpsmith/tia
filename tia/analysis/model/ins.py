import pandas as pd
import numpy as np
from pandas.io.data import get_data_yahoo

from tia.analysis.perf import periods_in_year
from tia.analysis.model.interface import CostCalculator, EodMarketData


__all__ = ['InstrumentPrices', 'Instrument', 'load_yahoo_stock', 'load_bbg_stock']


class InstrumentPrices(object):
    def __init__(self, frame):
        self._ensure_ohlc(frame)
        self.frame = frame

    def _ensure_ohlc(self, frame):
        missing = pd.Index(['open', 'high', 'low', 'close']).difference(frame.columns)
        if len(missing) != 0:
            raise ValueError('price frame missing expected columns: {0}'.format(','.join([m for m in missing])))

    open = property(lambda self: self.frame.open)
    high = property(lambda self: self.frame.high)
    low = property(lambda self: self.frame.low)
    close = property(lambda self: self.frame.close)
    dvds = property(lambda self: self.frame.dvds if 'dvds' in self.frame else pd.Series(np.nan, index=self.frame.index))

    def total_return(self):
        """http://en.wikipedia.org/wiki/Total_shareholder_return - mimics bloomberg total return"""
        pxend = self.close
        pxstart = pxend.shift(1).bfill()
        return (1. + (pxend - pxstart + self.dvds.fillna(0)) / pxstart).cumprod() - 1

    def volatility(self, n, freq=None, which='close', ann=True, model='ln', min_periods=1):
        """Return the annualized volatility series. N is the number of lookback periods.

        :param n: int, number of lookback periods
        :param freq: resample frequency or None
        :param which: price series to use
        :param ann: If True then annualize
        :param model: {'ln', 'pct', 'bbg'}
                        ln - use logarithmic price changes
                        pct - use pct price changes
                        bbg - use logarithmic price changes but Bloomberg uses actual business days
        :return:
        """
        if model not in ('bbg', 'ln', 'pct'):
            raise ValueError('model must be one of (bbg, ln, pct), not %s' % model)
        px = self.frame[which]
        px = px if not freq else px.resample(freq, how='last')
        if model == 'bbg' and periods_in_year(px) == 252:
            # Bloomberg uses business days, so need to convert and reindex
            orig = px.index
            px = px.resample('B').ffill()
            chg = np.log(px / px.shift(1))
            chg[chg.index - orig] = np.nan
            vol = pd.rolling_std(chg, n, min_periods=min_periods).reindex(orig)
            return vol if not ann else vol * np.sqrt(260)
        else:
            chg = px.pct_change() if model == 'pct' else np.log(px / px.shift(1))
            vol = pd.rolling_std(chg, n, min_periods=min_periods)
            return vol if not ann else vol * np.sqrt(periods_in_year(vol))


class Instrument(CostCalculator, EodMarketData):
    def __init__(self, sid, pxs=None, multiplier=None):
        if pxs and isinstance(pxs, pd.DataFrame):
            pxs = InstrumentPrices(pxs)
        self.sid = sid
        self.pxs = pxs
        self.multiplier = multiplier

    def get_mkt_val(self, pxs=None):
        """Return the market value series for the series of pxs"""
        pxs = pxs if pxs is None else self.pxs.close
        return pxs * self.multiplier

    def get_premium(self, qty, px, ts=None):
        return -qty * px * self.multiplier

    def get_eod_frame(self):
        """Return the eod market data frame for pricing"""
        close = self.pxs.close
        mktval = self.get_mkt_val(close)
        dvds = self.pxs.dvds
        df = pd.DataFrame({'close': close, 'mkt_val': mktval, 'dvds': dvds})
        df.index.name = 'date'
        return df

    def truncate(self, before=None, after=None):
        """Return an instrument with prices starting at before and ending at after"""
        pxframe = self.pxs.frame
        if (before is None or before == pxframe.index[0]) and (after is None or after == pxframe.index[-1]):
            return self
        else:
            tpxs = self.pxs.frame.truncate(before, after)
            return Instrument(self.sid, InstrumentPrices(tpxs), multiplier=self.multiplier)


class Instruments(object):
    def __init__(self, instruments=None):
        if instruments is None:
            instruments = pd.Series()
        elif isinstance(instruments, (tuple, list)):
            instruments = pd.Series(instruments, index=[i.sid for i in instruments])
        elif not isinstance(instruments, pd.Series):
            raise ValueError('instruments must be None, tuple, list, or Series. Not %s' % type(instruments))
        self._instruments = instruments

    sids = property(lambda self: self._instruments.index)

    def add(self, ins):
        self._instruments = self._instruments.append(pd.Series({ins.sid: ins}))

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self._instruments[key]
        else:
            return Instruments(self._instruments[key])

    @property
    def frame(self):
        kvals = {sid: ins.pxs.frame for sid, ins in self._instruments.iteritems()}
        return pd.concat(kvals.values(), axis=1, keys=kvals.keys())


def get_dividends_yahoo(sid, start, end):
    # Taken from get_data_yahoo in Pandas library and adjust a single parameter to get dividends
    from pandas.compat import StringIO, bytes_to_str
    from pandas.io.common import urlopen

    start, end = pd.to_datetime(start), pd.to_datetime(end)
    url = ('http://ichart.finance.yahoo.com/table.csv?' + 's=%s' % sid +
           '&a=%s' % (start.month - 1) +
           '&b=%s' % start.day +
           '&c=%s' % start.year +
           '&d=%s' % (end.month - 1) +
           '&e=%s' % end.day +
           '&f=%s' % end.year +
           '&g=v' +  # THE CHANGE
           '&ignore=.csv')

    with urlopen(url) as resp:
        lines = resp.read()
    rs = pd.read_csv(StringIO(bytes_to_str(lines)), index_col=0,
                     parse_dates=True, na_values='-')[::-1]
    # Yahoo! Finance sometimes does this awesome thing where they
    # return 2 rows for the most recent business day
    if len(rs) > 2 and rs.index[-1] == rs.index[-2]:  # pragma: no cover
        rs = rs[:-1]
    return rs


def load_yahoo_stock(sids, start=None, end=None, dvds=True):
    if hasattr(sids, '__iter__') and not isinstance(sids, basestring):
        return Instruments([load_yahoo_stock(sid, start=start, end=end, dvds=dvds) for sid in sids])
    else:
        sid = sids
        end = end and pd.to_datetime(end) or pd.datetime.now()
        start = start and pd.to_datetime(start) or end + pd.datetools.relativedelta(years=-1)
        data = get_data_yahoo(sid, start=start, end=end)
        data = data.rename(columns=lambda c: c.lower())
        if dvds:
            d = get_dividends_yahoo(sid, start, end)
            d.columns = ['dvds']
            if not d.empty:
                # sanity check - not expected currently
                missing = d.index.difference(data.index)
                if len(missing) > 0:
                    raise Exception('dividends occur on non-business day, not expecting this')
                # another sanity check to ensure yahoo rolls dividends up, in case a special occurs on same day
                if not d.index.is_unique:
                    d = d.groupby(lambda x: x).sum()
                data = data.join(d)
            else:
                data['dvds'] = np.nan
        pxs = InstrumentPrices(data)
        return Instrument(sid, pxs, multiplier=1.)


def load_bbg_stock(sid_or_accessor, start=None, end=None, dvds=True, terminal=None):
    """terminal and datamgr are mutually exclusive.

    :param sid_or_accessor: security identifier or SidAccessor from DataManager
    :param start:
    :param end:
    :param dvds:
    :param terminal: None or the bloomberg terminal object if just sid is passed
    :return:
    """
    end = end and pd.to_datetime(end) or pd.datetime.now()
    start = start and pd.to_datetime(start) or end + pd.datetools.relativedelta(years=-1)

    FLDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']
    DVD_FLD = 'DVD_HIST_ALL'
    RENAME = {'PX_OPEN': 'open', 'PX_HIGH': 'high', 'PX_LOW': 'low', 'PX_LAST': 'close'}
    if isinstance(sid_or_accessor, basestring):
        # If just the identifier is passed in then use terminal retrieve the data
        if terminal is None:
            from tia.bbg import LocalTerminal

            terminal = LocalTerminal

        pxframe = terminal.get_historical(sid_or_accessor, FLDS, start=start, end=end).as_frame()
        pxframe = pxframe[sid_or_accessor].rename(columns=RENAME)
        dvdframe = terminal.get_reference_data(sid_or_accessor, DVD_FLD, ignore_field_error=1).as_frame().iloc[0][0]
    else:
        if not hasattr(sid_or_accessor, 'get_historical'):
            raise ValueError('sid_or_accessor must define get_historical if it is not a sid')

        accessor = sid_or_accessor
        pxframe = accessor.get_historical(FLDS, start=start, end=end).rename(columns=RENAME)
        dvdframe = accessor.get_attributes(DVD_FLD, ignore_field_error=1)

    if isinstance(dvdframe, pd.DataFrame):
        dvdframe = dvdframe[['Ex-Date', 'Dividend Amount']].rename(
            columns={'Ex-Date': 'date', 'Dividend Amount': 'dvds'})
        dvdframe = dvdframe.set_index('date').sort_index()
        dvdframe = dvdframe.truncate(start, end)
        # sanity check - not expected currently
        missing = dvdframe.index.difference(pxframe.index)
        if len(missing) > 0:
            missing_dates = ','.join([m.strftime('%Y-%m-%d') for m in missing])
            raise Exception('dividends occur on non-business day, not expecting this. %s' % missing_dates)
        # another sanity check to ensure yahoo rolls dividends up, in case a special occurs on same day
        if not dvdframe.index.is_unique:
            dvdframe = dvdframe.groupby(lambda x: x).sum()
        pxframe = pxframe.join(dvdframe)
    pxs = InstrumentPrices(pxframe)
    return Instrument(sid_or_accessor, pxs, multiplier=1.)

