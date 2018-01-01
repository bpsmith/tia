import pandas as pd
import numpy as np
from pandas.io.data import get_data_yahoo

from tia.analysis.perf import periods_in_year
from tia.analysis.model.interface import CostCalculator, EodMarketData


__all__ = ['InstrumentPrices', 'Instrument', 'Instruments', 'load_yahoo_stock', 'load_bbg_stock', 'load_bbg_future',
           'BloombergInstrumentLoader']


class InstrumentPrices(object):
    def __init__(self, frame):
        self._ensure_ohlc(frame)
        self.frame = frame

    def _ensure_ohlc(self, frame):
        # missing = pd.Index(['open', 'high', 'low', 'close']).difference(frame.columns)
        missing = pd.Index(['open', 'high', 'low', 'close']) - frame.columns
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

    def volatility(self, n, freq=None, which='close', ann=True, model='ln', min_periods=1, rolling='simple'):
        """Return the annualized volatility series. N is the number of lookback periods.

        :param n: int, number of lookback periods
        :param freq: resample frequency or None
        :param which: price series to use
        :param ann: If True then annualize
        :param model: {'ln', 'pct', 'bbg'}
                        ln - use logarithmic price changes
                        pct - use pct price changes
                        bbg - use logarithmic price changes but Bloomberg uses actual business days
        :param rolling:{'simple', 'exp'}, if exp, use ewmstd. if simple, use rolling_std
        :return:
        """
        if model not in ('bbg', 'ln', 'pct'):
            raise ValueError('model must be one of (bbg, ln, pct), not %s' % model)
        if rolling not in ('simple', 'exp'):
            raise ValueError('rolling must be one of (simple, exp), not %s' % rolling)

        px = self.frame[which]
        px = px if not freq else px.resample(freq, how='last')
        if model == 'bbg' and periods_in_year(px) == 252:
            # Bloomberg uses business days, so need to convert and reindex
            orig = px.index
            px = px.resample('B').ffill()
            chg = np.log(px / px.shift(1))
            chg[chg.index - orig] = np.nan
            if rolling == 'simple':
                vol = pd.rolling_std(chg, n, min_periods=min_periods).reindex(orig)
            else:
                vol = pd.ewmstd(chg, span=n, min_periods=n)
            return vol if not ann else vol * np.sqrt(260)
        else:
            chg = px.pct_change() if model == 'pct' else np.log(px / px.shift(1))
            if rolling == 'simple':
                vol = pd.rolling_std(chg, n, min_periods=min_periods)
            else:
                vol = pd.ewmstd(chg, span=n, min_periods=n)
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
        pxs = pxs if pxs is not None else self.pxs.close
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

    def new_buy_and_hold_port(self, qty=1., open_px='close', open_dt=None, close_px='close', close_dt=None,
                              ret_calc=None):
        """

        :param qty: float
        :param open_px: one of {string, float}, opening trade price. If string define open, high, low, close as source.
        :param open_dt: opening trade date
        :param close_px: one of {string, float}, closing trade price. If string define open, high, low, close
        :param close_dt: closing trade date
        :param ret_calc:
        :return:
        """
        from tia.analysis.model.trd import TradeBlotter
        from tia.analysis.model.port import SingleAssetPortfolio

        getpx = lambda how, dt: how if not isinstance(how, str) else self.pxs.frame[how].asof(dt)

        open_dt = open_dt or self.pxs.frame.index[0]
        open_px = getpx(open_px, open_dt)
        close_dt = close_dt or self.pxs.frame.index[-1]
        close_px = getpx(close_px, close_dt)

        pricer = self.truncate(open_dt, close_dt)
        blotter = TradeBlotter()
        blotter.ts = open_dt
        blotter.open(qty, open_px)
        blotter.ts = close_dt
        blotter.close(close_px)
        trds = blotter.trades
        return SingleAssetPortfolio(pricer, trds, ret_calc=ret_calc)

    def __repr__(self):
        return '%s(%r, mult=%s)' % (self.__class__.__name__, self.sid, self.multiplier)


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
        if isinstance(key, str):
            return self._instruments[key]
        elif isinstance(key, int):
            return self._instruments.iloc[key]
        else:
            return Instruments(self._instruments[key])

    def __len__(self):
        return self._instruments.__len__()

    def __iter__(self):
        return self._instruments.__iter__()

    def iteritems(self):
        return iter(self._instruments.items())

    @property
    def frame(self):
        kvals = {sid: ins.pxs.frame for sid, ins in self._instruments.items()}
        return pd.concat(list(kvals.values()), axis=1, keys=list(kvals.keys()))

    def __repr__(self):
        return '[{0}]'.format(','.join([repr(i) for i in self._instruments]))


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
    if hasattr(sids, '__iter__') and not isinstance(sids, str):
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
                # missing = d.index.difference(data.index)
                missing = d.index - data.index
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


def _resolve_accessor(sid_or_accessor):
    if isinstance(sid_or_accessor, str):
        from tia.bbg import BbgDataManager

        mgr = BbgDataManager()
        return mgr.get_sid_accessor(sid_or_accessor)
    else:
        from tia.bbg import SidAccessor

        if not isinstance(sid_or_accessor, SidAccessor):
            raise ValueError('sid_or_accessor must be either a string or SidAccessor not %s' % type(sid_or_accessor))
        return sid_or_accessor


def load_bbg_stock(sid_or_accessor, start=None, end=None, dvds=True):
    """terminal and datamgr are mutually exclusive.

    :param sid_or_accessor: security identifier or SidAccessor from DataManager
    :param start:
    :param end:
    :param dvds:
    :return:
    """
    end = end and pd.to_datetime(end) or pd.datetime.now()
    start = start and pd.to_datetime(start) or end + pd.datetools.relativedelta(years=-1)

    FLDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']
    DVD_FLD = 'DVD_HIST_ALL'
    RENAME = {'PX_OPEN': 'open', 'PX_HIGH': 'high', 'PX_LOW': 'low', 'PX_LAST': 'close'}

    accessor = _resolve_accessor(sid_or_accessor)
    sid = accessor.sid
    pxframe = accessor.get_historical(FLDS, start=start, end=end).rename(columns=RENAME)
    dvdframe = accessor.get_attributes(DVD_FLD, ignore_field_error=1)

    if isinstance(dvdframe, pd.DataFrame):
        dvdframe = dvdframe[['Ex-Date', 'Dividend Amount']].rename(
            columns={'Ex-Date': 'date', 'Dividend Amount': 'dvds'})
        dvdframe = dvdframe.set_index('date').sort_index()
        dvdframe = dvdframe.truncate(start, end)
        # sanity check - not expected currently
        # missing = dvdframe.index.difference(pxframe.index)
        missing = dvdframe.index - pxframe.index
        if len(missing) > 0:
            missing_dates = ','.join([m.strftime('%Y-%m-%d') for m in missing])
            raise Exception('dividends occur on non-business day, not expecting this. %s' % missing_dates)
        # another sanity check to ensure yahoo rolls dividends up, in case a special occurs on same day
        if not dvdframe.index.is_unique:
            dvdframe = dvdframe.groupby(lambda x: x).sum()
        pxframe = pxframe.join(dvdframe)
    pxs = InstrumentPrices(pxframe)
    return Instrument(sid, pxs, multiplier=1.)


def load_bbg_generic(sid_or_accessor, start=None, end=None):
    """terminal and datamgr are mutually exclusive.

    :param sid_or_accessor: security identifier or SidAccessor from DataManager
    :param start:
    :param end:
    :return:
    """
    end = end and pd.to_datetime(end) or pd.datetime.now()
    start = start and pd.to_datetime(start) or end + pd.datetools.relativedelta(years=-1)

    FLDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']
    RENAME = {'PX_OPEN': 'open', 'PX_HIGH': 'high', 'PX_LOW': 'low', 'PX_LAST': 'close'}
    accessor = _resolve_accessor(sid_or_accessor)
    sid = accessor.sid
    pxframe = accessor.get_historical(FLDS, start=start, end=end).rename(columns=RENAME)
    pxs = InstrumentPrices(pxframe)
    return Instrument(sid, pxs, multiplier=1.)


def load_bbg_future(sid_or_accessor, start=None, end=None):
    """terminal and datamgr are mutually exclusive.

    :param sid_or_accessor: security identifier or SidAccessor from DataManager
    :param start:
    :param end:
    :return:
    """
    end = end and pd.to_datetime(end) or pd.datetime.now()
    start = start and pd.to_datetime(start) or end + pd.datetools.relativedelta(years=-1)

    FLDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']
    RENAME = {'PX_OPEN': 'open', 'PX_HIGH': 'high', 'PX_LOW': 'low', 'PX_LAST': 'close'}
    accessor = _resolve_accessor(sid_or_accessor)
    sid = accessor.sid
    pxframe = accessor.get_historical(FLDS, start=start, end=end).rename(columns=RENAME)
    pxs = InstrumentPrices(pxframe)
    mult = 1.
    try:
        mult = float(accessor.FUT_VAL_PT)
    except:
        pass

    return Instrument(sid, pxs, multiplier=mult)


class BloombergInstrumentLoader(object):
    StockTypes = ['Common Stock', 'Mutual Fund', 'Depositary Receipt', 'REIT', 'Partnership Shares']

    def __init__(self, mgr=None, start=None, end=None):
        from tia.bbg import BbgDataManager

        self.mgr = mgr or BbgDataManager()
        self.start = start
        self.end = end

    def load(self, sids, start=None, end=None):
        # TODO - subclss Instrument with specified instrument type
        if isinstance(sids, str):
            start = start or self.start
            end = end or self.end
            accessor = self.mgr[sids]
            sectype2 = accessor.SECURITY_TYP2
            if sectype2 == 'Future':
                return load_bbg_future(accessor, start=start, end=end)
            elif sectype2 == 'Index':
                return load_bbg_generic(accessor, start=start, end=end)
            elif sectype2 == 'CROSS':
                return load_bbg_generic(accessor, start=start, end=end)
            elif sectype2 in self.StockTypes:
                return load_bbg_stock(accessor, start=start, end=end)
            else:
                raise Exception('SECURITY_TYP2 "%s" is not mapped' % sectype2)
        else:
            return Instruments([self.load(sid, start, end) for sid in sids])

