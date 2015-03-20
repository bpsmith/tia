from collections import OrderedDict

import pandas as pd
import numpy as np

from tia.analysis.model.interface import TxnColumns as TC, MarketDataColumns as MC, PlColumns as PL
from tia.util.decorator import lazy_property


__all__ = ['Pl']


def _dly_to_ltd(frame, dly_cols):
    frame = frame.copy()
    ilocs = [frame.columns.get_loc(_) for _ in dly_cols]
    sums = frame[dly_cols].cumsum()
    frame.iloc[:, ilocs] = sums.iloc[:, range(len(dly_cols))]
    return frame


def _ltd_to_dly(frame, ltd_cols):
    pl = frame.copy()
    ilocs = [frame.columns.get_loc(_) for _ in ltd_cols]
    diff = frame[ltd_cols].diff()
    # not sure why this is failing
    #pl.iloc[1:, ilocs] = diff.iloc[1:]
    for i, cidx in enumerate(ilocs):
        pl.iloc[1:, cidx] = diff.iloc[1:, i]
    return pl


class Pl(object):
    _LTDS = [PL.FEES, PL.TOT_VAL, PL.RPL_GROSS, PL.DVDS, PL.RPL_GROSS, PL.RPL, PL.UPL, PL.PL]

    def __init__(self, txns):
        """
        :param txns: Txns object
        """
        self.txns = txns

    @lazy_property
    def ltd_txn_frame(self):
        """Compute the long/short live-to-date transaction level profit and loss. Uses an open average calculation"""
        txndata = self.txns.frame
        mktdata = self.txns.pricer.get_frame()
        if not isinstance(mktdata.index, pd.DatetimeIndex):
            mktdata.to_timestamp(freq='B')

        # get the set of all txn dts and mkt data dts
        pl = pd.merge(txndata, mktdata.reset_index(), how='outer', on=PL.DT)
        if pl[TC.PID].isnull().all():
            cols = [PL.DT, PL.POS, PL.PID, PL.TID,  PL.TXN_QTY, PL.TXN_PX, PL.TXN_FEES, PL.TXN_PREMIUM, PL.TXN_INTENT,
                     PL.TXN_ACTION, PL.CLOSE_PX, PL.OPEN_VAL, PL.MKT_VAL, PL.TOT_VAL, PL.DVDS, PL.FEES, PL.RPL_GROSS,
                     PL.RPL, PL.UPL, PL.PL]
            return pd.DataFrame(columns=cols)
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
            cols = [TC.DT, TC.POS, TC.PID, TC.TID, TC.INTENT, TC.ACTION, TC.FEES, TC.QTY, TC.PX, TC.PREMIUM, TC.OPEN_VAL]
            dts, pos_qtys, pids, tids, intents, sides, txn_fees, txn_qtys, txn_pxs, premiums, open_vals = [pl[c] for c in
                                                                                                           cols]

            dvds, closing_pxs, mkt_vals = [pl[c] for c in [MC.DVDS, MC.CLOSE, MC.MKT_VAL]]
            # Ensure only end of day is kept for dividends (join will match dvd to any transaction during day
            dvds[dts == dts.shift(-1)] = 0
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
            data[PL.DT] = dts
            data[PL.POS] = pos_qtys
            data[PL.PID] = pids
            data[PL.TID] = tids
            data[PL.TXN_QTY] = txn_qtys
            data[PL.TXN_PX] = txn_pxs
            data[PL.TXN_FEES] = txn_fees
            data[PL.TXN_PREMIUM] = premiums
            data[PL.TXN_INTENT] = intents
            data[PL.TXN_ACTION] = sides
            data[PL.CLOSE_PX] = closing_pxs
            data[PL.OPEN_VAL] = open_vals
            data[PL.MKT_VAL] = mkt_vals
            data[PL.TOT_VAL] = total_vals
            data[PL.DVDS] = dvds
            data[PL.FEES] = fees
            data[PL.RPL_GROSS] = rpl_gross
            data[PL.RPL] = rpl
            data[PL.UPL] = upl
            data[PL.PL] = tpl
            frame = pd.DataFrame(data, columns=data.keys())
            return frame

    @lazy_property
    def dly_txn_frame(self):
        ltds = [PL.FEES, PL.TOT_VAL, PL.RPL_GROSS, PL.DVDS, PL.RPL, PL.RPL, PL.UPL, PL.PL]
        return _ltd_to_dly(self.ltd_txn_frame, ltds)

    @property
    def ltd_frame(self):
        txnlvl = self.ltd_txn_frame[PL.PL_COLUMNS]
        dly = txnlvl.set_index(PL.DT).resample('B', 'last', kind='period').dropna(how='all')
        return dly

    @property
    def dly_frame(self):
        return _ltd_to_dly(self.ltd_frame, self._LTDS)

    ltd_txn = property(lambda self: self.ltd_txn_frame[PL.PL])
    dly_txn = property(lambda self: self.dly_txn_frame[PL.PL])
    ltd = property(lambda self: self.ltd_frame[PL.PL])
    annual = property(lambda self: self.dly.resample('A', how='sum'))
    quarterly = property(lambda self: self.dly.resample('Q', how='sum'))
    monthly = property(lambda self: self.dly.resample('M', how='sum'))
    weekly = property(lambda self: self.dly.resample('W', how='sum'))
    dly = property(lambda self: self.dly_frame[PL.PL])

    def subset(self, txns):
        """To perform a subset it is not possible to reuse the frame since it is LTD, so we convert to daily then
        compute ltd from daily
        :param txns: the update Txns object
        :return:
        """
        result = Pl(txns)
        # TODO - add reusing calcs. Issue is when removing PIDs, then could be multiple entries per dt
        # use daily txn, clear all values where != pid
        # determine which Timestamp columns can be removed as an old position may have multiple txns on same day
        # recreate ltd from dly
        return result

