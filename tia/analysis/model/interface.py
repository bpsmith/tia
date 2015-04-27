__all__ = ['CostCalculator', 'EodMarketData', 'MarketDataColumns', 'TxnColumns', 'PositionColumns', 'PlColumns',
           'TxnPlColumns']


class CostCalculator(object):
    """Define the methods necessary to be able to calculator the premium for a trade."""

    def get_premium(self, qty, px, ts=None):
        raise NotImplementedError()

    def get_mkt_val(self, qty, px, ts=None):
        raise NotImplementedError()


class EodMarketData(object):
    def get_eod_frame(self):
        """Return an end of day DataFrame with columns ('close', 'mktval', 'dvd')"""
        raise NotImplementedError()


class MarketDataColumns(object):
    CLOSE = 'close'
    MKT_VAL = 'mkt_val'
    DVDS = 'dvds'


class TxnColumns(object):
    DT = 'date'
    TS = 'txn_ts'
    PID = 'pid'
    TID = 'tid'
    QTY = 'txn_qty'
    PX = 'txn_px'
    FEES = 'txn_fees'
    PREMIUM = 'txn_premium'
    OPEN_VAL = 'open_val'
    POS = 'pos'
    INTENT = 'txn_intent'
    ACTION = 'txn_action'

    DESCRIPTIONS = {
        DT: 'Date-only portion of transaction',
        TS: 'Timestamp of transaction',
        PID: 'position id',
        TID: 'trade id',
        QTY: 'quantity',
        PX: 'price',
        FEES: 'fees',
        PREMIUM: 'premium',
        OPEN_VAL: 'open value of position',
        POS: 'position quantity',
        INTENT: 'trade intent',
        ACTION: 'trade action',
    }



class PlColumns(object):
    DT = 'date'
    DVDS = 'dvds'
    FEES = 'fees'
    RPL_GROSS = 'rpl_gross'
    RPL = 'rpl'
    UPL = 'upl'
    PL = 'pl'

    DESCRIPTIONS = {
        DT: 'p/l date',
        DVDS: 'dividends',
        FEES: 'fees',
        RPL_GROSS: 'realized gross p/l (TOT_VAL - OPEN_VAL)',
        RPL: 'realized pl (RPL_GROSS + FEES + DVDS)',
        UPL: 'unrealized pl (MKT_VAL + OPEN_VAL)',
        PL: 'Total p/l (UPL + RPL)'
    }

    ALL = [DT, DVDS, FEES, RPL_GROSS, RPL, UPL, PL]
    LTDS = [DVDS, FEES, RPL_GROSS, RPL, UPL, PL]


class TxnPlColumns(object):
    DT = 'date'
    PID = TxnColumns.PID
    TID = TxnColumns.TID
    POS = 'pos'
    TXN_QTY = 'txn_qty'
    TXN_PX = 'txn_px'
    TXN_FEES = 'txn_fees'
    TXN_PREMIUM = 'txn_premium'
    TXN_INTENT = 'txn_intent'
    TXN_ACTION = 'txn_action'
    CLOSE_PX = 'close'
    OPEN_VAL = 'open_val'
    MKT_VAL = 'mkt_val'
    TOT_VAL = 'total_val'
    DVDS = 'dvds'
    FEES = 'fees'
    RPL_GROSS = 'rpl_gross'
    RPL = 'rpl'
    UPL = 'upl'
    PL = 'pl'

    DESCRIPTIONS = {
        DT: 'p/l date',
        POS: 'end of day position quantity',
        CLOSE_PX: 'end of day closing price',
        OPEN_VAL: 'open value of the position',
        MKT_VAL: 'market value',
        TOT_VAL: 'total of trade premiums',
        DVDS: 'dividends',
        FEES: 'fees',
        RPL_GROSS: 'realized gross p/l (TOT_VAL - OPEN_VAL)',
        RPL: 'realized pl (RPL_GROSS + FEES + DVDS)',
        UPL: 'unrealized pl (MKT_VAL + OPEN_VAL)',
        PL: 'Total p/l (UPL + RPL)'
    }


class PositionColumns(object):
    PID = 'pid'
    SIDE = 'side'
    OPEN_DT = 'open_dt'
    CLOSE_DT = 'close_dt'
    OPEN_QTY = 'open_qty'
    OPEN_PX = 'open_px'
    CLOSE_PX = 'close_px'
    OPEN_PREMIUM = 'open_premium'
    PL = 'pl'
    DURATION = 'duration'
    NUM_TXNS = 'ntxns'
    RET = 'ret'
    STATE = 'state'