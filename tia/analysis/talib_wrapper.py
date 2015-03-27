"""

help to interface with ta-lib dealing with NaNs and returning timeseries and such

"""
import talib
import pandas as pd

from tia.analysis.util import per_series, per_level


@per_level()
def _frame_to_series(frame, colnames, fct, *fctargs):
    args = [frame[c].values for c in colnames]
    args.extend(fctargs)
    values = fct(*args)
    return pd.Series(values, index=frame.index)


@per_level()
def _frame_to_frame(frame, input_names, output_names, fct, *fctargs):
    args = [frame[c].values for c in input_names]
    args.extend(fctargs)
    result = fct(*args)
    data = {n: result[i] for i, n in enumerate(output_names)}
    f = pd.DataFrame(data, index=frame.index, columns=output_names)
    return f


@per_series(result_is_frame=1)
def _series_to_frame(series, output_names, fct, *fctargs):
    nonulls = series.dropna()
    result = fct(nonulls.values, *fctargs)
    data = {n: result[i] for i, n in enumerate(output_names)}
    f = pd.DataFrame(data, index=nonulls.index, columns=output_names)
    return f.reindex(series.index)


@per_series()
def _series_to_series(series, fct, *fctargs):
    nonulls = series.dropna()
    return pd.Series(fct(nonulls.values, *fctargs), index=nonulls.index, name=series.name).reindex(series.index)


def ACOS(series):
    return _series_to_series(series, talib.ACOS)


def AD(frame, high_col='high', low_col='low', close_col='close', vol_col='Volume'):
    """Chaikin A/D Line"""
    return _frame_to_series(frame, [high_col, low_col, close_col, vol_col], talib.AD)


def ADOSC(frame, fast=3, slow=10, high_col='high', low_col='low', close_col='close', vol_col='Volume'):
    """Chaikin A/D oscillator"""
    return _frame_to_series(frame, [high_col, low_col, close_col, vol_col], talib.ADOSC, fast, slow)


def ADX(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.ADX, n)


def ADXR(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.ADXR, n)


def APO(series, fast=12, slow=26, matype=0):
    """double exponential moving average"""
    return _series_to_series(series, talib.APO, fast, slow, matype)


def AROON(frame, n=14, high_col='high', low_col='low'):
    return _frame_to_frame(frame, [high_col, low_col], ['AroonDown', 'AroonUp'], talib.AROON, n)


def AROONOSC(frame, n=14, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.AROONOSC, n)


def ASIN(series):
    return _series_to_series(series, talib.ASIN)


def ATAN(series):
    return _series_to_series(series, talib.ATAN)


def ATR(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.ATR, n)


def AVGPRICE(frame, open_col='PX_OPEN', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.AVGPRICE)


def BBANDS(series, n=5, devup=2., devdn=2., matype=0):
    return _series_to_frame(series, ['UpperBand', 'MiddleBand', 'LowerBand'], talib.BBANDS, n, devup, devdn, matype)


def BETA(frame, col0, col1, n=5):
    return _frame_to_series(frame, [col0, col1], talib.BETA, n)


def BOP(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.BOP)


def CCI(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.CCI, n)


def CDL2CROWS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL2CROWS)


def CDL3BLACKCROWS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3BLACKCROWS)


def CDL3INSIDE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3INSIDE)


def CDL3LINESTRIKE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3LINESTRIKE)


def CDL3OUTSIDE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3OUTSIDE)


def CDL3STARSINSOUTH(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3STARSINSOUTH)


def CDL3WHITESOLDIERS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDL3WHITESOLDIERS)


def CDLABANDONEDBABY(frame, penetration=.3, open_col='open', high_col='high', low_col='low',
                     close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLABANDONEDBABY, penetration)


def CDLADVANCEBLOCK(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLADVANCEBLOCK)


def CDLBELTHOLD(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLBELTHOLD)


def CDLBREAKAWAY(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLBREAKAWAY)


def CDLCLOSINGMARUBOZU(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLCLOSINGMARUBOZU)


def CDLCONCEALBABYSWALL(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLCONCEALBABYSWALL)


def CDLCOUNTERATTACK(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLCOUNTERATTACK)


def CDLDARKCLOUDCOVER(frame, penetration=.5, open_col='open', high_col='high', low_col='low',
                      close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLDARKCLOUDCOVER, penetration)


def CDLDOJI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLDOJI)


def CDLDOJISTAR(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLDOJISTAR)


def CDLDRAGONFLYDOJI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLDRAGONFLYDOJI)


def CDLENGULFING(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLENGULFING)


def CDLEVENINGDOJISTAR(frame, penetration=.3, open_col='open', high_col='high', low_col='low',
                       close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLEVENINGDOJISTAR, penetration)


def CDLEVENINGSTAR(frame, penetration=.3, open_col='open', high_col='high', low_col='low',
                   close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLEVENINGSTAR, penetration)


def CDLGAPSIDESIDEWHITE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLGAPSIDESIDEWHITE)


def CDLGRAVESTONEDOJI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLGRAVESTONEDOJI)


def CDLHAMMER(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHAMMER)


def CDLHANGINGMAN(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHANGINGMAN)


def CDLHARAMI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHARAMI)


def CDLHARAMICROSS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHARAMICROSS)


def CDLHIGHWAVE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHIGHWAVE)


def CDLHIKKAKE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHIKKAKE)


def CDLHIKKAKEMOD(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHIKKAKEMOD)


def CDLHOMINGPIGEON(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLHOMINGPIGEON)


def CDLIDENTICAL3CROWS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLIDENTICAL3CROWS)


def CDLINNECK(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLINNECK)


def CDLINVERTEDHAMMER(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLINVERTEDHAMMER)


def CDLKICKING(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLKICKING)


def CDLKICKINGBYLENGTH(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLKICKINGBYLENGTH)


def CDLLADDERBOTTOM(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLLADDERBOTTOM)


def CDLLONGLEGGEDDOJI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLLONGLEGGEDDOJI)


def CDLLONGLINE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLLONGLINE)


def CDLMARUBOZU(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLMARUBOZU)


def CDLMATCHINGLOW(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLMATCHINGLOW)


def CDLMATHOLD(frame, penetration=.5, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLMATHOLD, penetration)


def CDLMORNINGDOJISTAR(frame, penetration=.3, open_col='open', high_col='high', low_col='low',
                       close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLMORNINGDOJISTAR, penetration)


def CDLMORNINGSTAR(frame, penetration=.3, open_col='open', high_col='high', low_col='low',
                   close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLMORNINGSTAR, penetration)


def CDLONNECK(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLONNECK)


def CDLPIERCING(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLPIERCING)


def CDLRICKSHAWMAN(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLRICKSHAWMAN)


def CDLRISEFALL3METHODS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLRISEFALL3METHODS)


def CDLSEPARATINGLINES(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSEPARATINGLINES)


def CDLSHOOTINGSTAR(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSHOOTINGSTAR)


def CDLSHORTLINE(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSHORTLINE)


def CDLSPINNINGTOP(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSPINNINGTOP)


def CDLSTALLEDPATTERN(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSTALLEDPATTERN)


def CDLSTICKSANDWICH(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLSTICKSANDWICH)


def CDLTAKURI(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLTAKURI)


def CDLTASUKIGAP(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLTASUKIGAP)


def CDLTHRUSTING(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLTHRUSTING)


def CDLTRISTAR(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLTRISTAR)


def CDLUNIQUE3RIVER(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLUNIQUE3RIVER)


def CDLUPSIDEGAP2CROWS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLUPSIDEGAP2CROWS)


def CDLXSIDEGAP3METHODS(frame, open_col='open', high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [open_col, high_col, low_col, close_col], talib.CDLXSIDEGAP3METHODS)


def CMO(series, n=14):
    """chande momentum oscillator"""
    return _series_to_series(series, talib.CMO, n)


def CORREL(frame, col0, col1, n=30):
    return _frame_to_series(frame, [col0, col1], talib.CORREL, n)


def COS(series):
    return _series_to_series(series, talib.COS)


def COSH(series):
    return _series_to_series(series, talib.COSH)


def DEMA(series, n=30):
    """double exponential moving average"""
    return _series_to_series(series, talib.DEMA, n)


def DX(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.DX, n)


def EMA(series, n=30):
    """exponential moving average"""
    return _series_to_series(series, talib.EMA, n)


def EXP(series):
    return _series_to_series(series, talib.EXP)


def FLOOR(series):
    return _series_to_series(series, talib.FLOOR)


def HT_DCPERIOD(series):
    return _series_to_series(series, talib.HT_DCPERIOD)


def HT_DCPHASE(series):
    return _series_to_series(series, talib.HT_DCPHASE)


def HT_PHASOR(series):
    return _series_to_frame(series, ['InPhase', 'Quadrature'], talib.HT_PHASOR)


def HT_SINE(series):
    return _series_to_frame(series, ['Sine', 'LeadSine'], talib.HT_SINE)


def HT_TRENDLINE(series):
    return _series_to_series(series, talib.HT_TRENDLINE)


def HT_TRENDMODE(series):
    return _series_to_series(series, talib.HT_TRENDMODE)


def KAMA(series, n=30):
    """Kaufman Adaptive Moving Average"""
    return _series_to_series(series, talib.KAMA, n)


def LINEARREG(series, n=14):
    return _series_to_series(series, talib.LINEARREG, n)


def LINEARREG_ANGLE(series, n=14):
    return _series_to_series(series, talib.LINEARREG_ANGLE, n)


def LINEARREG_INTERCEPT(series, n=14):
    return _series_to_series(series, talib.LINEARREG_INTERCEPT, n)


def LINEARREG_SLOPE(series, n=14):
    return _series_to_series(series, talib.LINEARREG_SLOPE, n)


def LN(series):
    return _series_to_series(series, talib.LN)


def LOG10(series):
    return _series_to_series(series, talib.LOG10)


def MA(series, n=30, matype=0):
    return _series_to_series(series, talib.MA, n, matype)


def MACD(series, fast=12, slow=26, signal=9):
    return _series_to_frame(series, ['MACD', 'MACD_SIGNAL', 'MACD_HIST'], talib.MACD, fast, slow, signal)


def MAMA(series, fast=.5, slow=.05):
    """MESA Adaptive Moving Average"""
    return _series_to_frame(series, ['MAMA', 'FAMA'], talib.MAMA, fast, slow)


def MAX(series, n=30):
    return _series_to_series(series, talib.MAX, n)


def MEDPRICE(frame, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.MEDPRICE)


def MFI(frame, n=14, high_col='high', low_col='low', close_col='close', vol_col='Volume'):
    """money flow inedx"""
    return _frame_to_series(frame, [high_col, low_col, close_col, vol_col], talib.MFI, n)


def MIDPOINT(series, n=14):
    return _series_to_series(series, talib.MIDPOINT, n)


def MIDPRICE(frame, n=14, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.MIDPRICE, n)


def MINMAX(series, n=30):
    return _series_to_frame(series, ['MIN', 'MAX'], talib.MINMAX, n)


def MIN(series, n=30):
    return _series_to_series(series, talib.MIN, n)


def MINUS_DI(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.MINUS_DI, n)


def MINUS_DM(frame, n=14, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.MINUS_DM, n)


def MOM(series, n=10):
    return _series_to_series(series, talib.MOM, n)


def NATR(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.NATR, n)


def PLUS_DI(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.PLUS_DI, n)


def PLUS_DM(frame, n=14, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.PLUS_DM, n)


def PPO(series, fast=12, slow=26, matype=0):
    return _series_to_series(series, talib.PPO, fast, slow, matype)


def RSI(series, n=14):
    return _series_to_series(series, talib.RSI, n)


def SAR(frame, acc_fator=.02, max_acc_factor=.2, high_col='high', low_col='low'):
    return _frame_to_series(frame, [high_col, low_col], talib.SAR, acc_fator, max_acc_factor)


def SIN(series):
    return _series_to_series(series, talib.SIN)


def SINH(series):
    return _series_to_series(series, talib.SINH)


def STOCH(frame, fastk=5, slowk=3, slowk_matype=0, slowd=3, slowd_matype=0, high_col='high', low_col='low',
          close_col='close'):
    return _frame_to_frame(frame, [high_col, low_col, close_col], ['SlowK', 'SlowD'], talib.STOCH, fastk, slowk,
                           slowk_matype, slowd, slowd_matype)


def STOCHF(frame, fastk=5, fastd=3, fastd_matype=0, high_col='high', low_col='low', close_col='close'):
    return _frame_to_frame(frame, [high_col, low_col, close_col], ['FAST_K', 'FAST_D'], talib.STOCHF, fastk, fastd,
                           fastd_matype)


def STOCHRSI(series, n=14, fastk=5, fastd=3, fastd_matype=0):
    return _series_to_frame(series, ['FAST_K', 'FAST_D'], talib.STOCHRSI, n, fastk, fastd, fastd_matype)


def T3(series, n=5, volume_factor=.7):
    return _series_to_series(series, talib.T3, n, volume_factor)


def TAN(series):
    return _series_to_series(series, talib.TAN)


def TANH(series):
    return _series_to_series(series, talib.TANH)


def TEMA(series, n=5):
    return _series_to_series(series, talib.TEMA, n)


def TRANGE(frame, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.TRANGE)


def TRIMA(series, n=30):
    return _series_to_series(series, talib.TRIMA, n)


def TRIX(series, n=30):
    return _series_to_series(series, talib.TRIX, n)


def TSF(series, n=14):
    return _series_to_series(series, talib.TSF, n)


def WILLR(frame, n=14, high_col='high', low_col='low', close_col='close'):
    return _frame_to_series(frame, [high_col, low_col, close_col], talib.WILLR, n)

