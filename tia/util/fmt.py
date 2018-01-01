"""Format helpers"""
import math

import pandas as pd
import pandas.lib as lib
import numpy as np

pd_is_datetime_arraylike = None
try:
    from pandas.core.common import is_datetime_arraylike as pd_is_datetime_arraylike
except:
    pass

from functools import partial


def is_datetime_arraylike(arr):
    if isinstance(arr, pd.DataFrame):
        return arr.apply(pd_is_datetime_arraylike).all()
    elif pd_is_datetime_arraylike is not None:
        return pd_is_datetime_arraylike(arr)
    elif isinstance(arr, pd.DatetimeIndex):
        return True
    else:
        inferred = lib.infer_dtype(arr)
        return 'datetime' in inferred


class DateTimeFormat(object):
    def __init__(self, fmtstr, coerce=True):
        self.fmtstr = fmtstr
        self.coerce = coerce

    def __call__(self, value):
        if isinstance(value, pd.Series):
            return value.apply(self.__call__)
        else:
            if not hasattr(value, 'strftime'):
                if self.coerce:
                    value = pd.to_datetime(value)
                    if not hasattr(value, 'strftime'):
                        raise ValueError('failed to coerce %s type=%s to datetime' % (value, type(value)))
                else:  #
                    raise ValueError('%s type(%s) has not method strftime' % (value, type(value)))
            return (value == value and value.strftime(self.fmtstr)) or str(value)


class NumberFormat(object):
    def __init__(self, precision=2, commas=True, parens=True, suffix=None, kind='f', coerce=True,
                 transform=None, nan='nan', prefix=None, lpad_zero=1, do_raise=0, trunc_dot_zeros=0):
        """
        Parameters
        ----------
        precision : int, defaults to 2
                    Number of decimals places to show
        commas : bool, default to True
                    If true then show commas, else do not
        parens : bool, default to True
                    If True then use parenthesis for showing negative numbers
        suffix:
        kind:
        coerce:
        transform:
        nan:
        prefix:
        lpad_zero:
        do_raise:
        trunc_dot_zeros: bool, default to false
                        if True and precision is greater than 0, a number such as 3.0 will be returned as just 3
        """
        self.transform = transform
        self.coerce = coerce
        # build format string
        self.precision = precision
        self.commas = commas
        self.parens = parens
        self.suffix = suffix or ''
        self.prefix = prefix or ''
        self.kind = kind
        self.nan = nan
        self.lpad_zero = lpad_zero
        self.do_raise = do_raise
        self.trunc_dot_zeros = trunc_dot_zeros

    def __call__(self, value, **kwargs):
        # apply any overrides
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self_with_args = partial(self.__call__, **kwargs)

        if isinstance(value, pd.Series):
            return value.apply(self_with_args)
        elif isinstance(value, pd.DataFrame):
            return value.applymap(self_with_args)
        elif isinstance(value, (list, tuple)):
            return list(map(self_with_args, value))
        elif isinstance(value, np.ndarray):
            if value.ndim == 2:
                return self_with_args(pd.DataFrame(value)).values
            elif value.ndim == 1:
                return self_with_args(pd.Series(value)).values
        elif not issubclass(type(value), (float, int)):
            if not self.coerce:
                raise ValueError('NumberFormat expected number type not %s' % (type(value)))
            else:
                if self.coerce and not issubclass(type(value), (float, int)):
                    try:
                        value = float(value)
                    except ValueError:
                        if self.do_raise:
                            raise
                        else:
                            # return the value without doing anything
                            return value

        if np.isnan(value):
            return self.nan

        # apply transform
        value = value if self.transform is None else self.transform(value)
        # Build format string
        fmt = '{:' + (self.lpad_zero and '0' or '') + (self.commas and ',' or '') + '.' + str(
            self.precision) + self.kind + '}'
        txt = fmt.format(value)
        if self.precision > 0 and self.trunc_dot_zeros:
            txt = txt.replace('.' + '0' * self.precision, '')

        if self.parens:
            isneg = txt[0] == '-'
            lp, rp = isneg and ('(', ')') or ('', '')
            txt = isneg and txt[1:] or txt
            return '{prefix}{lp}{txt}{suffix}{rp}'.format(prefix=self.prefix, txt=txt, suffix=self.suffix, lp=lp, rp=rp)
        else:
            return '{prefix}{txt}{suffix}'.format(prefix=self.prefix, txt=txt, suffix=self.suffix)


def new_int_formatter(commas=True, parens=True, prefix=None, suffix=None, coerce=True, nan='nan', trunc_dot_zeros=0):
    precision = 0
    return NumberFormat(**locals())


def new_float_formatter(precision=2, commas=True, parens=True, prefix=None, suffix=None, coerce=True, nan='nan',
                        trunc_dot_zeros=0):
    return NumberFormat(**locals())


def new_thousands_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None, trunc_dot_zeros=0,
                            suffix='k'):
    transform = lambda v: v * 1e-3
    return NumberFormat(**locals())


def new_millions_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None, trunc_dot_zeros=0,
                           suffix='M'):
    transform = lambda v: v * 1e-6
    return NumberFormat(**locals())


def new_billions_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None, trunc_dot_zeros=0,
                           suffix='B'):
    transform = lambda v: v * 1e-9
    return NumberFormat(**locals())


def new_trillions_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None, trunc_dot_zeros=0):
    transform = lambda v: v * 1e-12
    suffix = 'T'
    return NumberFormat(**locals())


def new_percent_formatter(precision=2, commas=True, parens=True, prefix=None, suffix=None, coerce=True,
                          transform=lambda v: v,
                          nan='nan', trunc_dot_zeros=0):
    kind = '%'
    return NumberFormat(**locals())


def new_datetime_formatter(fmtstr='%d-%b-%y', coerce=True):
    return DateTimeFormat(**locals())


def guess_formatter(values, precision=1, commas=True, parens=True, nan='nan', prefix=None, pcts=0,
                    trunc_dot_zeros=0):
    """Based on the values, return the most suitable formatter
    Parameters
    ----------
    values : Series, DataFrame, scalar, list, tuple, or ndarray
             Values used to determine which formatter is the best fit
    """
    formatter_args = dict(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix,
                          trunc_dot_zeros=trunc_dot_zeros)

    try:
        if isinstance(values, pd.datetime) and values.hour == 0 and values.minute == 0:
            return new_datetime_formatter()
        elif is_datetime_arraylike(values):
            # basic date formatter if no hours or minutes
            if hasattr(values, 'dt'):
                if (values.dt.hour == 0).all() and (values.dt.minute == 0).all():
                    return new_datetime_formatter()
            elif isinstance(values, pd.Series):
                if values.dropna().apply(lambda d: d.hour == 0).all() and values.apply(lambda d: d.minute == 0).all():
                    return new_datetime_formatter()
            elif isinstance(values, pd.DataFrame):
                if values.dropna().applymap(lambda d: d != d or (d.hour == 0 and d.minute == 0)).all().all():
                    return new_datetime_formatter()

        elif isinstance(values, pd.Series):
            aval = values.abs()
            vmax, vmin = aval.max(), aval.min()
        elif isinstance(values, np.ndarray):
            if values.ndim == 2:
                avalues = pd.DataFrame(values).abs()
                vmax = avalues.max().max()
                vmin = avalues.min().min()
            elif values.ndim == 1:
                aval = pd.Series(values).abs()
                vmax, vmin = aval.max(), aval.min()
            else:
                raise ValueError('cannot accept frame with more than 2-dimensions')
        elif isinstance(values, pd.DataFrame):
            avalues = values.abs()
            vmax = avalues.max().max()
            vmin = avalues.min().min()
        elif isinstance(values, (list, tuple)):
            vmax = max(values)
            vmin = min(values)
        else:
            vmax = vmin = abs(values)

        if np.isnan(vmin):
            return new_float_formatter(**formatter_args)
        else:
            min_digits = 0 if vmin == 0 else math.floor(math.log10(vmin))
            # max_digits = math.floor(math.log10(vmax))
            if min_digits >= 12:
                return new_trillions_formatter(**formatter_args)
            elif min_digits >= 9:
                return new_billions_formatter(**formatter_args)
            elif min_digits >= 6:
                return new_millions_formatter(**formatter_args)
            elif min_digits >= 3:
                return new_thousands_formatter(**formatter_args)
            elif pcts and min_digits < 0 and vmax < 1:
                return new_percent_formatter(**formatter_args)
            else:
                if isinstance(vmax, int):
                    formatter_args.pop('precision')
                    return new_int_formatter(**formatter_args)
                else:
                    return new_float_formatter(**formatter_args)
    except:
        # import sys
        # e = sys.exc_info()[0]
        return lambda x: x


class DynamicNumberFormat(object):
    def __init__(self, method=None, **formatter_args):
        """
        :param method: None, cell, col
        :param formatter_args:
        :return:
        """
        if method and method not in ('cell', 'col', 'row'):
            raise ValueError('method must be None, cell, row, or col')
        self.formatter_args = formatter_args
        self.method = method


    def __call__(self, value, **kwargs):
        for k in list(kwargs.keys()):
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
                kwargs.pop(k)
        method = self.method

        self_with_args = partial(self.__call__, **kwargs)

        if method is not None and isinstance(value, pd.DataFrame):
            if method == 'cell':
                return value.applymap(self_with_args)
            elif method == 'row':
                return value.T.apply(self_with_args).T
            else:
                return value.apply(self_with_args)
        elif method == 'cell' and isinstance(value, pd.Series):
            return value.apply(self_with_args)
        else:
            return guess_formatter(value, **self.formatter_args)(value, **kwargs)


def new_dynamic_formatter(method=None, precision=1, commas=True, parens=True, nan='nan', prefix=None, pcts=0,
                          trunc_dot_zeros=0):
    return DynamicNumberFormat(**locals())


# Common Formats
IntFormatter = new_int_formatter()
FloatFormatter = new_float_formatter()
PercentFormatter = new_percent_formatter()
ThousandsFormatter = new_thousands_formatter()
MillionsFormatter = new_millions_formatter()
BillionsFormatter = new_billions_formatter()
TrillionsFormatter = new_trillions_formatter()
DollarCentsFormatter = new_float_formatter(prefix='$')
DollarFormatter = new_int_formatter(prefix='$')
ThousandDollarsFormatter = new_thousands_formatter(prefix='$')
MillionDollarsFormatter = new_millions_formatter(prefix='$')
BillionDollarsFormatter = new_billions_formatter(prefix='$')
TrillionDollarsFormatter = new_trillions_formatter(prefix='$')
YmdFormatter = new_datetime_formatter('%Y%m%d', True)
Y_m_dFormatter = new_datetime_formatter('%Y_%m_%d', True)
DynamicNumberFormatter = DynamicNumberFormat(method='col', pcts=1, trunc_dot_zeros=1)
DynamicRowFormatter = DynamicNumberFormat(method='row', pcts=1, trunc_dot_zeros=1)
DynamicColumnFormatter = DynamicNumberFormat(method='col', pcts=1, trunc_dot_zeros=1)
DynamicCellFormatter = DynamicNumberFormat(method='cell', pcts=1, trunc_dot_zeros=1)
