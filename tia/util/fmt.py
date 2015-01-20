"""Format helpers"""
import pandas as pd
import numpy as np


class DateTimeFormatter(object):
    def __init__(self, fmtstr, coerce=True):
        self.fmtstr = fmtstr
        self.coerce = coerce

    def __call__(self, value):
        if not hasattr(value, 'strftime'):
            if self.coerce:
                value = pd.to_datetime(value)
                if not hasattr(value, 'strftime'):
                    raise ValueError('failed to coerce %s type=%s to datetime' % (value, type(value)))
            else:
                raise ValueError('%s type(%s) has not method strftime' % (value, type(value)))
        return value.strftime(self.fmtstr)


class NumberFormatter(object):
    def __init__(self, precision=2, commas=True, parens=True, suffix=None, kind='f', coerce=True,
                 transform=None, nan='nan', prefix=None, lpad_zero=1):
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

    def __call__(self, value, **kwargs):
        if isinstance(value, pd.Series):
            return value.apply(self)
        elif isinstance(value, pd.DataFrame):
            return value.applymap(self)
        elif isinstance(value, (list, tuple)):
            return map(self, value)
        elif not issubclass(type(value), (float, int)):
            if not self.coerce:
                raise ValueError('NumberFormat expected number type not %s' % (type(value)))
            else:
                if self.coerce and not issubclass(type(value), (float, int)):
                    try:
                        value = float(value)
                    except ValueError:
                        raise

        if np.isnan(value):
            return self.nan

        # apply transform
        value = value if self.transform is None else self.transform(value)
        # Build format string
        fmt = '{:' + (self.lpad_zero and '0' or '') + (self.commas and ',' or '') + '.' + str(
            self.precision) + self.kind + '}'
        txt = fmt.format(value)
        if self.parens:
            isneg = txt[0] == '-'
            lp, rp = isneg and ('(', ')') or ('', '')
            txt = isneg and txt[1:] or txt
            return '{prefix}{lp}{txt}{suffix}{rp}'.format(prefix=self.prefix, txt=txt, suffix=self.suffix, lp=lp, rp=rp)
        else:
            return '{prefix}{txt}{suffix}'.format(prefix=self.prefix, txt=txt, suffix=self.suffix)


def new_int_formatter(commas=True, parens=True, prefix=None, suffix=None, coerce=True, nan='nan'):
    precision = 0
    return NumberFormatter(**locals())


def new_float_formatter(precision=2, commas=True, parens=True, prefix=None, suffix=None, coerce=True, nan='nan'):
    return NumberFormatter(**locals())


def new_thousands_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None):
    transform = lambda v: v * 1e-3
    suffix = 'k'
    return NumberFormatter(**locals())


def new_millions_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None):
    transform = lambda v: v * 1e-6
    suffix = 'M'
    return NumberFormatter(**locals())


def new_billions_formatter(precision=1, commas=True, parens=True, nan='nan', prefix=None):
    transform = lambda v: v * 1e-9
    suffix = 'B'
    return NumberFormatter(**locals())


def new_percent_formatter(precision=2, commas=True, parens=True, prefix=None, suffix=None, coerce=True,
                          transform=lambda v: v,
                          nan='nan'):
    kind = '%'
    return NumberFormatter(**locals())


def new_datetime_formatter(fmtstr='%Y-%m-%d', coerce=True):
    return DateTimeFormatter(**locals())


def guess_formatter(values, precision=1, commas=True, parens=True, nan='nan', prefix=None):
    """Based on the values, this determines the best formatter to use without losing too much precision.
    For example if all the values are greater than 1 million but less than 1 billion, than the millions formatter
    will be used"""
    if isinstance(values, pd.Series):
        vmax = values.abs().max()
        vmin = values.abs().min()
    elif isinstance(values, pd.DataFrame):
        vmax = values.abs().max().max()
        vmin = values.abs().min().min()
    elif isinstance(values, (list, tuple)):
        vmax = max(values)
        vmin = min(values)
    else:
        vmax = values
        vmin = values

    if vmin > 10**9 and vmax < 10**12:
        return new_billions_formatter(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix)
    elif vmin > 10**6 and vmax < 10**9:
        return new_millions_formatter(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix)
    elif vmin > 10**3 and vmax < 10**6:
        return new_thousands_formatter(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix)
    elif vmin > .01 and vmax < 1:
        return new_percent_formatter(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix)
    else:
        if isinstance(vmax, int):
            return new_int_formatter(commas=commas, parens=parens, nan=nan, prefix=prefix)
        else:
            return new_float_formatter(precision=precision, commas=commas, parens=parens, nan=nan, prefix=prefix)


# Common Formats
IntFormatter = new_int_formatter()
FloatFormatter = new_float_formatter()
PercentFormatter = new_percent_formatter()
ThousandsFormatter = new_thousands_formatter()
MillionsFormatter = new_millions_formatter()
BillionsFormatter = new_billions_formatter()
DollarCentsFormatter = new_float_formatter(prefix='$')
DollarFormatter = new_int_formatter(prefix='$')
ThousandDollarsFormatter = new_thousands_formatter(prefix='$')
MillionDollarsFormatter = new_millions_formatter(prefix='$')
BillionDollarsFormatter = new_billions_formatter(prefix='$')
YmdFormatter = new_datetime_formatter('%Y%m%d', True)
Y_m_dFormatter = new_datetime_formatter('%Y_%m_%d', True)
