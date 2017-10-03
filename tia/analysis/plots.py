import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tia.analysis.perf import returns_cumulative, max_drawdown, guess_freq
from tia.util.mplot import AxesFormat
from tia.util.fmt import new_float_formatter

def plot_return_on_dollar(rets, title='Return on $1', show_maxdd=0, figsize=None, ax=None, append=0, label=None, **plot_args):
    """ Show the cumulative return of specified rets and max drawdowns if selected."""
    crets = (1. + returns_cumulative(rets, expanding=1))
    if isinstance(crets, pd.DataFrame):
        tmp = crets.copy()
        for c in tmp.columns:
            s = tmp[c]
            fv = s.first_valid_index()
            fi = s.index.get_loc(fv)
            if fi != 0:
                tmp.ix[fi - 1, c] = 1.
            else:
                if not s.index.freq:
                    # no frequency set
                    freq = guess_freq(s.index)
                    s = s.asfreq(freq)
                first = s.index.shift(-1)[0]
                tmp = pd.concat([pd.DataFrame({c: [1.]}, index=[first]), tmp])
        crets = tmp
        if append:
            toadd = crets.index.shift(1)[-1]
            crets = pd.concat([crets, pd.DataFrame(np.nan, columns=crets.columns, index=[toadd])])
    else:
        fv = crets.first_valid_index()
        fi = crets.index.get_loc(fv)
        if fi != 0:
            crets = crets.copy()
            crets.iloc[fi - 1] = 1.
        else:
            if not crets.index.freq:
                first = crets.asfreq(guess_freq(crets.index)).index.shift(-1)[0]
            else:
                first = crets.index.shift(-1)[0]
            tmp = pd.Series([1.], index=[first])
            tmp = tmp.append(crets)
            crets = tmp

        if append:
            toadd = pd.Series(np.nan, index=[crets.index.shift(1)[-1]])
            crets = crets.append(toadd)

    ax = crets.plot(figsize=figsize, title=title, ax=ax, label=label, **plot_args)
    AxesFormat().Y.apply_format(new_float_formatter()).X.label("").apply(ax)
    #ax.tick_params(labelsize=14)
    if show_maxdd:
        # find the max drawdown available by using original rets
        if isinstance(rets, pd.DataFrame):
            iterator = iter(rets.items())
        else:
            iterator = iter([('', rets)])

        for c, col in iterator:
            dd, dt = max_drawdown(col, inc_date=1)
            lbl = c and c + ' maxdd' or 'maxdd'
            # get cret to place annotation correctly
            if isinstance(crets, pd.DataFrame):
                amt = crets.ix[dt, c]
            else:
                amt = crets[dt]

            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
            # sub = lambda c: c and len(c) > 2 and c[:2] or c
            try:
                dtstr = '{0}'.format(dt.to_period())
            except:
                dtstr = '{0}'.format(dt)

            ax.text(dt, amt, "mdd {0}".format(dtstr).strip(), ha="center",
                    va="center", size=10, bbox=bbox_props)
    plt.tight_layout()

