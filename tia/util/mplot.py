"""
Common matplotlib utilities
"""
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas
import uuid
import os
import tia.util.fmt as fmt


class _CustomDateFormatter(DateFormatter):
    """Extend so I can use with pandas Period objects """
    def __call__(self, x, pos=0):
        if not hasattr(x, 'strftime'):
            x = pandas.to_datetime(x)
        x = x.strftime(self.fmt)
        return x


# Define the formatters for direct access
class AxesFormat(object):
    def __init__(self):
        self._deferred = []
        self.axes = None
        self.axis = None  # setup by subclass
        self.active = None

    @property
    def isX(self):
        return self.active == 'x'

    @property
    def isY(self):
        return self.active == 'y'

    def __getattribute__(self, name):
        attr = super(AxesFormat, self).__getattribute__(name)
        if callable(attr) and not name.startswith('_'):
            def wrapped(*args, **kwargs):
                self._deferred.append(lambda: attr(*args, **kwargs))
                return self

            return wrapped
        else:
            return attr

    def __call__(self, axes=None):
        self.axes = axes or plt.gca()
        [f() for f in self._deferred]

    def Y(self):
        self.active = 'y'
        self.axis = self.axes.yaxis
        return self

    def X(self):
        self.active = 'x'
        self.axis = self.axes.xaxis
        return self

    def percent(self, precision=2):
        fct = fmt.new_percent_formatter(precision=precision)
        wrapper = lambda x, pos: fct(x)
        self.axis.set_major_formatter(FuncFormatter(wrapper))
        return self

    def thousands(self, precision=1):
        fct = fmt.new_thousands_formatter(precision=precision)
        wrapper = lambda x, pos: fct(x)
        self.axis.set_major_formatter(FuncFormatter(wrapper))
        return self

    def millions(self, precision=1):
        fct = fmt.new_millions_formatter(precision=precision)
        wrapper = lambda x, pos: fct(x)
        self.axis.set_major_formatter(FuncFormatter(wrapper))
        return self

    def date(self, fmt='%Y-%m-%d'):
        fmtfct = DateFormatter(fmt)
        self.axis.set_major_formatter(fmtfct)
        return self

    def apply_format(self, fmtfct=lambda x: x):
        wrapper = lambda x, pos: fmtfct(x)
        self.axis.set_major_formatter(FuncFormatter(wrapper))
        return self

    def label(self, txt):
        self.isX and self.axes.set_xlabel(txt) or self.axes.set_ylabel(txt)
        return self

    def rotate(self, rot=40, ha='right'):
        rotate_labels(self.axes, which=self.isX and 'x' or 'y', rot=rot, ha=ha)
        return self

    def tight_layout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        plt.tight_layout(pad, h_pad, w_pad, rect)
        return self


class FigureHelper(object):
    def __init__(self, basedir=None, ext='.pdf'):
        if not basedir:
            import tempfile
            basedir = tempfile.gettempdir()
        self.basedir = basedir
        self.last = None
        self.ext = ext
        self.fnmap = {}

        self.ax = None
        self.axiter = None
        self.figure = None

    def keys(self):
        return self.fnmap.keys()

    def next_ax(self):
        self.ax = self.axiter.next()
        return self.ax

    def __getitem__(self, item):
        return self.fnmap[item]

    def savefig(self, fn=None, dpi=100, clear=1, ext=None, key=None):
        ext = ext or self.ext
        ext = ext.startswith('.') and ext or '.' + ext
        fn = fn or uuid.uuid1()
        key = key or ''
        fn = '%s%s%s' % (key, fn, ext)
        fn = os.path.join(self.basedir, fn)
        figure = self.figure or plt.gcf()
        figure.savefig(fn, dpi=dpi)
        if clear:
            figure.clf()
        if key:
            self.fnmap[key] = fn
        self.last = fn
        return fn

    def subplots(self, *params, **kwargs):
        f, ax = plt.subplots(*params, **kwargs)
        def axes_iter(axes):
            if not hasattr(axes, '__iter__'):
                return iter(list([axes]))
            else:
                if not hasattr(axes[0], '__iter__'):
                    return iter(axes)
                else:
                    # array of arrays
                    return iter([y for x in axes for y in x])

        self.axiter = axes_iter(ax)
        self.figure = f
        return self.next_ax()


def rotate_labels(ax, which='x', rot=40, ha='right'):
    which = which.upper()

    def _apply(lbls):
        for lbl in lbls:
            lbl.set_ha(ha)
            lbl.set_rotation(rot)

    'X' in which and _apply(ax.get_xticklabels())
    'Y' in which and _apply(ax.get_yticklabels())


class GridHelper(object):
    @staticmethod
    def build(numobjs, ncols, **subplot_kwargs):
        nrows = int(np.ceil(float(numobjs) / float(ncols)))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subplot_kwargs)

        if nrows == 1:
            axes = [axes]
        if ncols == 1:
            axes = [[ax] for ax in axes]
        return GridHelper(axes, nrows, ncols, fig=fig)

    def __init__(self, axarr, nrows, ncols, fig=None):
        self.axarr = axarr
        self.nrows = nrows
        self.ncols = ncols
        self.fig = fig

    def __iter__(self):
        import itertools

        flat = list(itertools.chain.from_iterable(self.axarr))
        return iter(flat)

    def get_axes(self, idx):
        """ Allow for simple indexing """
        cidx = 0
        if idx > 0:
            cidx = idx % self.ncols
        ridx = idx / self.ncols
        return self.axarr[ridx][cidx]

    def get_last_row(self):
        return self.axarr[self.nrows - 1]

    def get_first_col(self):
        """ Return the array of Axes objects for the first column """
        return [ax[0] for ax in self.axarr]
