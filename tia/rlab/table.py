from reportlab.platypus import Table, TableStyle, Flowable
from reportlab.lib import colors
from matplotlib.colors import rgb2hex, LinearSegmentedColormap
from matplotlib.pyplot import get_cmap
import numpy as np
import pandas as pd

from tia.rlab.components import KeepInFrame
import tia.util.fmt as fmt


__all__ = ['ConditionalRedBlack', 'DynamicTable', 'TableFormatter', 'RegionFormatter', 'IntFormatter', 'FloatFormatter',
           'PercentFormatter', 'ThousandsFormatter', 'MillionsFormatter', 'BillionsFormatter', 'DollarCentsFormatter',
           'DollarFormatter', 'ThousandDollarsFormatter', 'MillionDollarsFormatter', 'BillionDollarsFormatter',
           'YmdFormatter', 'Y_m_dFormatter']

DefaultHeaderStyle = {
    "GRID": (.5, colors.grey), "BOX": (.25, colors.black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 3, "TOPPADDING": 3, "FONTSIZE": 6, "BACKGROUND": colors.HexColor("#404040"),
    "FONTNAME": "Helvetica", "ALIGN": "CENTER", "TEXTCOLOR": colors.white
}

DefaultCellStyle = {
    "GRID": (.5, colors.grey), "BOX": (.25, colors.black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 2, "TOPPADDING": 2, "ALIGN": "CENTER", "TEXTCOLOR": colors.black,
    "ROWBACKGROUNDS": [[colors.HexColor("#e3ebf4"), colors.white]], "FONTSIZE": 6, "FONTNAME": "Courier"
}

DefaultIndexStyle = {
    "GRID": (.5, colors.grey), "BOX": (.25, colors.black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 2, "TOPPADDING": 2, "ALIGN": "RIGHT", "TEXTCOLOR": colors.black,
    "ROWBACKGROUNDS": [[colors.HexColor("#e3ebf4"), colors.white]], "FONTSIZE": 6, "FONTNAME": "Helvetica"
}

AlignRight = {'ALIGN': 'RIGHT'}

ConditionalRedBlack = lambda x: x < 0 and dict(TEXTCOLOR=colors.HexColor("#800000"))


def pad_positive_wrapper(fmtfct):
    """Ensure that numbers are aligned in table by appending a blank space to postive values if 'parenthesis' are
    used to denote negative numbers"""

    def check_and_append(*args, **kwargs):
        result = fmtfct(*args, **kwargs)
        if fmtfct.parens and not result.endswith(')'):
            result += ' '
        return result

    return check_and_append


IntFormatter = pad_positive_wrapper(fmt.new_int_formatter(nan='-'))
FloatFormatter = pad_positive_wrapper(fmt.new_float_formatter(nan='-'))
PercentFormatter = pad_positive_wrapper(fmt.new_percent_formatter(nan='-'))
ThousandsFormatter = pad_positive_wrapper(fmt.new_thousands_formatter(nan='-'))
MillionsFormatter = pad_positive_wrapper(fmt.new_millions_formatter(nan='-'))
BillionsFormatter = pad_positive_wrapper(fmt.new_billions_formatter(nan='-'))
DollarCentsFormatter = pad_positive_wrapper(fmt.new_float_formatter(prefix='$', nan='-'))
DollarFormatter = pad_positive_wrapper(fmt.new_int_formatter(prefix='$', nan='-'))
ThousandDollarsFormatter = pad_positive_wrapper(fmt.new_thousands_formatter(prefix='$', nan='-'))
MillionDollarsFormatter = pad_positive_wrapper(fmt.new_millions_formatter(prefix='$', nan='-'))
BillionDollarsFormatter = pad_positive_wrapper(fmt.new_billions_formatter(prefix='$', nan='-'))
YmdFormatter = fmt.new_datetime_formatter('%Y%m%d', True)
Y_m_dFormatter = fmt.new_datetime_formatter('%Y_%m_%d', True)


class DynamicTable(Table):
    def __init__(self, data, on_wrap=None, **kwargs):
        self.on_wrap = on_wrap
        Table.__init__(self, data, **kwargs)
        self._longTableOptimize = 0

    def wrap(self, awidth, aheight):
        self.on_wrap and self.on_wrap(self, awidth, aheight)
        return Table.wrap(self, awidth, aheight)


def is_contiguous(idx):
    if len(idx) > 0:
        s0, s1 = idx.min(), idx.max()
        expected = pd.Int64Index(np.array(list(range(s0, s1 + 1))))
        # return idx.isin(expected).all()
        return expected.isin(idx).all()


def find_locations(index, match_value_or_fct, levels=None, max_matches=0):
    matches = []
    fct = match_value_or_fct
    if not callable(fct):
        fct = lambda v: v == match_value_or_fct

    for lvl, loc, val in level_iter(index, levels):
        if fct(val):
            matches.append(loc)
            if max_matches and len(matches) >= matches:
                break
    return matches


def level_iter(index, levels=None):
    if levels is None:
        levels = range(index.nlevels)
    elif np.isscalar(levels):
        levels = [levels]

    for level in levels:
        for i, v in enumerate(index.get_level_values(level)):
            yield level, i, v


def span_iter(series):
    sorted = series.sort_index()
    isnull = pd.isnull(sorted).values
    isnulleq = isnull[1:] & isnull[:-1]
    iseq = sorted.values[1:] == sorted.values[:-1]
    eq = isnulleq | iseq
    li = 0
    for i in range(len(eq)):
        islast = i == (len(eq) - 1)
        if eq[i]:  # if currently true then consecutive
            if islast or not eq[i + 1]:
                yield sorted.index[li], sorted.index[i + 1]
        else:
            li = i + 1
    raise StopIteration


class RegionFormatter(object):
    def __init__(self, parent, row_ilocs, col_ilocs):
        self.row_ilocs = row_ilocs
        self.col_ilocs = col_ilocs
        self.parent = parent
        self.style_cmds = parent.style_cmds
        self.is_contiguous_rows = isrcont = is_contiguous(row_ilocs)
        self.is_contiguous_cols = isccont = is_contiguous(col_ilocs)
        self.iloc = _RegionIX(self, 'iloc')

        # Build coord arrays for easy iteration
        if isccont:
            self.col_coord_tuples = [(col_ilocs.min(), col_ilocs.max())]
        else:
            self.col_coord_tuples = list(zip(col_ilocs, col_ilocs))

        if isrcont:
            self.row_coord_tuples = [(row_ilocs.min(), row_ilocs.max())]
        else:
            self.row_coord_tuples = list(zip(row_ilocs, row_ilocs))

    @property
    def nrows(self):
        return len(self.row_ilocs)

    @property
    def ncols(self):
        return len(self.col_ilocs)

    @property
    def formatted_values(self):
        return self.parent.formatted_values.iloc[self.row_ilocs, self.col_ilocs]

    @property
    def actual_values(self):
        return self.parent.actual_values.iloc[self.row_ilocs, self.col_ilocs]

    def new_instance(self, local_row_idxs, local_col_idxs):
        rows = pd.Int64Index([self.row_ilocs[r] for r in local_row_idxs])
        cols = pd.Int64Index([self.col_ilocs[c] for c in local_col_idxs])
        return RegionFormatter(self.parent, rows, cols)

    def empty_frame(self):
        return self.new_instance([], [])

    def match_column_labels(self, match_value_or_fct, levels=None, max_matches=0, empty_res=1):
        """Check the original DataFrame's column labels to find a subset of the current region
        :param match_value_or_fct: value or function(hdr_value) which returns True for match
        :param levels: [None, scalar, indexer]
        :param max_matches: maximum number of columns to return
        :return:
        """
        allmatches = self.parent._find_column_label_positions(match_value_or_fct, levels)
        # only keep matches which are within this region
        matches = [m for m in allmatches if m in self.col_ilocs]
        if max_matches and len(matches) > max_matches:
            matches = matches[:max_matches]

        if matches:
            return RegionFormatter(self.parent, self.row_ilocs, pd.Int64Index(matches))
        elif empty_res:
            return self.empty_frame()

    def match_row_labels(self, match_value_or_fct, levels=None, max_matches=0, empty_res=1):
        """Check the original DataFrame's row labels to find a subset of the current region
        :param match_value_or_fct: value or function(hdr_value) which returns True for match
        :param levels: [None, scalar, indexer]
        :param max_matches: maximum number of columns to return
        :return:
        """
        allmatches = self.parent._find_row_label_positions(match_value_or_fct, levels)
        # only keep matches which are within this region
        matches = [m for m in allmatches if m in self.row_ilocs]
        if max_matches and len(matches) > max_matches:
            matches = matches[:max_matches]

        if matches:
            return RegionFormatter(self.parent, pd.Int64Index(matches), self.col_ilocs)
        elif empty_res:
            return self.empty_frame()

    def match_any_labels(self, match_value_or_fct, levels=None, max_matches=0, empty_res=1):
        res = self.match_column_labels(match_value_or_fct, levels, max_matches, empty_res=0)
        res = res or self.match_row_labels(match_value_or_fct, levels, max_matches, empty_res)
        return res

    def __repr__(self):
        return repr(self.formatted_values)

    def apply_style(self, cmd, *args):
        """
        Apply the specified style cmd to this region. For example, set all fonts to size 12, apply_style('FONTSIZE', 12)
        :param cmd: reportlab  format command
        :param args: arguments for the cmd
        :return: self
        """
        for c0, c1 in self.col_coord_tuples:
            for r0, r1 in self.row_coord_tuples:
                cmd = [cmd, (c0, r0), (c1, r1)] + list(args)
                self.style_cmds.append(cmd)
        return self

    def apply_styles(self, cmdmap):
        """
        Apply the set of commands defined in cmdmap. for example, apply_styles({'FONTSIZE': 12, 'BACKGROUND': white})
        :param cmdmap: dict of commands mapped to the command arguments
        :return: self
        """
        for cmd, args in cmdmap.iteritems():
            if not isinstance(args, (list, tuple)):
                args = [args]
            self.apply_style(cmd, *args)
        return self

    def apply_conditional_styles(self, cbfct):
        """
        Ability to provide dynamic styling of the cell based on its value.
        :param cbfct: function(cell_value) should return a dict of format commands to apply to that cell
        :return: self
        """
        for ridx in range(self.nrows):
            for cidx in range(self.ncols):
                fmts = cbfct(self.actual_values.iloc[ridx, cidx])
                fmts and self.iloc[ridx, cidx].apply_styles(fmts)
        return self

    def detect_colspans(self, use_actual=1):
        """Determine if any col spans are present in the values.
        :param use_actual:  if True, check actual_values for span. if False, use the formatted_values
        :return: self
        """
        vals = self.actual_values if use_actual else self.formatted_values
        if self.is_contiguous_cols:
            for ridx in range(self.nrows):
                for c0, c1 in span_iter(vals.iloc[ridx, :]):
                    actual_idx = self.row_ilocs[ridx]
                    self.style_cmds.append(['SPAN', (c0, actual_idx), (c1, actual_idx)])
        return self

    def detect_rowspans(self, use_actual=1):
        """Determine if any row spans are present in the values.
        :param use_actual:  if True, check actual_values for span. if False, use the formatted_values
        :return: self
        """
        """ Determine if any row spans are present"""
        vals = self.actual_values if use_actual else self.formatted_values
        if self.is_contiguous_rows:
            for cidx in range(self.ncols):
                for r0, r1 in span_iter(vals.iloc[:, cidx]):
                    actual_idx = self.col_ilocs[cidx]
                    self.style_cmds.append(['SPAN', (actual_idx, r0), (actual_idx, r1)])
        return self

    def detect_spans(self, colspans=1, rowspans=1, use_actual=1):
        colspans and self.detect_colspans(use_actual)
        rowspans and self.detect_rowspans(use_actual)

    def apply_format(self, fmtfct):
        """
        For each cell in the region, invoke fmtfct(cell_value) and store result in the formatted_values
        :param fmtfct: function(cell_value) which should return a formatted value for display
        :return: self
        """
        for ridx in range(self.nrows):
            for cidx in range(self.ncols):
                # MUST set the parent as local view is immutable
                riloc = self.row_ilocs[ridx]
                ciloc = self.col_ilocs[cidx]
                self.parent.formatted_values.iloc[riloc, ciloc] = fmtfct(self.actual_values.iloc[ridx, cidx])
        return self

    def apply_rowattrs(self, **kwargs):
        for k, v in kwargs.iteritems():
            self.parent.rowattrs.iloc[self.row_ilocs, k] = v
        return self

    def apply_colattrs(self, **kwargs):
        for k, v in kwargs.iteritems():
            self.parent.colattrs.loc[self.col_ilocs, k] = v
        return self

    def apply(self, **kwargs):
        """
        Accepts the following keys:
        'styles': see apply_styles for args
        'cstyles': see apply_condition_styles for args
        'format': see apply_format for args
        'c': col width (array or scalar)
        'cmin': min col width (array or scalar)
        'cmax': max col width (array or scalar)
        'cweight: col weight use at runtime to determine width
        'r': row height (array or scalar)
        'rmin': min row height (array or scalar)
        'rmax': max row height (array or scalar)
        'rweight: row weight use at runtime to determine height
        'cspans': detect colspans
        'rspans': detect rowspans
        'spans': bool, detetch both rowspans and colspans

        @param kwargs:
        @return:
        """

        def _apply_if_avail(key, fct):
            if key in kwargs:
                val = kwargs.pop(key)
                if val is not None:
                    fct(val)

        _apply_if_avail('styles', lambda v: self.apply_styles(v))
        _apply_if_avail('cstyles', lambda v: self.apply_conditional_styles(v))
        _apply_if_avail('format', lambda v: self.apply_format(v))
        _apply_if_avail('c', lambda v: self.apply_colattrs(value=v))
        _apply_if_avail('cmin', lambda v: self.apply_colattrs(min=v))
        _apply_if_avail('cmax', lambda v: self.apply_colattrs(max=v))
        _apply_if_avail('cweight', lambda v: self.apply_colattrs(weight=v))
        _apply_if_avail('r', lambda v: self.apply_rowattrs(value=v))
        _apply_if_avail('rmin', lambda v: self.apply_rowattrs(min=v))
        _apply_if_avail('rmax', lambda v: self.apply_rowattrs(max=v))
        _apply_if_avail('rweight', lambda v: self.apply_rowattrs(weight=v))
        _apply_if_avail('rspans', lambda v: v and self.detect_rowspans())
        _apply_if_avail('cspans', lambda v: v and self.detect_colspans())
        _apply_if_avail('spans', lambda v: v and (self.detect_rowspans(), self.detect_colspans()))

    def apply_number_format(self, formatter, rb=1, align=1):
        styles = align and AlignRight or {}
        cstyles = rb and ConditionalRedBlack or None
        self.apply(format=formatter, styles=styles, cstyles=cstyles)
        return self

    # def heat_map(self, cmap=None, min=None, max=None, font_cmap=None):
    def heat_map(self, cmap='RdYlGn', vmin=None, vmax=None, font_cmap=None):
        if cmap is None:
            carr = ['#d7191c', '#fdae61', '#ffffff', '#a6d96a', '#1a9641']
            cmap = LinearSegmentedColormap.from_list('default-heatmap', carr)

        if isinstance(cmap, basestring):
            cmap = get_cmap(cmap)
        if isinstance(font_cmap, basestring):
            font_cmap = get_cmap(font_cmap)

        vals = self.actual_values.astype(float)
        if vmin is None:
            vmin = vals.min().min()
        if vmax is None:
            vmax = vals.max().max()
        norm = (vals - vmin) / (vmax - vmin)
        for ridx in range(self.nrows):
            for cidx in range(self.ncols):
                v = norm.iloc[ridx, cidx]
                if np.isnan(v):
                    continue
                color = cmap(v)
                hex = rgb2hex(color)
                styles = {'BACKGROUND': colors.HexColor(hex)}
                if font_cmap is not None:
                    styles['TEXTCOLOR'] = colors.HexColor(rgb2hex(font_cmap(v)))
                self.iloc[ridx, cidx].apply_styles(styles)
        return self


class _RegionIX(object):
    """ Custom version of indexer which ensures a DataFrame is created for proper use with the  RangeFormatter"""

    def __init__(self, region, idx_fct_name='iloc'):
        self.region = region
        self.idx_fct_name = idx_fct_name

    def __getitem__(self, key):
        """Sloppy implementation as I do not handle nested tuples properly"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise Exception('if tuple is used, it must contain 2 indexers')
            ridx = key[0]
            cidx = key[1]
        else:
            ridx = key
            cidx = slice(None)

        region = self.region
        ridx = [ridx] if np.isscalar(ridx) else ridx
        cidx = [cidx] if np.isscalar(cidx) else cidx
        idx = getattr(region.formatted_values, self.idx_fct_name)
        result = idx[ridx, cidx]
        if not isinstance(result, pd.DataFrame):
            raise Exception('index %s is expected to return a DataFrame, not %s' % (key, type(result)))
        return RegionFormatter(self.region.parent, result.index, result.columns)


class TableFormatter(object):
    def __init__(self, df, inc_header=1, inc_index=1):
        self.df = df
        self.inc_header = inc_header
        self.inc_index = inc_index
        self.ncols = ncols = len(df.columns)
        self.nrows = nrows = len(df.index)
        self.nhdrs = nhdrs = inc_header and df.columns.nlevels or 0
        self.nidxs = nidxs = inc_index and df.index.nlevels or 0
        self.style_cmds = []

        # copy the actual values to the formatted cells
        values = df.reset_index(drop=not inc_index).T.reset_index(drop=not inc_header).T.reset_index(drop=True)
        if inc_index and nhdrs > 1:  # move index name down
            values.iloc[nhdrs - 1, :nidxs] = values.iloc[0, :nidxs]
            values.iloc[:nhdrs - 1, :nidxs] = ''

        formatted_values = pd.DataFrame(np.empty((nhdrs + nrows, nidxs + ncols), dtype=object))
        formatted_values.ix[:, :] = values.copy().values
        self.actual_values = values
        self.formatted_values = formatted_values
        self.named_regions = {
            'ALL': RegionFormatter(self, formatted_values.index, formatted_values.columns),
            'HEADER': RegionFormatter(self, formatted_values.index[:nhdrs], formatted_values.columns[nidxs:]),
            'INDEX': RegionFormatter(self, formatted_values.index[nhdrs:], formatted_values.columns[:nidxs]),
            'CELLS': RegionFormatter(self, formatted_values.index[nhdrs:], formatted_values.columns[nidxs:]),
            'INDEX_HEADER': RegionFormatter(self, formatted_values.index[:nhdrs], formatted_values.columns[:nidxs]),
        }

        # Define some fields to handle weight of rows/columns
        self.rowattrs = pd.DataFrame(np.empty((nhdrs + nrows, 4)), columns=['weight', 'min', 'max', 'value'])
        self.rowattrs[:] = np.nan
        self.colattrs = pd.DataFrame(np.empty((nidxs + ncols, 4)), columns=['weight', 'min', 'max', 'value'])
        self.colattrs[:] = np.nan

    def __getitem__(self, name):
        return self.named_regions[name]

    def get_default_header_style(self, **overrides):
        return dict(DefaultHeaderStyle, **overrides)

    def apply_default_cell_style(self, **overrides):
        styles = dict(DefaultCellStyle, **overrides)
        self.cells.apply_styles(styles)
        return self

    def apply_default_header_style(self, **overrides):
        styles = self.get_default_header_style(**overrides)
        self.header.apply_styles(styles)
        return self

    def apply_default_index_style(self, **overrides):
        styles = dict(DefaultIndexStyle, **overrides)
        self.index.apply_styles(styles)
        return self

    def apply_default_style(self, inc_cells=1, inc_header=1, inc_index=1, cells_override=None, header_override=None,
                            index_override=None):
        inc_cells and self.apply_default_cell_style(**(cells_override or {}))
        inc_header and self.apply_default_header_style(**(header_override or {}))
        inc_index and self.apply_default_index_style(**(index_override or {}))
        return self

    @property
    def all(self):
        return self['ALL']

    @property
    def header(self):
        return self['HEADER']

    @property
    def index(self):
        return self['INDEX']

    @property
    def index_header(self):
        return self['INDEX_HEADER']

    @property
    def cells(self):
        return self['CELLS']

    def set_row_heights(self, pcts=None, amts=None, maxs=None, mins=None):
        """
        :param pcts: the percent of available height to use or ratio is also ok
        :param amts: (Array or scalar) the fixed height of the rows
        :param maxs: (Array or scalar) the maximum height of the rows (only use when pcts is used)
        :param mins: (Array or scalar) the minimum height of the rows (only used when pcts is used)
        :return:
        """
        for arr, attr in zip([pcts, amts, maxs, mins], ['weight', 'value', 'max', 'min']):
            if arr is not None:
                if not np.isscalar(arr):
                    if len(arr) != len(self.formatted_values.index):
                        raise ValueError(
                            '%s: expected %s rows but got %s' % (attr, len(arr), len(self.formatted_values.index)))
                self.rowattrs.ix[:, attr] = arr
        return self

    def set_col_widths(self, pcts=None, amts=None, maxs=None, mins=None):
        """
        :param pcts: the percent of available width to use or ratio is also ok
        :param amts: (Array or scalar) the fixed height of the rows
        :param maxs: (Array or scalar) the maximum height of the rows (only use when pcts is used)
        :param mins: (Array or scalar) the minimum height of the rows (only used when pcts is used)
        :return:
        """
        for arr, attr in zip([pcts, amts, maxs, mins], ['weight', 'value', 'max', 'min']):
            if arr is not None:
                if not np.isscalar(arr):
                    if len(arr) != len(self.formatted_values.columns):
                        raise ValueError(
                            '%s: expected %s rows but got %s' % (attr, len(arr), len(self.formatted_values.columns)))
                self.colattrs.ix[:, attr] = arr
        return self

    def _resolve_dims(self, available, attrs):
        def _clean(v):
            return None if np.isnan(v) else v

        if attrs['value'].notnull().any():  # Static values
            # Assume that if one is set than all are set
            return [_clean(a) for a in attrs['value']]
        elif attrs['weight'].notnull().any():
            # Dynamic values
            f = attrs
            f['active'] = (attrs['weight'] * available) / attrs['weight'].sum()
            f['active'] = f[['active', 'min']].max(axis=1)
            f['active'] = f[['active', 'max']].min(axis=1)
            return list(f.active.fillna(0))
        elif attrs['min'].notnull().any():
            return [_clean(a) for a in attrs['min']]
        else:
            return None

    def resolve_col_widths(self, availWidth):
        return self._resolve_dims(availWidth, self.colattrs)

    def resolve_row_heights(self, availHeight):
        return self._resolve_dims(availHeight, self.rowattrs)

    def build(self, expand='wh', shrink='wh', vAlign='MIDDLE', hAlign='CENTER'):
        return TableLayout(self, expand, shrink, hAlign, vAlign)

    def _find_column_label_positions(self, match_value_or_fct, levels=None):
        """Check the original DataFrame's column labels to find the locations of columns. And return the adjusted
        column indexing within region (offset if including index)"""
        allmatches = find_locations(self.df.columns, match_value_or_fct, levels)
        if allmatches and self.inc_index:  # tramslate back
            allmatches = [m + self.nidxs for m in allmatches]
        return allmatches

    def _find_row_label_positions(self, match_value_or_fct, levels=None):
        """Check the original DataFrame's row labels to find the locations of rows. And return the adjusted
        row indexing within region (offset if including index)"""
        allmatches = find_locations(self.df.index, match_value_or_fct, levels)
        if allmatches and self.inc_index:  # tramslate back
            allmatches = [m + self.nhdrs for m in allmatches]
        return allmatches


class TableLayout(Flowable):
    def __init__(self, tb, expand='wh', shrink='wh', hAlign='CENTER', vAlign='MIDDLE'):
        self.tb = tb
        self.expand = expand or ''
        self.shrink = shrink or ''
        self.vAlign = vAlign
        self.hAlign = hAlign
        self._style_and_data = None
        self.component = None

    @property
    def style_and_data(self):
        if self._style_and_data is None:
            data = self.tb.formatted_values.values.tolist()
            style = TableStyle(self.tb.style_cmds)
            self._style_and_data = style, data
        return self._style_and_data

    def wrap(self, aw, ah):
        style, data = self.style_and_data
        # Apply any column / row sizes requested
        widths = self.tb.resolve_col_widths(aw)
        heights = self.tb.resolve_row_heights(ah)
        tbl = Table(data, colWidths=widths, rowHeights=heights, style=style, vAlign=self.vAlign, hAlign=self.hAlign,
                    repeatCols=False, repeatRows=True)
        w, h = tbl.wrap(aw, ah)
        pw, ph = w / float(aw), h / float(ah)
        shrink, expand = self.shrink, self.expand
        scale = 0
        if expand and pw < 1. and ph < 1.:
            scale = max('w' in expand and pw or 0, 'h' in expand and ph or 0)
        elif shrink and (pw > 1. or ph > 1.):
            scale = max('w' in shrink and pw or 0, 'h' in expand and ph or 0)

        if scale:
            self.component = comp = KeepInFrame(aw, ah, content=[tbl], hAlign=self.hAlign, vAlign=self.vAlign)
            w, h = comp.wrapOn(self.canv, aw, ah)
            comp._scale = scale
        else:
            self.component = tbl
        return w, h

    def drawOn(self, canvas, x, y, _sW=0):
        return self.component.drawOn(canvas, x, y, _sW=_sW)

    def split(self, aw, ah):
        return self.component.split(aw, ah)
