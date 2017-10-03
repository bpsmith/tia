from reportlab.platypus import Table, TableStyle, Flowable
from reportlab.lib.colors import grey, white, HexColor, black, gray
from matplotlib.colors import rgb2hex, LinearSegmentedColormap
from matplotlib.pyplot import get_cmap
import numpy as np
import pandas as pd

from tia.rlab.components import KeepInFrame
import tia.util.fmt as fmt


__all__ = ['ConditionalRedBlack', 'DynamicTable', 'TableFormatter', 'RegionFormatter', 'IntFormatter', 'FloatFormatter',
           'PercentFormatter', 'ThousandsFormatter', 'MillionsFormatter', 'BillionsFormatter', 'DollarCentsFormatter',
           'DollarFormatter', 'ThousandDollarsFormatter', 'MillionDollarsFormatter', 'BillionDollarsFormatter',
           'YmdFormatter', 'Y_m_dFormatter', 'DynamicNumberFormatter', 'BorderTypeGrid', 'BorderTypeHorizontal',
           'BorderTypeOutline', 'BorderTypeOutline', 'BorderTypeVertical', 'Style', 'BorderTypeOutlineCols']

DefaultHeaderStyle = {
    "GRID": (.5, grey), "BOX": (.25, black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 3, "TOPPADDING": 3, "FONTSIZE": 6, "BACKGROUND": HexColor("#404040"),
    "FONTNAME": "Helvetica", "ALIGN": "CENTER", "TEXTCOLOR": white
}

DefaultCellStyle = {
    "GRID": (.5, grey), "BOX": (.25, black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 2, "TOPPADDING": 2, "ALIGN": "CENTER", "TEXTCOLOR": black,
    "ROWBACKGROUNDS": [[HexColor("#e3ebf4"), white]], "FONTSIZE": 6, "FONTNAME": "Helvetica"  # "FONTNAME": "Courier"
}

DefaultIndexStyle = {
    "GRID": (.5, grey), "BOX": (.25, black), "VALIGN": "MIDDLE", "LEADING": 6, "LEFTPADDING": 3,
    "RIGHTPADDING": 3, "BOTTOMPADDING": 2, "TOPPADDING": 2, "ALIGN": "RIGHT", "TEXTCOLOR": black,
    "ROWBACKGROUNDS": [[HexColor("#e3ebf4"), white]], "FONTSIZE": 6, "FONTNAME": "Helvetica"
}

DefaultWeight = .7

AlignRight = {'ALIGN': 'RIGHT'}

ConditionalRedBlack = lambda x: x < 0 and dict(TEXTCOLOR=HexColor("#800000"))


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
# Don't attempt to pad
DynamicNumberFormatter = fmt.DynamicNumberFormat(method='col', nan='-', pcts=1, trunc_dot_zeros=1)

DollarCentsFormatter = pad_positive_wrapper(fmt.new_float_formatter(prefix='$', nan='-'))
DollarFormatter = pad_positive_wrapper(fmt.new_int_formatter(prefix='$', nan='-'))
ThousandDollarsFormatter = pad_positive_wrapper(fmt.new_thousands_formatter(prefix='$', nan='-'))
MillionDollarsFormatter = pad_positive_wrapper(fmt.new_millions_formatter(prefix='$', nan='-'))
BillionDollarsFormatter = pad_positive_wrapper(fmt.new_billions_formatter(prefix='$', nan='-'))
YmdFormatter = fmt.new_datetime_formatter('%Y%m%d', True)
Y_m_dFormatter = fmt.new_datetime_formatter('%Y-%m-%d', True)
mdYFormatter = fmt.new_datetime_formatter('%m/%d/%Y', True)


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
        match_value = match_value_or_fct
        if not isinstance(match_value, str) and hasattr(match_value, '__iter__'):
            fct = lambda v: v in match_value
        else:
            fct = lambda v: v == match_value_or_fct

    for lvl, loc, val in level_iter(index, levels):
        if fct(val):
            matches.append(loc)
            if max_matches and len(matches) >= matches:
                break
    return matches


def level_iter(index, levels=None):
    if levels is None:
        levels = list(range(index.nlevels))
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


class BorderType(object):
    def __init__(self, weight=DefaultWeight, color=black, cap=None, dashes=None, join=None, count=None, space=None):
        args = locals()
        args.pop('self')
        self.kwargs = args

    def apply(self, rng, **overrides):
        args = self.kwargs.copy()
        args.update(overrides)
        self._do_apply(rng, args)

    def _do_apply(self, rng, args):
        raise NotImplementedError()


class BorderTypeGrid(BorderType):
    def _do_apply(self, rng, args):
        rng.set_grid(**args)


class BorderTypeOutline(BorderType):
    def _do_apply(self, rng, args):
        rng.set_box(**args)


class BorderTypeHorizontal(BorderType):
    def _do_apply(self, rng, args):
        fct = lambda r: (r.set_lineabove(**args), r.set_linebelow(**args))
        [fct(row) for row in rng.iter_rows()]


class BorderTypeOutlineCols(BorderType):
    def _do_apply(self, rng, args):
        [col.set_box(**args) for col in rng.iter_cols()]


class BorderTypeVertical(BorderType):
    def _do_apply(self, rng, args):
        fct = lambda r: (r.set_linebefore(**args), r.set_lineafter(**args))
        [fct(col) for col in rng.iter_cols()]


class Style(object):
    Blue = {'Light': HexColor('#dce6f1'),
            'Medium': HexColor('#95b3d7'),
            'Dark': HexColor('#4f81bd')}

    Black = {'Light': HexColor('#d9d9d9'),
             'Medium': HexColor('#6c6c6c'),
             'Dark': black}

    Red = {'Light': HexColor('#f2dcdb'),
           'Medium': HexColor('#da9694'),
           'Dark': HexColor('#c0504d')}

    Lime = {'Light': HexColor('#ebf1de'),
            'Medium': HexColor('#c4d79b'),
            'Dark': HexColor('#9bbb59')}

    Purple = {'Light': HexColor('#e4dfec'),
              'Medium': HexColor('#b1a0c7'),
              'Dark': HexColor('#8064a2')}

    Orange = {'Light': HexColor('#fde9d9'),
              'Medium': HexColor('#fabf8f'),
              'Dark': HexColor('#f79646')}

    Cyan = {'Light': HexColor('#eff4f6'),
            'Medium': HexColor('#a8c2cb'),
            'Dark': HexColor('#3b595f')}

    DarkBlue = {'Light': HexColor('#e5eaee '),
                'Medium': HexColor('#9aabbc'),
                'Dark': HexColor('#042f59')}

    @staticmethod
    def apply_basic(formatter, font='Helvetica', font_bold='Helvetica-Bold', font_size=8, rpad=None, lpad=None,
                    bpad=None, tpad=None, colspans=1, rowspans=0):
        lpad = 4. / 8. * font_size if lpad is None else 3
        rpad = 4. / 8. * font_size if rpad is None else 3
        bpad = 4. / 8. * font_size if bpad is None else 4
        tpad = 4. / 8. * font_size if tpad is None else 4
        formatter.all.set_font(font, size=font_size, leading=font_size)
        formatter.all.set_pad(lpad, bpad, rpad, tpad)
        formatter.all.set_valign_middle()
        # do the default things
        formatter.header.set_font(font_bold)
        formatter.header.set_align_center()
        formatter.index_header.set_font(font_bold)
        formatter.index_header.set_align_left()
        formatter.index.set_font(font_bold)
        formatter.index.set_align_left()
        formatter.cells.set_font(font)
        formatter.cells.set_align_right()
        # do col spans and row spans
        if rowspans and formatter.index.ncols > 1:
            formatter.index.iloc[:, :formatter.index.ncols - 1].detect_rowspans()
        if colspans and formatter.header.nrows > 1:
            formatter.header.iloc[:formatter.header.nrows - 1, :].detect_colspans()

    @staticmethod
    def apply_color(formatter, cmap=None, font_bw=1, stripe_rows=1, stripe_cols=0,
                    hdr_border_clazz=BorderTypeGrid, cell_border_clazz=BorderTypeOutline, border_weight=.7):
        """
        font_bw: bool, If True use black and white fonts. If False, then use the cmap
        """
        cmap = cmap or Style.Blue
        light = cmap.get('Light', white)
        medium = cmap.get('Medium', gray)
        dark = cmap.get('Dark', black)
        # the ranges
        header = formatter.all.iloc[:formatter.header.nrows]
        cells = formatter.all.iloc[formatter.header.nrows:]
        # color the header
        hdr_border_clazz and header.set_border_type(hdr_border_clazz, color=medium, weight=border_weight)
        header.set_textcolor(font_bw and white or light)
        header.set_background(dark)
        # color the cells
        cell_border_clazz and cells.set_border_type(cell_border_clazz, color=medium, weight=border_weight)
        stripe_rows and cells.set_row_backgrounds([light, white])
        stripe_cols and cells.set_col_backgrounds([white, light])
        not font_bw and cells.set_textcolor(dark)


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
    def last_row(self):
        return self.empty_frame() if self.nrows == 0 else self.iloc[-1:, :]

    @property
    def last_col(self):
        return self.empty_frame() if self.ncols == 0 else self.iloc[:, -1:]

    def is_empty(self):
        return self.nrows == 0 and self.ncols == 0

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

    def iter_rows(self, start=None, end=None):
        """Iterate each of the Region rows in this region"""
        start = start or 0
        end = end or self.nrows
        for i in range(start, end):
            yield self.iloc[i, :]

    def iter_cols(self, start=None, end=None):
        """Iterate each of the Region cols in this region"""
        start = start or 0
        end = end or self.ncols
        for i in range(start, end):
            yield self.iloc[:, i]

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
                c = [cmd, (c0, r0), (c1, r1)] + list(args)
                self.style_cmds.append(c)
        return self

    def apply_styles(self, cmdmap):
        """
        Apply the set of commands defined in cmdmap. for example, apply_styles({'FONTSIZE': 12, 'BACKGROUND': white})
        :param cmdmap: dict of commands mapped to the command arguments
        :return: self
        """
        is_list_like = lambda arg: isinstance(arg, (list, tuple))
        is_first_param_list = lambda c: c in ('COLBACKGROUNDS', 'ROWBACKGROUNDS')
        for cmd, args in cmdmap.items():
            if not is_list_like(args):
                args = [args]
            elif is_first_param_list(cmd) and is_list_like(args) and not is_list_like(args[0]):
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
        for k, v in kwargs.items():
            self.parent.rowattrs.iloc[self.row_ilocs, k] = v
        return self

    def apply_colattrs(self, **kwargs):
        for k, v in kwargs.items():
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

    def _do_number_format(self, rb, align, fmt_fct, fmt_args, defaults):
        args = {}
        defaults and args.update(defaults)
        fmt_args and args.update(fmt_args)
        f = pad_positive_wrapper(fmt_fct(**args))
        return self.apply_number_format(f, rb=rb, align=align)

    def percent_format(self, rb=1, align=1, **fmt_args):
        defaults = {'precision': 2, 'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_percent_formatter, fmt_args, defaults)

    def int_format(self, rb=1, align=1, **fmt_args):
        defaults = {'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_int_formatter, fmt_args, defaults)

    def float_format(self, rb=1, align=1, **fmt_args):
        defaults = {'precision': 2, 'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_float_formatter, fmt_args, defaults)

    def thousands_format(self, rb=1, align=1, **fmt_args):
        defaults = {'precision': 1, 'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_thousands_formatter, fmt_args, defaults)

    def millions_format(self, rb=1, align=1, **fmt_args):
        defaults = {'precision': 1, 'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_millions_formatter, fmt_args, defaults)

    def billions_format(self, rb=1, align=1, **fmt_args):
        defaults = {'precision': 1, 'nan': '-'}
        return self._do_number_format(rb, align, fmt.new_billions_formatter, fmt_args, defaults)

    def guess_number_format(self, rb=1, align=1, **fmt_args):
        """Determine the most appropriate formatter by inspected all the region values"""
        fct = fmt.guess_formatter(self.actual_values, **fmt_args)
        return self.apply_number_format(fct, rb=rb, align=align)

    def guess_format(self, rb=1, align=1, **fmt_args):
        from tia.util.fmt import NumberFormat

        fct = fmt.guess_formatter(self.actual_values, **fmt_args)
        if isinstance(fmt, NumberFormat):
            return self.apply_number_format(fct, rb=rb, align=align)
        else:
            return self.apply_format(fct)

    def dynamic_number_format(self, rb=1, align=1, **fmt_args):
        """Formatter changes based on the cell value"""
        fct = fmt.DynamicNumberFormatter(**fmt_args)
        return self.apply_number_format(fct, rb=rb, align=align)

    # def heat_map(self, cmap=None, min=None, max=None, font_cmap=None):
    def heat_map(self, cmap='RdYlGn', vmin=None, vmax=None, font_cmap=None):
        if cmap is None:
            carr = ['#d7191c', '#fdae61', '#ffffff', '#a6d96a', '#1a9641']
            cmap = LinearSegmentedColormap.from_list('default-heatmap', carr)

        if isinstance(cmap, str):
            cmap = get_cmap(cmap)
        if isinstance(font_cmap, str):
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
                styles = {'BACKGROUND': HexColor(hex)}
                if font_cmap is not None:
                    styles['TEXTCOLOR'] = HexColor(rgb2hex(font_cmap(v)))
                self.iloc[ridx, cidx].apply_styles(styles)
        return self

    heatmap = heat_map

    def set_font(self, name=None, size=None, leading=None, color=None):
        name and self.set_fontname(name)
        size and self.set_fontsize(size)
        leading and self.set_leading(leading)
        color and self.set_textcolor(color)
        return self

    def set_fontname(self, name):
        return self.apply_style('FONTNAME', name)

    def set_fontsize(self, size):
        return self.apply_style('FONTSIZE', size)

    def set_textcolor(self, color):
        return self.apply_style('TEXTCOLOR', color)

    def set_leading(self, n):
        return self.apply_style('LEADING', n)

    def set_valign(self, pos):
        return self.apply_style('VALIGN', pos)

    def set_valign_middle(self):
        return self.set_valign('MIDDLE')

    def set_valign_center(self):
        return self.set_valign_middle()

    def set_valign_top(self):
        return self.set_valign('TOP')

    def set_valign_bottom(self):
        return self.set_valign('BOTTOM')

    def set_align(self, pos):
        return self.apply_style('ALIGN', pos)

    def set_align_center(self):
        return self.set_align('CENTER')

    def set_align_middle(self):
        return self.set_align_center()

    def set_align_left(self):
        return self.set_align('LEFT')

    def set_align_right(self):
        return self.set_align('RIGHT')

    def set_pad(self, left, bottom, right, top):
        return self.set_lpad(left).set_bpad(bottom).set_rpad(right).set_tpad(top)

    def set_lpad(self, n):
        return self.apply_style('LEFTPADDING', n)

    def set_bpad(self, n):
        return self.apply_style('BOTTOMPADDING', n)

    def set_rpad(self, n):
        return self.apply_style('RIGHTPADDING', n)

    def set_tpad(self, n):
        return self.apply_style('TOPPADDING', n)

    def set_box(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None, space=None):
        return self.apply_style('BOX', weight, color, cap, dashes, join, count, space)

    def set_grid(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None, space=None):
        return self.apply_style('GRID', weight, color, cap, dashes, join, count, space)

    def set_lineabove(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None, space=None):
        return self.apply_style('LINEABOVE', weight, color, cap, dashes, join, count, space)

    def set_linebelow(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None, space=None):
        return self.apply_style('LINEBELOW', weight, color, cap, dashes, join, count, space)

    def set_linebefore(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None,
                       space=None):
        return self.apply_style('LINEBEFORE', weight, color, cap, dashes, join, count, space)

    def set_lineafter(self, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None, space=None):
        return self.apply_style('LINEAFTER', weight, color, cap, dashes, join, count, space)

    def set_border_type(self, clazz, weight=DefaultWeight, color=None, cap=None, dashes=None, join=None, count=None,
                        space=None):
        """example: set_border_type(BorderTypePartialRows) would set a border above and below each row in the range"""
        args = locals()
        args.pop('clazz')
        args.pop('self')
        clazz(**args).apply(self)

    def set_background(self, color):
        return self.apply_style('BACKGROUND', color)

    def set_col_backgrounds(self, colors):
        """Set alternative column colors"""
        return self.apply_style('COLBACKGROUNDS', colors)

    def set_row_backgrounds(self, colors):
        """Set alternative row colors"""
        return self.apply_style('ROWBACKGROUNDS', colors)


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
        # bug when ridx is -1 and only a single row - cannot get DataFrame
        if np.isscalar(ridx) and ridx == -1 and len(region.formatted_values.index) == 1:
            ridx = [0]
        else:
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

    def apply_default_header_style(self, inc_index=0, **overrides):
        styles = self.get_default_header_style(**overrides)
        self.header.apply_styles(styles)
        if inc_index:
            self.index_header.apply_styles(styles)
        return self

    def apply_default_index_style(self, **overrides):
        styles = dict(DefaultIndexStyle, **overrides)
        self.index.apply_styles(styles)
        return self

    def apply_default_style(self, inc_cells=1, inc_header=1, inc_index=1, inc_index_header=0, cells_override=None,
                            header_override=None,
                            index_override=None):
        inc_cells and self.apply_default_cell_style(**(cells_override or {}))
        inc_header and self.apply_default_header_style(inc_index=inc_index_header, **(header_override or {}))
        inc_index and self.apply_default_index_style(**(index_override or {}))
        return self

    def apply_basic_style(self, font='Helvetica', font_bold='Helvetica-Bold', font_size=8, rpad=None, lpad=None,
                          bpad=None, tpad=None, colspans=1, rowspans=0, cmap=None, font_bw=1, stripe_rows=1,
                          stripe_cols=0,
                          hdr_border_clazz=BorderTypeGrid, cell_border_clazz=BorderTypeOutline, border_weight=.7):
        Style.apply_basic(self, font=font, font_bold=font_bold, font_size=font_size, rpad=rpad, lpad=lpad, bpad=bpad,
                          tpad=tpad, colspans=colspans, rowspans=rowspans)
        Style.apply_color(self, cmap, font_bw=font_bw, stripe_cols=stripe_cols, stripe_rows=stripe_rows,
                          hdr_border_clazz=hdr_border_clazz, cell_border_clazz=cell_border_clazz,
                          border_weight=border_weight)
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
        :param amts: (Array or scalar) the fixed width of the cols
        :param maxs: (Array or scalar) the maximum width of the cols (only use when pcts is used)
        :param mins: (Array or scalar) the minimum width of the cols (only used when pcts is used)
        :return:
        """
        for arr, attr in zip([pcts, amts, maxs, mins], ['weight', 'value', 'max', 'min']):
            if arr is not None:
                if not np.isscalar(arr):
                    if len(arr) != len(self.formatted_values.columns):
                        raise ValueError(
                            '%s: expected %s cols but got %s' % (attr, len(arr), len(self.formatted_values.columns)))
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
        if self.component:
            return self.component.split(aw, ah)
        else:
            return []
