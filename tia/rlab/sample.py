AlignRight = {'ALIGN': 'RIGHT'}
import os
import tempfile
import itertools

import pandas as pd
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import *

from tia.rlab import *
import tia.util.fmt as fmt


def sample1():
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_4.pdf')
    cols = ['pct', 'int', 'k', 'M', '$', 'date']
    df = pd.DataFrame(np.random.randn(40, len(cols)), columns=cols)
    df['int'] = 10000. * df['pct']
    df['k'] = 50000. * df['pct']
    df['M'] = 5000000. * df['pct']
    df['$'] = 500000. * df['pct']
    df['date'] = pd.date_range('1/1/2010', periods=len(df.index))
    df['id'] = 'ID-1'
    # Make this a multi-index frame
    df2 = df.copy()
    df2['id'] = 'ID-2'
    df = df.set_index('id', append=True).unstack().reorder_levels([1, 0], axis=1)
    df2 = df2.set_index('id', append=True).unstack().reorder_levels([1, 0], axis=1)
    aggdf = pd.concat([df, df2], axis=1)
    # Start building the pdf
    pdf = PdfBuilder(pdf_path)
    # build the templates to use
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)

    # Build the pdf tables to marry with the template
    def make_builder(hdr=1, idx=1, cstyles=None):
        tf = TableFormatter(aggdf, inc_header=hdr, inc_index=idx)
        tf.apply_default_style(index_override={'BACKGROUND': colors.beige})
        tf.header.detect_colspans()
        tf.header.apply_style('ALIGN', 'CENTER')
        tf.cells.match_any_labels('pct').apply_number_format(PercentFormatter)
        tf.cells.match_any_labels('k').apply_number_format(ThousandsFormatter)
        tf.cells.match_any_labels('int').apply_number_format(IntFormatter)
        tf.cells.match_any_labels('M').apply_number_format(MillionsFormatter)
        tf.cells.match_any_labels('$').apply_number_format(DollarCentsFormatter)

        def red_weekend(x):
            if x.dayofweek in (5, 6):
                return dict(BACKGROUND=colors.HexColor("#800000"), TEXTCOLOR=colors.white)

        tf.cells.match_any_labels('date').apply(format=fmt.Y_m_dFormatter, cstyles=red_weekend)
        return tf

    # Build PDF
    for hon, ion in list(itertools.product([True, False], repeat=2)):
        offon = lambda v: v and 'On' or 'Off'
        for cstyle in [None, ConditionalRedBlack]:
            hdr = 'Index=%s Header=%s Color=%s' % (offon(ion), offon(hon), offon(cstyle is not None))
            data = {
                'HEADER': Paragraph(hdr, getSampleStyleSheet()['Normal']),
                'TBL': make_builder(hon, ion, cstyle).build(),
            }
            pdf.build_page('T1', data)
    pdf.save()
    print(pdf_path)


def sample_long_table():
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_long_table.pdf')
    cols = ['pct', 'int', 'k', 'M', '$', 'date']
    df = pd.DataFrame(np.random.randn(200, len(cols)), columns=cols)
    df['int'] = 10000. * df['pct']
    df['k'] = 50000. * df['pct']
    df['M'] = 5000000. * df['pct']
    df['$'] = 500000. * df['pct']
    df['date'] = pd.date_range('1/1/2010', periods=len(df.index))
    df['id'] = 'ID-1'
    # Start building the pdf
    pdf = PdfBuilder(pdf_path)
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)

    tf = TableFormatter(df)
    tf.apply_default_style()
    pdf.build_page('T1', {'HEADER': pdf.para('HEADER'), 'TBL': tf.build()})
    pdf.save()
    print(pdf_path)


def sample_wide_table():
    # Should shrink to fit the page
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_wide_table.pdf')
    cols = ['C%s' % i for i in range(36)]
    df = pd.DataFrame(np.random.randn(20, len(cols)), columns=cols)
    pdf = PdfBuilder(pdf_path)
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)
    tf = TableFormatter(df)
    tf.apply_default_style()
    pdf.build_page('T1', {'HEADER': pdf.para('HEADER'), 'TBL': tf.build()})
    pdf.save()
    print(pdf_path)


def sample_dyn_col_row_table():
    # Should shrink to fit the page
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_dyn_col_row_table.pdf')
    cols = ['C%s' % i for i in range(4)]
    df = pd.DataFrame(np.random.randn(4, len(cols)), columns=cols)
    pdf = PdfBuilder(pdf_path)
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)
    tf = TableFormatter(df)
    tf.apply_default_style()
    tf.set_col_widths(pcts=[.2, .1, .2, .3, .4])
    tf.set_row_heights(pcts=[.2, .1, .2, .3, .4])
    pdf.build_page('T1', {'HEADER': pdf.para('HEADER'), 'TBL': tf.build()})
    pdf.save()
    print(pdf_path)


def sample_multi_page():
    # Expand the colwidths to fill page width, but allow rows to be split across pages
    #
    # TODO - make this right
    #
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_multi_page.pdf')
    cols = ['C%s' % i for i in range(4)]
    df = pd.DataFrame(np.random.randn(400, len(cols)), columns=cols)
    pdf = PdfBuilder(pdf_path)
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)
    tf = TableFormatter(df)
    tf.apply_default_style()
    tf.set_col_widths(pcts=[.2, .1, .2, .3, .4])
    pdf.build_page('T1', {'HEADER': pdf.para('HEADER'), 'TBL': tf.build(shrink=None)})
    pdf.save()
    print(pdf_path)


def sample_heatmap():
    # Expand the colwidths to fill page width, but allow rows to be split across pages
    pdf_path = os.path.join(tempfile.gettempdir(), 'pdf_test_heat_map.pdf')
    df = pd.DataFrame(np.arange(-100, 100, 10).reshape((5, 4)), columns=['A', 'B', 'C', 'D'])
    pdf = PdfBuilder(pdf_path)
    gt = GridTemplate('T1', 100, 100)
    gt.define_frames({
        'HEADER': gt[:10, :],
        'TBL': gt[10:],
    })
    gt.register(pdf)
    tf = TableFormatter(df)
    tf.apply_default_style()
    tf.cells.heat_map()
    pdf.build_page('T1', {'HEADER': pdf.para('ALL CELLS'), 'TBL': tf.build(shrink=None)})
    tf = TableFormatter(df)
    tf.apply_default_style()
    tf.cells.iloc[:, 0].heat_map()
    pdf.build_page('T1', {'HEADER': pdf.para('First Column'), 'TBL': tf.build(shrink=None)})
    tf = TableFormatter(df)
    tf.apply_default_style()
    # change font for fun
    tf.cells.iloc[0, :].heat_map(font_cmap='Greys')
    pdf.build_page('T1', {'HEADER': pdf.para('First Row'), 'TBL': tf.build(shrink=None)})
    pdf.save()
    print(pdf_path)

def runall():
    sample1()
    sample_long_table()
    sample_wide_table()
    sample_dyn_col_row_table()
    #sample_multi_page()
    sample_heatmap()

if __name__ == '__main__':
    runall()
