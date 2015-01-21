# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Example of using the rlab helper classes
# the rlab package provides a simple way to create complex pdfs. It provides the following pieces. 
# 
# - components: provide wrappers to pdf components so they can dynamically scale themselves based on available space
# - table: numpy indexing to provide formatting
# - builder: manage the building of a pdf and defining grid based templates
# 
# Below I will show the following simplified example. For each security (MSFT, CSCO, INTC), we want to generate 2 pdf pages, one page has images and the other has tabular data. 

# <codecell>

import tia.rlab as rlab

# <codecell>

# get some real data to plot and build tables for
from pandas.io.data import get_data_yahoo
pxs = get_data_yahoo(['MSFT', 'INTC', 'CSCO'])

# <codecell>

# - Generate a pdf path
# - define a cover page
# - Create a Pdf Builder
# - Define the templates and register with the builder
#
#        TEMPLATE_1
#  |-------------------------|
#  |        HEADER           | 
#  |-------------------------|
#  |            |            |
#  |            |            |
#  |            |            |
#  |   IMG_1    |    IMG_2   |
#  |            |            |
#  |            |            |
#  |            |            |
#  |-------------------------|
#
#        TEMPLATE_2
#  |-------------------------|
#  |        HEADER           | 
#  |-------------------------|
#  |            |            |
#  |            |    TBL_2   |
#  |            |            |
#  |   TBL_1    |------------|
#  |            |            |
#  |            |    TBL_3   |
#  |            |            |
#  |-------------------------|

pdfpath = r'c:\temp\sample.pdf'

coverpage = rlab.CoverPage('SecurityOverview', 'Reported on Jan-20-2015')
pdf = rlab.PdfBuilder(pdfpath, coverpage=coverpage, showBoundary=0)

# Define TEMPLATE_1
template = rlab.GridTemplate('TEMPLATE_1', nrows=100, ncols=100)
# uses numpy style slicing to define the dimensions
template.define_frames({
    'HEADER': template[:10, :],
    'IMG_1': template[10:, :50],
    'IMG_2': template[10:, 50:]
})
template.register(pdf)

# Define TEMPLATE_2
template = rlab.GridTemplate('TEMPLATE_2', nrows=100, ncols=100)
template.define_frames({
    'HEADER': template[:10, :],
    'TBL_1': template[10:, :50],
    'TBL_2': template[10:55, 50:],
    'TBL_3': template[55:, 50:]
})
template.register(pdf)

# <codecell>


# <codecell>

from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, TA_CENTER

# Add a stylesheet to the pdf
tb = ParagraphStyle('TitleBar', parent=pdf.stylesheet['Normal'], fontName='Helvetica-Bold', fontSize=14, 
                    leading=14, alignment=TA_CENTER)


'TitleBar' not in pdf.stylesheet and pdf.stylesheet.add(tb)


def title_bar(pdf, title):
    # Build a title bar for top of page
    w, t, c = '100%', 2, HexColor('#404040')
    title = '<b>{0}</b>'.format(title)    
    return [HRFlowable(width=w, thickness=t, color=c, spaceAfter=5, vAlign='MIDDLE', lineCap='square'),
            pdf.new_paragraph(title, 'TitleBar'),
            HRFlowable(width=w, thickness=t, color=c, spaceBefore=5, vAlign='MIDDLE', lineCap='square')]

# <codecell>

import tia.util.fmt as fmt
def to_pdf_table(frame):
    table = rlab.TableFormatter(frame, inc_header=1, inc_index=1)
    # use the default style to add a bit of color
    table.apply_default_style()
    # apply a percent formatter to the return column
    table.cells.match_column_labels('return').apply_number_format(fmt.PercentFormatter)
    # apply a millions formatter to volumn column
    table.cells.match_column_labels('Volume').apply_number_format(fmt.MillionsFormatter)
    table.index.apply(format=fmt.Y_m_dFormatter)
    return table.build(vAlign='MIDDLE')
    

# <codecell>

# Define a matplotlib helper to store images by key
from tia.util.mplot import FigureHelper
figures = FigureHelper()

def add_security_to_report(sid, pxframe):
    # build the images
    img1_key = '{0}_open_pxs'.format(sid)
    img2_key = '{0}_close_pxs'.format(sid)
    pxframe['Open'].plot(title='{0} Open Price'.format(sid))    
    figures.savefig(key=img1_key)
    pxframe['Close'].plot(title='{0} Close Price'.format(sid))
    figures.savefig(key=img2_key)
    # build the tables
    pxframe['return'] = pxframe.Close.pct_change()
    tbl1 = to_pdf_table(pxframe[['Open', 'High', 'Low', 'Close']].tail(50))
    tbl2 = to_pdf_table(pxframe.iloc[:10])
    tbl3 = to_pdf_table(pxframe.iloc[-10:])
    
    # Marry the template with the components
    pdf.build_page('TEMPLATE_1', {
        'HEADER': title_bar(pdf, '{0} Images'.format(sid)),
        'IMG_1': rlab.DynamicPdfImage(figures[img1_key]),
        'IMG_2': rlab.DynamicPdfImage(figures[img2_key]),        
    })
    
    pdf.build_page('TEMPLATE_2', {
        'HEADER': title_bar(pdf, '{0} Tables'.format(sid)),
        'TBL_1': tbl1,
        'TBL_2': tbl2,
        'TBL_3': tbl3,
    })

# <codecell>

#
# Add each of the securities
#
for sid in pxs.minor_axis:
    add_security_to_report(sid, pxs.minor_xs(sid))

pdf.save()

# <codecell>


