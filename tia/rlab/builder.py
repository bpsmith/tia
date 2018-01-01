from reportlab.platypus import BaseDocTemplate, Paragraph, Frame, PageBreak, FrameBreak, NextPageTemplate, \
    PageTemplate
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, TA_CENTER
from reportlab.lib import units
from reportlab.lib.colors import HexColor
from reportlab.platypus.flowables import Flowable, HRFlowable
from tia.rlab.table import TableFormatter

import numpy as np

__all__ = ['CoverPage', 'GridFrame', 'GridTemplate', 'PdfBuilder']


class CoverPage(object):
    def __init__(self, title='Title', subtitle='Subtitle', subtitle2=None, font='Helvetica', logo_path=None):
        self.title = title
        self.subtitle = subtitle
        self.subtitle2 = subtitle2
        self.font = font
        self.logo_path = logo_path

    def onPage(self, canvas, doc):
        c = canvas
        w, h = c._pagesize
        # The cover page just has some drawing on the canvas.
        c.saveState()
        isletter = (w, h) == letter
        c.setFont(self.font, isletter and 16 or 20)
        imgw, imgh = 2.83 * units.inch, .7 * units.inch

        c.drawString(25, h / 2 - 6, self.title)
        if self.logo_path:
            c.drawImage(self.logo_path, w - imgw - 25,
                        h / 2 - .5 * imgh, width=imgw, height=imgh,
                        preserveAspectRatio=True)
        c.setFillColorRGB(0, 0, 0)
        c.rect(0, h / 2 + .5 * imgh + 5, w, 1, fill=1)
        c.rect(0, h / 2 - .5 * imgh - 5, w, 1, fill=1)
        c.setFontSize(isletter and 12 or 16)
        c.drawString(25, h / 2 - .5 * imgh - 50, self.subtitle)
        if self.subtitle2:
            c.drawString(25, h / 2 - .5 * imgh - 70, self.subtitle2)
        c.restoreState()


def _to_points(ix, n):
    if isinstance(ix, slice):
        p0, p1, _ = ix.indices(n)
        return p0, p1
    elif np.isscalar(ix):
        ix = (ix < 0 and ix + n) or ix
        if ix < 0 or ix >= n:
            raise IndexError('index %s out of range (0, %s)' % (ix, n))
        return ix, ix + 1
    else:
        raise Exception('invalid indexer type %s, expected slice or scalar' % type(ix))


class GridFrame(object):
    def __init__(self, grid, x0, y0, x1, y1):
        self.grid = grid
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    nrows = property(lambda self: self.grid.nrows)
    ncols = property(lambda self: self.grid.ncols)

    def as_frame(self, builder, alias, **frame_args):
        rheight = builder.height / self.nrows
        cwidth = builder.width / self.ncols
        rs, re, cs, ce = self.y0, self.y1, self.x0, self.x1
        rs, re = abs(self.nrows - rs), abs(self.nrows - re)
        x = cs * cwidth
        y = re * rheight
        h = (rs - re) * rheight
        w = (ce - cs) * cwidth
        return Frame(x, y, w, h, id=alias, **frame_args)


class GridTemplate(object):
    """User defined grid system which will map to pdf page template. uses numpy style slicing to define GridFrames"""

    def __init__(self, template_id, nrows, ncols):
        self.template_id = template_id
        self.nrows = nrows
        self.ncols = ncols
        self.gframes = {}

    def __getitem__(self, key):
        nrows, ncols = self.nrows, self.ncols
        if isinstance(key, tuple):
            ridx = key[0]
            cidx = key[1] if len(key) > 1 else slice(None)
        else:
            ridx = key
            cidx = slice(None)

        row0, row1 = _to_points(ridx, nrows)
        col0, col1 = _to_points(cidx, ncols)
        return GridFrame(self, col0, row0, col1, row1)

    def define_frame(self, alias, grid_frame, **frame_args):
        self.gframes[alias] = grid_frame, frame_args

    def define_frames(self, alias_map):
        for alias, value in alias_map.items():
            if isinstance(value, GridFrame):
                gf = value
                frame_args = {}
            else:
                gf = value[0]
                frame_args = len(value) > 0 and value[1] or {}
            self.define_frame(alias, gf, **frame_args)

    def as_page_template(self, builder):
        rheight = builder.height / self.nrows
        cwidth = builder.width / self.ncols
        frames = []
        for alias, (gframe, fargs) in self.gframes.items():
            rs, re, cs, ce = gframe.y0, gframe.y1, gframe.x0, gframe.x1
            # Flip since y-axis starting at bottom
            rs, re = abs(self.nrows - rs), abs(self.nrows - re)
            x = cs * cwidth
            y = re * rheight
            h = (rs - re) * rheight
            w = (ce - cs) * cwidth
            frames.append(Frame(x, y, w, h, id=alias, **fargs))
        pt = PageTemplate(frames=frames)
        pt.id = self.template_id
        return pt

    def register(self, builder):
        pt = self.as_page_template(builder)
        builder.add_page_template(pt)


def raise_template_not_found(template_id):
    msg = "unable to find page template with id: %s" % template_id
    raise ValueError(msg)


class PdfBuilder(object):
    @classmethod
    def build_doc(cls, path, pagesize=None, showBoundary=1, allowSplitting=1, **dargs):
        if pagesize is None:
            pagesize = landscape(letter)
        return BaseDocTemplate(path, pagesize=pagesize, showBoundary=showBoundary, allowSplitting=allowSplitting,
                               **dargs)

    def __init__(self, doc_or_path, coverpage=None, pagesize=None, stylesheet=None, showBoundary=0):
        self.path = None
        if isinstance(doc_or_path, str):
            self.path = doc_or_path
            doc = self.build_doc(doc_or_path, pagesize=pagesize, showBoundary=showBoundary)

        self.doc = doc
        self.pagesize = doc.pagesize
        self.width, self.height = self.pagesize
        self.inc_cover = inc_coverpage = coverpage is not None
        self.template_defs = {}
        self.story = []
        self.active_template_id = None
        self.stylesheet = stylesheet or getSampleStyleSheet()
        if inc_coverpage:
            # Allow user to override the cover page template
            if not self.get_page_template('cover', err=0):
                f = Frame(0, 0, self.width, self.height)
                pt = PageTemplate(id='cover', frames=[f], onPage=coverpage.onPage)
                self.add_page_template(pt)

    def new_title_bar(self, title, color=None):
        """Return an array of Pdf Objects which constitute a Header"""
        # Build a title bar for top of page
        w, t, c = '100%', 2, color or HexColor('#404040')
        title = '<b>{0}</b>'.format(title)
        if 'TitleBar' not in self.stylesheet:
            tb = ParagraphStyle('TitleBar', parent=self.stylesheet['Normal'], fontName='Helvetica-Bold', fontSize=10,
                                leading=10, alignment=TA_CENTER)
            self.stylesheet.add(tb)
        return [HRFlowable(width=w, thickness=t, color=c, spaceAfter=2, vAlign='MIDDLE', lineCap='square'),
                self.new_paragraph(title, 'TitleBar'),
                HRFlowable(width=w, thickness=t, color=c, spaceBefore=2, vAlign='MIDDLE', lineCap='square')]

    def new_paragraph(self, txt, style='Normal'):
        s = self.stylesheet[style]
        return Paragraph(txt, style=self.stylesheet[style])

    para = new_paragraph
    p = new_paragraph

    def add_page_template(self, pt):
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
        self.doc.addPageTemplates(pt)
        return self

    def get_page_template(self, template_id, default=None, err=1):
        for pt in self.doc.pageTemplates:
            if pt.id == template_id:
                return pt
        return raise_template_not_found(template_id) if err else default

    def has_page_template(self, template_id):
        for pt in self.doc.pageTemplates:
            if pt.id == template_id:
                return True
        return False

    def make_template_first(self, template_id):
        ids = [pt.id for pt in self.doc.pageTemplates]
        if template_id not in ids:
            raise_template_not_found(template_id)
        elif (not self.inc_cover and ids[0] != template_id) or (self.inc_cover and ids[1] != template_id):
            tmp = self.doc.pageTemplates.pop(ids.index(template_id))
            self.doc.pageTemplates.insert(self.inc_cover and 1 or 0, tmp)

    def build_page(self, template_id, flowable_map):
        """Build a pdf page by looking up the specified template and then mapping the flowable_map items to the
        appropriate named Frame
        """
        pt = self.get_page_template(template_id)
        # If this is the first page then ensure the page template is ordered first and no breaks or changes
        # are requested otherwise blank page shows up
        if self.active_template_id is None:
            self.make_template_first(template_id)
            self.story.append(NextPageTemplate(template_id))
            self.inc_cover and self.story.append(PageBreak())
            self.active_template_id = template_id
        elif self.active_template_id == template_id:
            # TODO - understand why this is necessary to not get a blank page between pages
            self.story.append(PageBreak())
        else:
            self.story.append(NextPageTemplate(template_id))
            self.story.append(PageBreak())
            self.active_template_id = template_id

        for idx, frame in enumerate(pt.frames):
            if frame.id not in flowable_map:
                # Add a note to the template to show that nothing was defined for this area
                self.story.append(Paragraph('NOT DEFINED: %s' % frame.id, getSampleStyleSheet()['Normal']))
            else:
                flowables = flowable_map[frame.id]
                if not isinstance(flowables, Flowable) and hasattr(flowables, '__iter__'):
                    [self.story.append(f) for f in flowables]
                else:
                    self.story.append(flowables)
            if idx < (len(pt.frames) - 1):
                self.story.append(FrameBreak())
        return self

    def define_simple_grid_template(self, template_id, nrows, ncols):
        """Define a simple grid template. This will define nrows*ncols frames, which will be indexed starting with '0,0'
            and using numpy style indexing. So '0,1' is row 0 , col 1"""
        template = GridTemplate(template_id, nrows, ncols)
        [template.define_frame('%s,%s' % (i, j), template[i, j]) for i in range(nrows) for j in range(ncols)]
        template.register(self)
        return self

    def table_formatter(self, dataframe, inc_header=1, inc_index=1):
        """Return a table formatter for the dataframe. Saves the user the need to import this class"""
        return TableFormatter(dataframe, inc_header=inc_header, inc_index=inc_index)

    def save(self):
        if isinstance(self.story[-1], PageBreak):
            del self.story[-1]

        self.doc.build(self.story)
        # self.doc.multiBuild(self.story)


