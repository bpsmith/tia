import tia.rlab as rlab
import tempfile
import os
from reportlab.platypus.flowables import KeepInFrame, Flowable

class TestBox(Flowable):
    def __init__(self, width, height, prefix=''):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.prefix = prefix

    text = property(lambda self: '{0}{1}x{2}'.format(self.prefix, self.width, self.height))

    def __repr__(self):
        return "TestBox(w=%s, h=%s, t=%s)" % (self.width, self.height, self.text)

    def draw(self):
        self.canv.rect(0, 0, self.width, self.height)
        self.canv.line(0, 0, self.width, self.height)
        self.canv.line(0, self.height, self.width, 0)

        #centre the text
        self.canv.setFont('Helvetica', 8)
        self.canv.drawCentredString(0.5*self.width, 0.5*self.height, self.text)


def dynamic_in_frame_sample():
    path = os.path.join(tempfile.gettempdir(), 'dyn_frame.pdf')

    pdf = rlab.PdfBuilder(path, showBoundary=True)
    pdf.define_simple_grid_template('test', 2, 4)

    box = TestBox(100, 100, 'Some Text')
    pdf.build_page('test', {
        '0,0': KeepInFrame(0, 0, [TestBox(100, 100)]),
        '0,1': rlab.DynamicKeepInFrame(TestBox(100, 100, 'Too Small'), zoom=True),
        '0,2': rlab.DynamicKeepInFrame(TestBox(50, 100, 'Too Short'), zoom=True),
        '0,3': rlab.DynamicKeepInFrame(TestBox(100, 50, 'Too Skinny'), zoom=True),
        '1,0': rlab.DynamicKeepInFrame(TestBox(200, 100, 'Too Wide'), zoom=True),
        '1,1': rlab.DynamicKeepInFrame(TestBox(100, 300, 'Too Tall'), zoom=True),
        #'1,1': KeepInFrame(0, 0, [TestBox(200, 100)]),
        #'1,2': KeepInFrame(150, 150, [TestBox(200, 100)]),
        #'1,3': KeepInFrame(150, 150, [XBox(200, 100, 'hi')]),
        #'1,1': KeepInFrame(0, 0, [TestBox(200, 100)]),
        #'1,2': rlab.DynamicKeepInFrame(TestBox(100, 50), zoom=True),
        })
        #'0,3': rlab.DynamicKeepInFrame(TestBox(100, 100, 'DynFrame with Zoom'), zoom=True, maxWidth=125)})
    pdf.save()
    print path


def dynamic_table_sample():

    path = os.path.join(tempfile.gettempdir(), 'dyn_table.pdf')
    pdf = rlab.PdfBuilder(path, showBoundary=True)
    #pdf.define_simple_grid_template('test', 3, 3, sequential=True)
    t = rlab.GridTemplate('test', 100, 100)
    t.define_frames({'desc': t[10:20, 30:50]})
    t.define_frames({'tbl': t[30:70, 30:70]})
    t.register(pdf)

    def mktbl(nrows, ncols, layout):
        import pandas as pd

        df = pd.DataFrame({'C{0}'.format(i): ['A' * i*2] * nrows for i in range(1, ncols+1)})
        tbl = pdf.new_table(df, layout=layout, truncate=True)
        tbl.apply_basic_style()
        return tbl

    def mkpg(nrows, ncols):
        for n, itm in enumerate(dir(rlab.DynamicTableLayout)):
            if not itm.startswith('_'):
                tbl = mktbl(nrows, ncols, getattr(rlab.DynamicTableLayout, itm))
                pdf.build_page('test', {'desc': pdf.para('{0}x{1} - {2}'.format(nrows, ncols, itm)), 'tbl': tbl})

    mkpg(4, 3)
    mkpg(15, 3)
    mkpg(3, 8)

    print path
    pdf.save()




#dynamic_in_frame_sample()
dynamic_table_sample()




