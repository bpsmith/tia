"""
Customize ReportLab components for pdf creation
"""
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.platypus import Flowable, KeepInFrame, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


__all__ = ['PdfImage', 'DynamicPdfImage', 'DynamicImage', 'DynamicKeepInFrame', 'new_dynamic_image']


class PdfImage(Flowable):
    """PdfImage wraps the first page from a PDF file as a Flowable
    which can be included into a ReportLab Platypus document.
    Based on the vectorpdf extension in rst2pdf (http://code.google.com/p/rst2pdf/)
    """

    def __init__(self, filename_or_object, width=None, height=None, kind='direct'):
        if hasattr(filename_or_object, 'read'):
            filename_or_object.seek(0)
        page = PdfReader(filename_or_object, decompress=False).pages[0]
        self.xobj = pagexobj(page)
        self.dynamic = 0
        # Actual image size
        x1, y1, x2, y2 = self.xobj.BBox
        imgw, imgh = x2 - x1, y2 - y1
        self._imgw, self._imgh = imgw, imgh
        if kind in ['direct', 'absolute']:
            self.drawWidth = width or imgw
            self.drawHeight = height or imgh
        elif kind in ['percentage', '%']:
            self.drawWidth = imgw * width * 0.01
            self.drawHeight = imgh * height * 0.01
        elif kind in ['bound', 'proportional']:
            w, h = width or imgw, height or imgh
            factor = min(float(w) / imgw, float(h) / imgh)
            self.drawWidth = imgw * factor
            self.drawHeight = imgh * factor
        elif kind in ['dynamic']:
            self.dynamic = 1

    def wrap(self, aw, ah):
        if self.dynamic:
            wr, wh = self._imgw / aw, self._imgh / ah
            r = max(wr, wh)
            self.drawHeight = self._imgh / r
            self.drawWidth = self._imgw / r

        return self.drawWidth, self.drawHeight

    def drawOn(self, canv, x, y, _sW=0):
        if _sW > 0 and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5 * _sW
            elif a in ('RIGHT', TA_RIGHT):
                x += _sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))

        xobj = self.xobj
        xobj_name = makerl(canv._doc, xobj)
        xscale = self.drawWidth / self._imgw
        yscale = self.drawHeight / self._imgh
        canv.saveState()
        canv.translate(x, y)
        canv.scale(xscale, yscale)
        canv.doForm(xobj_name)
        canv.restoreState()


class DynamicPdfImage(PdfImage):
    def __init__(self, filename_or_object, hAlign='CENTER'):
        PdfImage.__init__(self, filename_or_object, kind='dynamic')
        self.hAlign = hAlign


class DynamicKeepInFrame(KeepInFrame):
    def __init__(self, content=[], maxWidth=0, maxHeight=0, zoom=1, **kwargs):
        if content and not hasattr(content, '__iter__'):
            raise ValueError('content expected to be a list of flowables')
        self.zoom = zoom
        KeepInFrame.__init__(self, maxWidth, maxHeight, content=content, **kwargs)

    def wrap(self, awidth, aheight):
        self.maxHeight = aheight
        self.maxWidth = awidth
        w, h = KeepInFrame.wrap(self, awidth, aheight)
        if w < awidth and h < aheight:
            self._scale = max(w / awidth, h / aheight)
        return w, h


class DynamicImage(Image):
    def __init__(self, path):
        Image.__init__(self, path)
        import PIL

        w, h = PIL.Image.open(path).size
        self.maxw = w
        self.maxh = h

    def wrap(self, awidth, aheight):
        wr, wh = self.maxw / awidth, self.maxh / aheight
        if wr > 1 or wh > 1:
            r = max(wr, wh)
            self.drawHeight = self.maxh / r
            self.drawWidth = self.maxw / r
        return Image.wrap(self, awidth, aheight)


def new_dynamic_image(path, hAlign=None):
    if path.lower().endswith('pdf'):
        return DynamicPdfImage(path)
    else:
        img = DynamicImage(path)
        hAlign and setattr(img, 'hAlign', hAlign)
        return img

