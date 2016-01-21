"""
Customize ReportLab components for pdf creation
"""
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.platypus import Flowable, KeepInFrame, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.rl_config import _FUZZ


__all__ = ['PdfImage', 'DynamicPdfImage', 'DynamicImage', 'DynamicKeepInFrame', 'new_dynamic_image']


class PdfImage(Flowable):
    """PdfImage wraps the first page from a PDF file as a Flowable
    which can be included into a ReportLab Platypus document.
    Based on the vectorpdf extension in rst2pdf (http://code.google.com/p/rst2pdf/)
    """

    def __init__(self, filename_or_object, width=None, height=None, kind='direct', border_color=None):
        self.border_color = border_color

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
        if self.border_color:
            canv.setStrokeColor(self.border_color)
            canv.rect(0, 0, self.drawWidth / xscale, self.drawHeight / yscale)
        canv.restoreState()


class DynamicPdfImage(PdfImage):
    def __init__(self, filename_or_object, hAlign='CENTER', border_color=None):
        PdfImage.__init__(self, filename_or_object, kind='dynamic', border_color=border_color)
        self.hAlign = hAlign


class DynamicKeepInFrame(KeepInFrame):
    def __init__(self, content, maxWidth=0, maxHeight=0, zoom=True, **kwargs):
        """
        :param content: Flowable(s) to show within the Frame
        :param maxHeight:
        :param zoom:
        :param kwargs:
        :return:
        """
        content = content or []
        content = content if hasattr(content, '__iter__') else [content]
        self.zoom = zoom
        self.mh = maxHeight
        self.mw = maxWidth
        KeepInFrame.__init__(self, maxWidth, maxHeight, content=content, **kwargs)

    def wrap(self, aw, ah):
        self.maxHeight = min(self.mh or ah, ah)
        self.maxWidth = min(self.mw or aw, aw)
        w, h = KeepInFrame.wrap(self, aw, ah)
        zoom = self.zoom
        if zoom and w < (aw - _FUZZ) and h < (ah - _FUZZ):
           self._scale = max(w / aw, h / ah)
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

