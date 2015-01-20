from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


class Font(object):
    def __init__(self, name, fname):
        self.name = name
        self.fname = fname

    def try_load(self, default=None):
        try:
            pdfmetrics.registerFont(TTFont(self.name, self.fname))
            return self.name
        except:
            return default

