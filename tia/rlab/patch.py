"""
This hack is necessary in order to force tables to fit into a frame. Without this hack, the width will spill
over the boundaries of the frame.
"""
import reportlab.platypus.flowables as fl


def _listWrapOn(F, availWidth, canv, mergeSpace=1, obj=None, dims=None):
    '''return max width, required height for a list of flowables F'''
    doct = getattr(canv, '_doctemplate', None)
    cframe = getattr(doct, 'frame', None)
    if cframe:
        from reportlab.platypus.doctemplate import _addGeneratedContent, Indenter

        doct_frame = cframe
        from copy import deepcopy

        cframe = doct.frame = deepcopy(doct_frame)
        cframe._generated_content = None
        del cframe._generated_content
    try:
        W = 0
        H = 0
        pS = 0
        atTop = 1
        F = F[:]
        while F:
            f = F.pop(0)
            if hasattr(f, 'frameAction'):
                from reportlab.platypus.doctemplate import Indenter

                if isinstance(f, Indenter):
                    availWidth -= f.left + f.right
                continue
            w, h = f.wrapOn(canv, availWidth, 0xfffffff)
            if dims is not None: dims.append((w, h))
            if cframe:
                _addGeneratedContent(F, cframe)
            if w <= fl._FUZZ or h <= fl._FUZZ: continue
            #
            # THE HACK
            #
            # W = max(W,min(availWidth, w))
            W = max(W, w)
            H += h
            if not atTop:
                h = f.getSpaceBefore()
                if mergeSpace:
                    if getattr(f, '_SPACETRANSFER', False):
                        h = pS
                    h = max(h - pS, 0)
                H += h
            else:
                if obj is not None: obj._spaceBefore = f.getSpaceBefore()
                atTop = 0
            s = f.getSpaceAfter()
            if getattr(f, '_SPACETRANSFER', False):
                s = pS
            pS = s
            H += pS
        if obj is not None: obj._spaceAfter = pS
        return W, H - pS
    finally:
        if cframe:
            doct.frame = doct_frame

# Hack in order to get width constrained
#fl._listWrapOn = _listWrapOn
