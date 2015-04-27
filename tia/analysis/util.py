import functools

import numpy as np
import pandas as pd


is_decrease = lambda q1, q2: (q1 * q2) < 0
is_increase = lambda q1, q2: (q1 * q2) > 0
crosses_zero = lambda q1, q2: ((q1 + q2) * q1) < 0


def has_weekends(index):
    """
    :param index: pandas Timestamp index
    :return: True if this index contains weekends
    """
    return 5 in index.dayofweek or 6 in index.dayofweek


class PerLevel(object):
    def __init__(self, fct):
        """Provide logic to apply function to each subframe level

        Parameters
        ----------
        fct: function to apply
        reduce_hdr: when converting from DataFrame to Series its ok to drop bottom level name
        """
        self.fct = fct

    def __call__(self, *args, **kwargs):
        df_or_series = args[0]
        if isinstance(df_or_series, pd.DataFrame) and df_or_series.columns.nlevels > 1:
            df = df_or_series
            pieces = []
            hdrs = set(_[:-1] for _ in df.columns)
            for hdr in hdrs:
                sub = df[hdr]
                res = self.fct(sub, *args[1:], **kwargs)
                if isinstance(res, pd.Series):
                    res = res.to_frame()
                elif not isinstance(res, pd.DataFrame):
                    raise Exception('Expected Series or DataFrame as result not %s' % type(res))

                arrs = [res.columns.get_level_values(lvl) for lvl in range(res.columns.nlevels)]
                names = list(res.columns.names)
                for i in range(df.columns.nlevels - 1):
                    arrs.insert(i, [hdr[i]] * len(res.columns))
                    names.insert(i, df.columns.names[i])

                if self.fct.__name__ == '_frame_to_series':
                    arrs = arrs[:-1]
                    names = names[:-1]
                res.columns = pd.MultiIndex.from_arrays(arrs, names=names)
                pieces.append(res)
            return pd.concat(pieces, axis=1)
        else:
            return self.fct(*args, **kwargs)


class PerSeries(object):
    def __init__(self, fct, result_is_frame=0):
        self.fct = fct
        self.result_is_frame = result_is_frame
        functools.update_wrapper(self, fct)

    def __call__(self, *args, **kwargs):
        df_or_series = args[0]
        if isinstance(df_or_series, (np.ndarray, pd.Series)):  # or len(df_or_series.columns) == 1:
            return self.fct(*args, **kwargs)
        elif not isinstance(df_or_series, pd.DataFrame):
            raise ValueError("Expected argument to be Series or DataFrame not %s" % type(df_or_series))
        else:  # assume dataframe
            df = df_or_series
            if self.result_is_frame:
                pieces = []
                for i, hdrs in enumerate(df.columns):
                    sres = self.fct(df.icol(i), *args[1:], **kwargs)
                    if df.columns.nlevels == 1:
                        arrs = [[hdrs] * len(sres.columns)]
                    else:
                        arrs = [[hdr] * len(sres.columns) for hdr in hdrs]
                    arrs.append(sres.columns)
                    sres.columns = pd.MultiIndex.from_arrays(arrs)
                    pieces.append(sres)
                return pd.concat(pieces, axis=1)
            else:
                return df.apply(self.fct, args=args[1:], **kwargs)


def per_series(result_is_frame=0):
    def _ps(fct):
        return PerSeries(fct, result_is_frame=result_is_frame)

    return _ps


def per_level():
    def _pl(fct):
        return PerLevel(fct)

    return _pl


def insert_level(df, label, level=0, copy=0, axis=0, level_name=None):
    """Add a new level to the index with the specified label. The newly created index will be a MultiIndex.

       :param df: DataFrame
       :param label: label to insert
       :param copy: If True, copy the DataFrame before assigning new index
       :param axis: If 0, then columns. If 1, then index
       :return:
    """
    df = df if not copy else df.copy()
    src = df.columns if axis == 0 else df.index
    current = [src.get_level_values(lvl) for lvl in range(src.nlevels)]
    current.insert(level, [label] * len(src))
    idx = pd.MultiIndex.from_arrays(current)
    level_name and idx.set_names(level_name, level, inplace=1)
    if axis == 0:
        df.columns = idx
    else:
        df.index = idx
    return df