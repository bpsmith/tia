"""
Provide a simple way to interfacing with the Bloomberg API. Provide functions for aggregating and caching
"""
import os
from collections import OrderedDict

import pandas as pd

from tia.bbg import LocalTerminal
import tia.util.log as log


__all__ = ['DataManager', 'BbgDataManager', 'MemoryStorage', 'HDFStorage', 'CachedDataManager', 'Storage',
           'CacheOnlyDataManager', 'SidAccessor', 'MultiSidAccessor']

_force_array = lambda x: isinstance(x, str) and [x] or x


class SidAccessor(object):
    """ Bloomberg API accessor for a single security id (SID). """
    def __init__(self, sid, mgr, **overrides):
        self.sid = sid
        self.yellow_key = sid.split()[-1]
        self.sid_no_yellow_key = sid.rsplit(' ', 1)[0]
        self.mgr = mgr
        self.overrides = overrides or {}

    def __getattribute__(self, item):
        """Access bloomberg fields directly by using all upper case field names"""
        if item.isupper():
            return self.get_attributes(item, **self.overrides)
        else:
            return object.__getattribute__(self, item)

    def get_attributes(self, flds, **overrides):
        frame = self.mgr.get_attributes(self.sid, flds, **overrides)
        if self.mgr.sid_result_mode == 'frame':
            return frame
        else:
            if isinstance(flds, str):
                return frame.iloc[0, 0]
            else:
                return frame.values[0].tolist()

    def __getitem__(self, flds):
        return self.get_attributes(flds, **self.overrides)

    def get_historical(self, flds, start, end, period=None, **overrides):
        return self.mgr.get_historical(self.sid, flds, start, end, period, **overrides)

    @property
    def currency(self):
        curr = self['CRNCY']
        sid = '%s CURNCY' % curr
        return self.mgr[sid]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.sid)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.sid.upper() == other.sid.upper()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.sid.__hash__()


class MultiSidAccessor(object):
    """ Bloomberg API accessor for multiple security ids """
    def __init__(self, sids, mgr, **overrides):
        self.sids = sids
        self.yellow_keys = pd.Series({sid: sid.split()[-1] for sid in sids})
        self.sid_no_yellow_keys = pd.Series({sid: sid.rsplit(' ', 1)[0] for sid in sids})
        self.mgr = mgr
        self.overrides = overrides or {}

    def __getattribute__(self, item):
        """Access bloomberg fields directly by using all upper case field names"""
        if item.isupper():
            return self.get_attributes(item, **self.overrides)
        else:
            return object.__getattribute__(self, item)

    def get_attributes(self, flds, **overrides):
        frame = self.mgr.get_attributes(self.sids, flds, **overrides)
        return frame

    def __getitem__(self, flds):
        return self.get_attributes(flds, **self.overrides)

    def get_historical(self, flds, start, end, period=None, **overrides):
        return self.mgr.get_historical(self.sids, flds, start, end, period, **overrides)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ','.join(self.sids))


class DataManager(object):
    RESULT_MODE_VALUES = 'values'
    RESULT_MODE_FRAME = 'frame'

    def __init__(self, sid_result_mode=None):
        self._sid_result_mode = sid_result_mode or self.RESULT_MODE_VALUES

    def get_attributes(self, sids, flds, **overrides):
        raise NotImplementedError('must implement get_attributes')

    def get_historical(self, sids, flds, start, end, period=None, **overrides):
        raise NotImplementedError('must implement get_historical')

    def get_sid_accessor(self, sid, **overrides):
        if isinstance(sid, str):
            return SidAccessor(sid, self, **overrides)
        else:
            return MultiSidAccessor(sid, self, **overrides)

    def __getitem__(self, sid):
        return self.get_sid_accessor(sid)

    @property
    def sid_result_mode(self):
        return self._sid_result_mode

    @sid_result_mode.setter
    def sid_result_mode(self, value):
        self._sid_result_mode = value


class BbgDataManager(DataManager):
    def __init__(self, terminal=None, sid_result_mode=None):
        """ Provide simple access to the Bloomberg API.

        Parameters
        ----------
        terminal : Terminal, default to None
                    If None, then use the default LocalTerminal object defined in the bbg package
        sid_result_mode: (values|frame) values will return tuples, frame returns DataFrame
        """
        DataManager.__init__(self, sid_result_mode)
        self.terminal = terminal or LocalTerminal

    def get_attributes(self, sids, flds, **overrides):
        return self.terminal.get_reference_data(sids, flds, **overrides).as_frame()

    def get_historical(self, sids, flds, start, end, period=None, **overrides):
        end = end
        start = start
        frame = self.terminal.get_historical(sids, flds, start=start, end=end, period=period, **overrides).as_frame()
        if isinstance(sids, str):
            return frame[sids]
        else:  # multi-indexed frame
            if isinstance(flds, str):
                frame.columns = frame.columns.droplevel(1)
            return frame


class Storage(object):
    def key_to_string(self, key):
        def _to_str(val):
            if hasattr(val, 'iteritems'):
                if val:
                    # Sort keys to keep order and drop any null values
                    tmp = ','.join(['{0}={1}'.format(k, _to_str(val[k])) for k in sorted(val.keys()) if val[k]])
                    return tmp if tmp else str(None)
                else:
                    return str(None)
            elif isinstance(val, (tuple, list)):
                if val:
                    tmp = ','.join([_to_str(_) for _ in val])
                    return tmp if tmp else str(None)
                else:
                    return str(None)
            else:
                sval = str(val)
                return sval.replace('/', '-')

        if isinstance(key, (list, tuple)):
            return '/'.join([_to_str(v) for v in key])
        else:
            return _to_str(key)


class MemoryStorage(Storage):
    def __init__(self):
        self._cache = {}

    def get(self, key, default=(None, None)):
        strkey = self.key_to_string(key)
        return self._cache.get(strkey, default)

    def set(self, key, frame, **data):
        strkey = self.key_to_string(key)
        self._cache[strkey] = (frame, data)


class HDFStorage(Storage):
    def __init__(self, hdfpath, readonly=0, complevel=None, complib=None, fletcher32=False, format=None):
        self.hdfpath = hdfpath
        self.readonly = readonly
        self._file_exists = None
        self._store = None
        self.format = format
        self.get_store_kwargs = {'complevel': complevel, 'complib': complib, 'fletcher32': fletcher32}

    def get_store(self, write=0):
        if self._store is not None:
            return self._store
        else:
            if write:
                return pd.HDFStore(self.hdfpath, mode=self.file_exists and 'a' or 'w', **self.get_store_kwargs)
            else:
                return pd.HDFStore(self.hdfpath, **self.get_store_kwargs)

    @property
    def file_exists(self):
        exists = self._file_exists
        if not exists:
            exists = os.path.exists(self.hdfpath)
            self._file_exists = exists
        return exists

    def get(self, key, default=None):
        if self.file_exists:
            store = None
            managed = self._store is None
            try:
                store = self.get_store(write=0)
                path = self.key_to_string(key)
                if path in store:
                    df = store[path]
                    storer = store.get_storer(path)
                    if 'userdata' in storer.attrs:
                        userdata = storer.attrs.userdata
                    else:
                        userdata = {}
                    return df, userdata
            finally:
                if store is not None and managed:
                    store.close()
        return None, None

    def set(self, key, frame, **userdata):
        if self.readonly:
            raise Exception('storage is read-only')
        else:
            store = None
            managed = self._store is None
            try:
                #
                # if format is table and all NaN values, then an empty table is stored. Stop this from occuring.
                #
                if self.format == 'table' and frame.isnull().all().all():
                    frame['__FAKE_DATA__'] = 1

                store = self.get_store(write=1)
                path = self.key_to_string(key)
                store.put(path, frame, format=self.format)
                if userdata:
                    store.get_storer(path).attrs.userdata = userdata
            finally:
                if store is not None and managed:
                    store.close()


class CacheMissError(Exception):
    """Raised when cache lookup fails and there is no fallback"""


class CacheOnlyDataManager(DataManager):
    def get_attributes(self, sids, flds, **overrides):
        sids = _force_array(sids)
        flds = _force_array(flds)
        sstr = ','.join(sids)
        fstr = ','.join(flds)
        ostr = ''
        if overrides:
            ostr = ', overrides=' + ','.join(['{0}={1}'.format(str(k), str(v)) for k, v in overrides.items()])
        msg = 'Reference data for sids={0}, flds={1}{2}'.format(sstr, fstr, ostr)
        raise CacheMissError(msg)

    def get_historical(self, sids, flds, start, end, period=None, **overrides):
        sids = _force_array(sids)
        flds = _force_array(flds)
        sstr = ','.join(sids)
        fstr = ','.join(flds)
        msg = 'Historical data for sids={0}, flds={1}, start={2}, end={3}, period={4}'.format(sstr, fstr, start, end,
                                                                                              period)
        raise CacheMissError(msg)


class CachedDataManager(DataManager):
    def __init__(self, dm, storage, ts=None):
        """
        :param dm: DataManager, if not available in cache then use dm to request data
        :param storage: Storage for the cached data
        :param ts:
        """
        DataManager.__init__(self)
        self.dm = dm
        self.storage = storage
        self.ts = ts or pd.datetime.now()
        self.logger = log.instance_logger('cachemgr', self)

    @staticmethod
    def no_fallback(storage, ts=None):
        return CachedDataManager(CacheOnlyDataManager(), storage, ts)

    @property
    def sid_result_mode(self):
        return self.dm.sid_result_mode

    @sid_result_mode.setter
    def sid_result_mode(self, value):
        self.dm.sid_result_mode = value

    def _cache_get_attribute(self, sids, flds, **overrides):
        if isinstance(sids, str):
            key = (sids, 'attributes', overrides)
            vframe, userdata = self.storage.get(key)
            if vframe is not None:
                # do this to keep order
                matches = [c for c in flds if c in vframe.columns]
                if matches:
                    return vframe[matches]
        else:
            matches = [self._cache_get_attribute(sid, flds, **overrides) for sid in sids]
            # Keep order so don't have to sort after the fact
            res = OrderedDict()
            for sid, match in zip(sids, matches):
                if match is not None:
                    res[sid] = match
            return res

    def _cache_update_attribute(self, sid, frame, **overrides):
        key = (sid, 'attributes', overrides)
        oframe, data = self.storage.get(key)
        if oframe is not None:
            frame = pd.concat([oframe, frame], axis=1)
        self.storage.set(key, frame, **overrides)

    def get_attributes(self, sids, flds, **overrides):
        """Check cache first, then defer to data manager
        :param sids: security identifiers
        :param flds: fields to retrieve
        :param overrides: key-value pairs to pass to the mgr get_attributes method
        :return: DataFrame with flds as columns and sids as the row indices
        """
        # Unfortunately must be inefficient with request
        flds = _force_array(flds)
        sids = _force_array(sids)
        cached = self._cache_get_attribute(sids, flds, **overrides)
        if not cached:  # build get
            df = self.dm.get_attributes(sids, flds, **overrides)
            [self._cache_update_attribute(sid, df.ix[sid:sid], **overrides) for sid in sids]
            return df
        else:
            # Retrieve all missing and merge with existing cache
            for sid in sids:
                missed = flds if sid not in cached else set(flds) - set(cached[sid].columns)
                if missed:
                    df = self.dm.get_attributes(sid, missed, **overrides)
                    self._cache_update_attribute(sid, df, **overrides)

            # now just retrieve from cache
            data = self._cache_get_attribute(sids, flds, **overrides)
            # reindex and grab columns to sort
            frame = pd.concat(list(data.values()))
            return frame

    def _date_only(self, ts_or_period):
        if isinstance(ts_or_period, pd.Period):
            return ts_or_period.to_timestamp()
        else:
            ts = pd.to_datetime(ts_or_period)
            return ts.to_period('D').to_timestamp()

    def get_historical(self, sids, flds, start, end, period=None, **overrides):
        # TODO - Revisit date handling for caching
        is_str = isinstance(sids, str)
        is_fld_str = isinstance(flds, str)
        flds = _force_array(flds)
        sids = _force_array(sids)
        end = (end and self._date_only(end)) or self._date_only(self.ts)
        start = self._date_only(start)
        frames = {}

        for sid in sids:
            key = (sid, 'historical', dict(period=period))
            if overrides:
                    for k, v in overrides.items():
                        key[2][k] = v

            cached_frame, userdata = self.storage.get(key)
            if cached_frame is None:
                frame = self.dm.get_historical(sid, flds, start, end, **overrides)
                self.storage.set(key, frame, start=start, end=end)
                frames[sid] = frame
            else:
                cache_start = userdata.get('start', cached_frame.index[0])
                cache_end = userdata.get('end', cached_frame.index[-1])
                cache_columns = cached_frame.columns
                requested_columns = pd.Index(flds)
                missing_columns = requested_columns - cache_columns
                dirty = 0
                # Ensure any currently stored fields are kept in synch with dates
                if start < cache_start:
                    self.logger.info('%s request for %s is older than data in cache %s' % (sid, ','.join(cache_columns),
                                                                                           cache_start))
                    previous = self.dm.get_historical(sid, cache_columns, start, cache_start)
                    # Easy way to ensure we don't dup data
                    previous = previous.ix[previous.index < cache_start]
                    if len(previous.index) > 0:
                        cached_frame = pd.concat([previous, cached_frame])
                        dirty = 1
                if end > cache_end:
                    ccols = ','.join(cache_columns)
                    self.logger.info('%s request for %s is more recent than data in cache %s' % (sid, ccols, cache_end))
                    post = self.dm.get_historical(sid, cache_columns, cache_end, end)
                    # Easy way to ensure we don't dup data
                    post = post.ix[post.index > cache_end]
                    if len(post.index) > 0:
                        cached_frame = pd.concat([cached_frame, post])
                        dirty = 1

                if dirty:
                    cached_frame.sort_index()

                if len(missing_columns) > 0:
                    # For missing need to get maximum range to match cache. Don't want to manage pieces
                    self.logger.info('%s: %s not in cache, requested for dates %s to %s' % (sid,
                                                                                            ','.join(missing_columns),
                                                                                            min(cache_start, start),
                                                                                            max(cache_end, end)))
                    newdata = self.dm.get_historical(sid, missing_columns, min(cache_start, start), max(end, cache_end))
                    cached_frame = pd.concat([cached_frame, newdata], axis=1)
                    dirty = 1

                dirty and self.storage.set(key, cached_frame, start=min(cache_start, start), end=max(cache_end, end))
                frames[sid] = cached_frame.ix[start:end, flds]

        if is_str:
            return frames[sids[0]]
        else:
            result = pd.concat(list(frames.values()), keys=list(frames.keys()), axis=1)
            if is_fld_str:
                result.columns = result.columns.droplevel(1)
            return result




