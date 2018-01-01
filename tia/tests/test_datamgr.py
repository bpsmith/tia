import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdtest
from tia.bbg.datamgr import CachedDataManager, MemoryStorage, HDFStorage


as_date = pd.to_datetime


class MockDataManager(object):
    def __init__(self):
        data = {
            'FLDA': [1, 2, 3],
            'FLDB': ['a', 'b', 'c'],
            'FLDC': [99., 98., 97.]
        }
        self.df = pd.DataFrame(data, index=['SID1', 'SID2', 'SID3'])
        self.access_cnt = 0

        data = {
            'FLDA': [1, 2, 3, 4],
            'FLDB': ['a', 'b', 'c', 'd'],
            'FLDC': [99., 98., 97., 96.]
        }
        index = pd.date_range('1/1/2014', '1/4/2014')
        self.hist = {'SID%s' % i: pd.DataFrame(data, index=index) for i in range(1, 4)}
        self.access_cnt = 0

    def get_attributes(self, sids, flds, **overrides):
        sids = [sids] if isinstance(sids, str) else sids
        flds = [flds] if isinstance(flds, str) else flds
        self.access_cnt += 1
        return self.df.ix[sids, flds]

    def get_historical(self, sid, flds, start, end, **overrides):
        self.access_cnt += 1
        return self.hist[sid].ix[start:end, flds]


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.dm = MockDataManager()

    def _do_cache_test(self, storage):
        cdm = CachedDataManager(self.dm, storage, pd.datetime.now())
        sids = ['SID1', 'SID2', 'SID3']
        flds = ['FLDA', 'FLDB', 'FLDC']
        res = cdm.get_attributes(sids, flds)
        pdtest.assert_frame_equal(res, self.dm.df)
        self.assertEqual(1, self.dm.access_cnt)
        # try when reordering
        sids = ['SID3', 'SID1', 'SID2']
        flds = ['FLDB', 'FLDA', 'FLDC']
        res = cdm.get_attributes(sids, flds)
        pdtest.assert_frame_equal(res, self.dm.df.ix[sids, flds])
        self.assertEqual(1, self.dm.access_cnt)
        # miss the cache by setting an override which is none - this should not effect anything
        res = cdm.get_attributes(sids, flds, fake=None)
        pdtest.assert_frame_equal(res, self.dm.df.ix[sids, flds])
        self.assertEqual(1, self.dm.access_cnt)
        # REAL cache miss
        res = cdm.get_attributes(sids, flds, fake='value')
        pdtest.assert_frame_equal(res, self.dm.df.ix[sids, flds])
        self.assertEqual(2, self.dm.access_cnt)

    def test_memory_cache(self):
        self._do_cache_test(MemoryStorage())

    def test_hdf_cache(self):
        import tempfile
        path = tempfile.mktemp('h5', 'hdftest')
        self._do_cache_test(HDFStorage(path))

    def test_cache_sub(self):
        cdm = CachedDataManager(self.dm, MemoryStorage(), pd.datetime.now())
        # get a single field for each sid
        for i, (sid, fld) in enumerate([('SID1', 'FLDA'), ('SID2', 'FLDB'), ('SID3', 'FLDC')]):
            res = cdm.get_attributes(sid, fld)
            pdtest.assert_frame_equal(res, self.dm.df.ix[sid:sid, fld:fld])
            self.assertEqual(i+1, self.dm.access_cnt)
        # now force cache to make multiple requests to build entire frame
        sids = ['SID1', 'SID2', 'SID3']
        flds = ['FLDA', 'FLDB', 'FLDC']
        res = cdm.get_attributes(sids, flds)
        pdtest.assert_frame_equal(res, self.dm.df)
        # should be since each sid must be made whjole individually
        self.assertEqual(6, self.dm.access_cnt)

    def _do_historical_cache_test(self, storage):
        # Cache pieces and then request entire and ensure cache is built properly
        cdm = CachedDataManager(self.dm, storage, pd.datetime.now())
        start, end = pd.to_datetime('1/2/2014'), pd.to_datetime('1/3/2014')
        res = cdm.get_historical('SID1', 'FLDA', start, end)
        pdtest.assert_frame_equal(res, self.dm.hist['SID1'].ix[start:end, ['FLDA']])
        self.assertEqual(1, self.dm.access_cnt)

        start, end = pd.to_datetime('1/1/2014'), pd.to_datetime('1/4/2014')
        res = cdm.get_historical('SID3', ['FLDA', 'FLDB', 'FLDC'], start, end)
        pdtest.assert_frame_equal(res, self.dm.hist['SID3'])
        self.assertEqual(2, self.dm.access_cnt)

        #
        # Request entire cache:
        # SID1 - should result in 3 calls, 1 for prepending dates, 1 for appending dates, 1 for missing columns
        # SID2 - should result in 1 call for entire frame
        # SID3 - should result in 0 calls since it should be cached
        start, end = as_date('1/1/2014'), as_date('1/4/2014')
        res = cdm.get_historical(['SID1', 'SID2', 'SID3'], ['FLDA', 'FLDB', 'FLDC'], start, end)
        self.assertEqual(6, self.dm.access_cnt)
        pdtest.assert_frame_equal(res['SID1'], self.dm.hist['SID1'])
        pdtest.assert_frame_equal(res['SID2'], self.dm.hist['SID2'])
        pdtest.assert_frame_equal(res['SID3'], self.dm.hist['SID3'])

    def test_historical_memory_cache(self):
        self._do_historical_cache_test(MemoryStorage())

    def test_historical_hdf_cache(self):
        import tempfile
        path = tempfile.mktemp('h5', 'hdftest')
        self._do_historical_cache_test(HDFStorage(path))

    def test_hdf_nan(self):
        df = pd.DataFrame({'FLDA': np.array([1., np.nan], dtype=float)}, index=['SID1', 'SID2'])
        self.access_cnt = 0

        data = {
            'FLDA': [1, 2, 3, 4],
            'FLDB': ['a', 'b', 'c', 'd'],
            'FLDC': [99., 98., 97., 96.]
        }
        index = pd.date_range('1/1/2014', '1/4/2014')
        self.hist = {'SID%s' % i: pd.DataFrame(data, index=index) for i in range(1, 4)}
        self.access_cnt = 0

    def _do_request_returns_less_cached(self, storage):
        """Request data for dates d1 to d2, but the returning data starts d1 + delta (imagine stock which just got listed)
        An issue because cache will attempt to get missing date, so mechanism must exist to store not on the data
        but the request for the data
        """
        cdm = CachedDataManager(self.dm, storage, pd.datetime.now())
        start, end = as_date('12/31/2013'), as_date('1/3/2014')
        res = cdm.get_historical('SID1', 'FLDA', start, end)
        pdtest.assert_frame_equal(res, self.dm.hist['SID1'].ix[start:end, ['FLDA']])
        self.assertEqual(1, self.dm.access_cnt)

        res = cdm.get_historical('SID1', 'FLDA', start, end)
        self.assertEqual(1, self.dm.access_cnt)

    def test_request_returns_less_cached(self):
        self._do_request_returns_less_cached(MemoryStorage())

    def test_request_returns_less_cached(self):
        import tempfile
        path = tempfile.mktemp('h5', 'hdftest')
        self._do_request_returns_less_cached(HDFStorage(path))
