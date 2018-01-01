import unittest
import pandas as pd
import pandas.util.testing as pt
import tia.util.fmt as fmt


def tof(astr):
    return float(astr.replace(',', ''))


class TestFormat(unittest.TestCase):
    def ae(self, expected, fct, value, **kwargs):
        cb = fct(**kwargs)
        actual = cb(value)
        self.assertEqual(expected, actual)

    def test_default_formats(self):
        B = float('-1,250,500,880.76'.replace(',', ''))
        M = B / 1000.
        k = M / 1000.
        p = k / 1000000.
        tests = [
            (B, '$(1.3B)', fmt.BillionDollarsFormatter),
            (B, '(1.3B)', fmt.BillionsFormatter),
            (M, '$(1.3M)', fmt.MillionDollarsFormatter),
            (M, '(1.3M)', fmt.MillionsFormatter),
            (k, '$(1.3k)', fmt.ThousandDollarsFormatter),
            (k, '(1.3k)', fmt.ThousandsFormatter),
            (k, '(1,250.50)', fmt.FloatFormatter),
            (k, '(1,251)', fmt.IntFormatter),
            # Floats
            (k, '-1,251', fmt.new_int_formatter(commas=1, parens=False)),
            (k, '-1251', fmt.new_int_formatter(commas=0, parens=False)),
            (abs(k), '1251', fmt.new_int_formatter(commas=0, parens=False)),
            (abs(k), '1,251', fmt.new_int_formatter(commas=1)),
            (str(k), '-1,251', fmt.new_int_formatter(commas=1, coerce=True, parens=0)),
            # Ints
            (k, '-1,251', fmt.new_int_formatter(commas=1, parens=False)),
            (k, '-1251', fmt.new_int_formatter(commas=0, parens=False)),
            (abs(k), '1251', fmt.new_int_formatter(commas=0, parens=False)),
            (abs(k), '1,251', fmt.new_int_formatter(commas=1)),
            # Percents
            (.12433, '12.4%', fmt.new_percent_formatter(commas=1, precision=1)),
            (.12433, '12.433%', fmt.new_percent_formatter(commas=1, precision=3)),
            (-.12433, '-12.4%', fmt.new_percent_formatter(commas=1, parens=0, precision=1)),
            (-.12433, '(12.4%)', fmt.new_percent_formatter(commas=1, parens=1, precision=1)),
        ]

        for val, expected, fct in tests:
            actual = fct(val)
            self.assertEqual(expected, actual)
            # Test if it were a list
            actual = fct([val]*5)
            self.assertEqual([expected]*5, actual)
            # Test if it were a series
            actual = fct(pd.Series([val]*5))
            pt.assert_series_equal(pd.Series([expected]*5), actual)
            # Test if it were a DataFrame
            actual = fct(pd.DataFrame({'a': [val]*5, 'b': [val]*5}))
            pt.assert_frame_equal(pd.DataFrame({'a': [expected]*5, 'b': [expected]*5}), actual)

    def test_fmt_datetime(self):
        self.assertEqual(fmt.new_datetime_formatter('%Y-%m')(pd.to_datetime('1/1/2013')), '2013-01')

    def test_guess_formatter(self):
        for n, t in (3, 'k'), (6, 'M'), (9, 'B'):
            m = 10 ** n
            s = pd.Series([2.1 * m, -20.1 * m, 200.1 * m])
            actual = fmt.guess_formatter(s, precision=1)(s)
            expected = pd.Series(['2.1' + t, '(20.1%s)' % t, '200.1' + t])
            pt.assert_series_equal(expected, actual)

        # percents
        s = pd.Series([.024, -.561, .987])
        actual = fmt.guess_formatter(s, precision=1, pcts=1)(s)
        expected = pd.Series(['2.4%', '(56.1%)', '98.7%'])
        pt.assert_series_equal(expected, actual)

    def test_dynamic_formatter(self):
        kwargs = dict(precision=1, commas=1, parens=1, pcts=1, trunc_dot_zeros=1)
        byrow = fmt.new_dynamic_formatter('row', **kwargs)
        bycol = fmt.new_dynamic_formatter('col', **kwargs)
        bycell = fmt.new_dynamic_formatter('cell', **kwargs)

        todt = pd.to_datetime
        f = pd.DataFrame(dict(pcts=[.1, .2343, -.9234], flt=[123., 1234., -12345.],
                              ts=[todt('1/1/2012'), todt('1/1/2013'), todt('1/1/2014')]))

        # by column
        expected_bycol = {'pcts': ['10%', '23.4%', '(92.3%)'],
                          'flt': ['123', '1,234', '(12,345)'],
                          'ts': ['2012-01-01', '2013-01-01', '2014-01-01']}


        pt.assert_frame_equal(pd.DataFrame(expected_bycol), bycol(f))
        pt.assert_frame_equal(pd.DataFrame(expected_bycol).T, byrow(f.T))

        expected_bycell = {'pcts': ['10%', '23.4%', '(92.3%)'],
                          'flt': ['123', '1.2k', '(12.3k)'],
                          'ts': ['2012-01-01', '2013-01-01', '2014-01-01']}
        pt.assert_frame_equal(pd.DataFrame(expected_bycell), bycell(f))
        # ensure nothing different
        pt.assert_frame_equal(pd.DataFrame(expected_bycell).T, bycell(f.T))






