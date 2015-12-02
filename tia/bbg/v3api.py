from collections import defaultdict, namedtuple
from datetime import datetime

import blpapi
import pandas as pd
import numpy as np

import tia.util.log as log


SecurityErrorAttrs = ['security', 'source', 'code', 'category', 'message', 'subcategory']
SecurityError = namedtuple('SecurityError', SecurityErrorAttrs)
FieldErrorAttrs = ['security', 'field', 'source', 'code', 'category', 'message', 'subcategory']
FieldError = namedtuple('FieldError', FieldErrorAttrs)

logger = log.get_logger(__name__)

__all__ = ['Terminal']


class XmlHelper(object):
    @staticmethod
    def security_iter(nodearr):
        """ provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive """
        assert nodearr.name() == 'securityData' and nodearr.isArray()
        for i in range(nodearr.numValues()):
            node = nodearr.getValue(i)
            err = XmlHelper.get_security_error(node)
            result = (None, err) if err else (node, None)
            yield result

    @staticmethod
    def node_iter(nodearr):
        assert nodearr.isArray()
        for i in range(nodearr.numValues()):
            yield nodearr.getValue(i)

    @staticmethod
    def message_iter(evt):
        """ provide a message iterator which checks for a response error prior to returning """
        for msg in evt:
            if logger.isEnabledFor(log.logging.DEBUG):
                logger.debug(msg.toString())
            if msg.asElement().hasElement('responseError'):
                raise Exception(msg.toString())
            yield msg

    @staticmethod
    def get_sequence_value(node):
        """Convert an element with DataType Sequence to a DataFrame.
        Note this may be a naive implementation as I assume that bulk data is always a table
        """
        assert node.datatype() == 15
        data = defaultdict(list)
        cols = []
        for i in range(node.numValues()):
            row = node.getValue(i)
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(row.getElement(_).name()) for _ in range(row.numElements())]

            for cidx in range(row.numElements()):
                col = row.getElement(cidx)
                data[str(col.name())].append(XmlHelper.as_value(col))
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def as_value(ele):
        """ convert the specified element as a python value """
        dtype = ele.datatype()
        # print '%s = %s' % (ele.name(), dtype)
        if dtype in (1, 2, 3, 4, 5, 6, 7, 9, 12):
            # BOOL, CHAR, BYTE, INT32, INT64, FLOAT32, FLOAT64, BYTEARRAY, DECIMAL)
            return ele.getValue()
        elif dtype == 8:  # String
            val = ele.getValue()
            """
            if val:
                # us centric :)
                val = val.encode('ascii', 'replace')
            """
            return str(val)
        elif dtype == 10:  # Date
            if ele.isNull():
                return pd.NaT
            else:
                v = ele.getValue()
                return datetime(year=v.year, month=v.month, day=v.day) if v else pd.NaT
        elif dtype == 11:  # Time
            if ele.isNull():
                return pd.NaT
            else:
                v = ele.getValue()
                now = pd.datetime.now()
                return datetime(year=now.year, month=now.month, day=now.day, hour=v.hour, minute=v.minute, second=v.second).time() if v else np.nan
        elif dtype == 13:  # Datetime
            if ele.isNull():
                return pd.NaT
            else:
                v = ele.getValue()
                return v
        elif dtype == 14:  # Enumeration
            # raise NotImplementedError('ENUMERATION data type needs implemented')
            return str(ele.getValue())
        elif dtype == 16:  # Choice
            raise NotImplementedError('CHOICE data type needs implemented')
        elif dtype == 15:  # SEQUENCE
            return XmlHelper.get_sequence_value(ele)
        else:
            raise NotImplementedError('Unexpected data type %s. Check documentation' % dtype)

    @staticmethod
    def get_child_value(parent, name, allow_missing=0):
        """ return the value of the child element with name in the parent Element """
        if not parent.hasElement(name):
            if allow_missing:
                return np.nan
            else:
                raise Exception('failed to find child element %s in parent' % name)
        else:
            return XmlHelper.as_value(parent.getElement(name))

    @staticmethod
    def get_child_values(parent, names):
        """ return a list of values for the specified child fields. If field not in Element then replace with nan. """
        vals = []
        for name in names:
            if parent.hasElement(name):
                vals.append(XmlHelper.as_value(parent.getElement(name)))
            else:
                vals.append(np.nan)
        return vals

    @staticmethod
    def as_security_error(node, secid):
        """ convert the securityError element to a SecurityError """
        assert node.name() == 'securityError'
        src = XmlHelper.get_child_value(node, 'source')
        code = XmlHelper.get_child_value(node, 'code')
        cat = XmlHelper.get_child_value(node, 'category')
        msg = XmlHelper.get_child_value(node, 'message')
        subcat = XmlHelper.get_child_value(node, 'subcategory')
        return SecurityError(security=secid, source=src, code=code, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def as_field_error(node, secid):
        """ convert a fieldExceptions element to a FieldError or FieldError array """
        assert node.name() == 'fieldExceptions'
        if node.isArray():
            return [XmlHelper.as_field_error(node.getValue(_), secid) for _ in range(node.numValues())]
        else:
            fld = XmlHelper.get_child_value(node, 'fieldId')
            info = node.getElement('errorInfo')
            src = XmlHelper.get_child_value(info, 'source')
            code = XmlHelper.get_child_value(info, 'code')
            cat = XmlHelper.get_child_value(info, 'category')
            msg = XmlHelper.get_child_value(info, 'message')
            subcat = XmlHelper.get_child_value(info, 'subcategory')
            return FieldError(security=secid, field=fld, source=src, code=code, category=cat, message=msg,
                              subcategory=subcat)

    @staticmethod
    def get_security_error(node):
        """ return a SecurityError if the specified securityData element has one, else return None """
        assert node.name() == 'securityData' and not node.isArray()
        if node.hasElement('securityError'):
            secid = XmlHelper.get_child_value(node, 'security')
            err = XmlHelper.as_security_error(node.getElement('securityError'), secid)
            return err
        else:
            return None

    @staticmethod
    def get_field_errors(node):
        """ return a list of FieldErrors if the specified securityData element has field errors """
        assert node.name() == 'securityData' and not node.isArray()
        nodearr = node.getElement('fieldExceptions')
        if nodearr.numValues() > 0:
            secid = XmlHelper.get_child_value(node, 'security')
            errors = XmlHelper.as_field_error(nodearr, secid)
            return errors
        else:
            return None


def debug_event(evt):
    print 'unhandled event: %s' % evt.EventType
    if evt.EventType in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
        print 'messages:'
        for msg in XmlHelper.message_iter(evt):
            print msg.Print


class Request(object):
    def __init__(self, svcname, ignore_security_error=0, ignore_field_error=0):
        self.field_errors = []
        self.security_errors = []
        self.ignore_security_error = ignore_security_error
        self.ignore_field_error = ignore_field_error
        self.svcname = svcname
        self.response = None

    def new_response(self):
        raise NotImplementedError('subclass must implement')

    @property
    def has_exception(self):
        if not self.ignore_security_error and len(self.security_errors) > 0:
            return True
        if not self.ignore_field_error and len(self.field_errors) > 0:
            return True

    def raise_exception(self):
        if not self.ignore_security_error and len(self.security_errors) > 0:
            msgs = ['(%s, %s, %s)' % (s.security, s.category, s.message) for s in self.security_errors]
            raise Exception('SecurityError: %s' % ','.join(msgs))
        if not self.ignore_field_error and len(self.field_errors) > 0:
            msgs = ['(%s, %s, %s, %s)' % (s.security, s.field, s.category, s.message) for s in self.field_errors]
            raise Exception('FieldError: %s' % ','.join(msgs))
        raise Exception('Programmer Error: No exception to raise')

    def get_bbg_request(self, svc, session):
        raise NotImplementedError()

    def on_event(self, evt, is_final):
        raise NotImplementedError()

    def on_admin_event(self, evt):
        pass

    @staticmethod
    def apply_overrides(request, overrides):
        if overrides:
            for k, v in overrides.iteritems():
                o = request.getElement('overrides').appendElement()
                o.setElement('fieldId', k)
                o.setElement('value', v)

    def set_flag(self, request, val, fld):
        """If the specified val is not None, then set the specified field to its boolean value"""
        if val is not None:
            val = bool(val)
            request.set(fld, val)

    def set_response(self, response):
        """Set the response to handle and store the results """
        self.response = response


class HistoricalDataResponse(object):
    def __init__(self, request):
        self.request = request
        self.response_map = {}

    def on_security_complete(self, sid, frame):
        self.response_map[sid] = frame

    def as_panel(self):
        return pd.Panel(self.response_map)

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """ :return: Multi-Index DataFrame """
        sids, frames = self.response_map.keys(), self.response_map.values()
        frame = pd.concat(frames, keys=sids, axis=1)
        return frame


class HistoricalDataRequest(Request):
    """A class which manages the creation of the Bloomberg HistoricalDataRequest and
    the processing of the associated Response.

    Parameters
    ----------
    sids: bbg security identifier(s)
    fields: bbg field name(s)
    start: (optional) date, date string , or None. If None, defaults to 1 year ago.
    end: (optional) date, date string, or None. If None, defaults to today.
    period: (optional) periodicity of data [DAILY, WEEKLY, MONTHLY, QUARTERLY, SEMI-ANNUAL, YEARLY]
    ignore_security_error: If True, ignore exceptions caused by invalid sids
    ignore_field_error: If True, ignore exceptions caused by invalid fields
    period_adjustment: (ACTUAL, CALENDAR, FISCAL)
                        Set the frequency and calendar type of the output
    currency: ISO Code
              Amends the value from local to desired currency
    override_option: (OVERRIDE_OPTION_CLOSE | OVERRIDE_OPTION_GPA)
    pricing_option: (PRICING_OPTION_PRICE | PRICING_OPTION_YIELD)
    non_trading_day_fill_option: (NON_TRADING_WEEKDAYS | ALL_CALENDAR_DAYS | ACTIVE_DAYS_ONLY)
    non_trading_day_fill_method: (PREVIOUS_VALUE | NIL_VALUE)
    calendar_code_override: 2 letter county iso code
    """

    def __init__(self, sids, fields, start=None, end=None, period=None, ignore_security_error=0,
                 ignore_field_error=0, period_adjustment=None, currency=None, override_option=None,
                 pricing_option=None, non_trading_day_fill_option=None, non_trading_day_fill_method=None,
                 max_data_points=None, adjustment_normal=None, adjustment_abnormal=None, adjustment_split=None,
                 adjustment_follow_DPDF=None, calendar_code_override=None, **overrides):

        Request.__init__(self, '//blp/refdata', ignore_security_error=ignore_security_error,
                         ignore_field_error=ignore_field_error)
        period = period or 'DAILY'
        assert period in ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI-ANNUAL', 'YEARLY')
        self.is_single_sid = is_single_sid = isinstance(sids, basestring)
        self.is_single_field = is_single_field = isinstance(fields, basestring)
        self.sids = is_single_sid and [sids] or list(sids)
        self.fields = is_single_field and [fields] or list(fields)
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + pd.datetools.relativedelta(years=-1)
        self.period = period
        self.period_adjustment = period_adjustment
        self.currency = currency
        self.override_option = override_option
        self.pricing_option = pricing_option
        self.non_trading_day_fill_option = non_trading_day_fill_option
        self.non_trading_day_fill_method = non_trading_day_fill_method
        self.max_data_points = max_data_points
        self.adjustment_normal = adjustment_normal
        self.adjustment_abnormal = adjustment_abnormal
        self.adjustment_split = adjustment_split
        self.adjustment_follow_DPDF = adjustment_follow_DPDF
        self.calendar_code_override = calendar_code_override
        self.overrides = overrides

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__,
                       symbols=','.join(self.sids),
                       fields=','.join(self.fields),
                       start=self.start.strftime('%Y-%m-%d'),
                       end=self.end.strftime('%Y-%m-%d'),
                       period=self.period,
        )
        #TODO: add self.overrides if defined
        return '<{clz}([{symbols}], [{fields}], start={start}, end={end}, period={period}'.format(**fmtargs)

    def new_response(self):
        self.response = HistoricalDataResponse(self)

    def get_bbg_request(self, svc, session):
        # create the bloomberg request object
        request = svc.createRequest('HistoricalDataRequest')
        [request.append('securities', sec) for sec in self.sids]
        [request.append('fields', fld) for fld in self.fields]
        request.set('startDate', self.start.strftime('%Y%m%d'))
        request.set('endDate', self.end.strftime('%Y%m%d'))
        request.set('periodicitySelection', self.period)
        self.period_adjustment and request.set('periodicityAdjustment', self.period_adjustment)
        self.currency and request.set('currency', self.currency)
        self.override_option and request.set('overrideOption', self.override_option)
        self.pricing_option and request.set('pricingOption', self.pricing_option)
        self.non_trading_day_fill_option and request.set('nonTradingDayFillOption', self.non_trading_day_fill_option)
        self.non_trading_day_fill_method and request.set('nonTradingDayFillMethod', self.non_trading_day_fill_method)
        self.max_data_points and request.set('maxDataPoints', self.max_data_points)
        self.calendar_code_override and request.set('calendarCodeOverride', self.calendar_code_override)
        self.set_flag(request, self.adjustment_normal, 'adjustmentNormal')
        self.set_flag(request, self.adjustment_abnormal, 'adjustmentAbnormal')
        self.set_flag(request, self.adjustment_split, 'adjustmentSplit')
        self.set_flag(request, self.adjustment_follow_DPDF, 'adjustmentFollowDPDF')


        if hasattr(self, 'overrides') and self.overrides is not None:
            Request.apply_overrides(request, self.overrides)
        return request

    def on_security_data_node(self, node):
        """process a securityData node - FIXME: currently not handling relateDate node """
        sid = XmlHelper.get_child_value(node, 'security')
        farr = node.getElement('fieldData')
        dmap = defaultdict(list)
        for i in range(farr.numValues()):
            pt = farr.getValue(i)
            [dmap[f].append(XmlHelper.get_child_value(pt, f, allow_missing=1)) for f in ['date'] + self.fields]

        if not dmap:
            frame = pd.DataFrame(columns=self.fields)
        else:
            idx = dmap.pop('date')
            frame = pd.DataFrame(dmap, columns=self.fields, index=idx)
            frame.index.name = 'date'
        self.response.on_security_complete(sid, frame)

    def on_event(self, evt, is_final):
        for msg in XmlHelper.message_iter(evt):
            # Single security element in historical request
            node = msg.getElement('securityData')
            if node.hasElement('securityError'):
                sid = XmlHelper.get_child_value(node, 'security')
                self.security_errors.append(XmlHelper.as_security_error(node.getElement('securityError'), sid))
            else:
                self.on_security_data_node(node)


class ReferenceDataResponse(object):
    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """ :return: Multi-Index DataFrame """
        data = {sid: pd.Series(data) for sid, data in self.response_map.iteritems()}
        frame = pd.DataFrame.from_dict(data, orient='index')
        # layer in any missing fields just in case
        frame = frame.reindex_axis(self.request.fields, axis=1)
        return frame


class ReferenceDataRequest(Request):
    def __init__(self, sids, fields, ignore_security_error=0, ignore_field_error=0, return_formatted_value=None,
                 use_utc_time=None, **overrides):
        """
        response_type: (frame, map) how to return the results
        """
        Request.__init__(self, '//blp/refdata', ignore_security_error=ignore_security_error,
                         ignore_field_error=ignore_field_error)
        self.is_single_sid = is_single_sid = isinstance(sids, basestring)
        self.is_single_field = is_single_field = isinstance(fields, basestring)
        self.sids = isinstance(sids, basestring) and [sids] or sids
        self.fields = isinstance(fields, basestring) and [fields] or fields
        self.return_formatted_value = return_formatted_value
        self.use_utc_time = use_utc_time
        self.overrides = overrides

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__,
                       sids=','.join(self.sids),
                       fields=','.join(self.fields),
                       overrides=','.join(['%s=%s' % (k, v) for k, v in self.overrides.iteritems()]))
        return '<{clz}([{sids}], [{fields}], overrides={overrides})'.format(**fmtargs)

    def new_response(self):
        self.response = ReferenceDataResponse(self)

    def get_bbg_request(self, svc, session):
        # create the bloomberg request object
        request = svc.createRequest('ReferenceDataRequest')
        [request.append('securities', sec) for sec in self.sids]
        [request.append('fields', fld) for fld in self.fields]
        self.set_flag(request, self.return_formatted_value, 'returnFormattedValue')
        self.set_flag(request, self.use_utc_time, 'useUTCTime')
        Request.apply_overrides(request, self.overrides)
        return request

    def on_security_node(self, node):
        sid = XmlHelper.get_child_value(node, 'security')
        farr = node.getElement('fieldData')
        fdata = XmlHelper.get_child_values(farr, self.fields)
        assert len(fdata) == len(self.fields), 'field length must match data length'
        self.response.on_security_data(sid, dict(zip(self.fields, fdata)))
        ferrors = XmlHelper.get_field_errors(node)
        ferrors and self.field_errors.extend(ferrors)

    def on_event(self, evt, is_final):
        for msg in XmlHelper.message_iter(evt):
            for node, error in XmlHelper.security_iter(msg.getElement('securityData')):
                if error:
                    self.security_errors.append(error)
                else:
                    self.on_security_node(node)


class IntradayTickResponse(object):
    def __init__(self, request):
        self.request = request
        self.ticks = []  # array of dicts

    def as_frame(self):
        """Return a data frame with no set index"""
        return pd.DataFrame.from_records(self.ticks)


class IntradayTickRequest(Request):
    def __init__(self, sid, start=None, end=None, events=['TRADE'], include_condition_codes=None,
                 include_nonplottable_events=None, include_exchange_codes=None, return_eids=None,
                 include_broker_codes=None, include_rsp_codes=None, include_bic_mic_codes=None):
        """
        Parameters
        ----------
        events: array containing any of (TRADE, BID, ASK, BID_BEST, ASK_BEST, MID_PRICE, AT_TRADE, BEST_BID, BEST_ASK)
        """
        Request.__init__(self, '//blp/refdata')
        self.sid = sid
        self.events = isinstance(events, basestring) and [events] or events
        self.include_condition_codes = include_condition_codes
        self.include_nonplottable_events = include_nonplottable_events
        self.include_exchange_codes = include_exchange_codes
        self.return_eids = return_eids
        self.include_broker_codes = include_broker_codes
        self.include_rsp_codes = include_rsp_codes
        self.include_bic_mic_codes = include_bic_mic_codes
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + pd.datetools.relativedelta(days=-1)

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__,
                       sid=','.join(self.sid),
                       events=','.join(self.events))
        return '<{clz}({sid}, [{events}])'.format(**fmtargs)

    def new_response(self):
        self.response = IntradayTickResponse(self)

    def get_bbg_request(self, svc, session):
        # create the bloomberg request object
        request = svc.createRequest('IntradayTickRequest')
        request.set('security', self.sid)
        [request.append('eventTypes', evt) for evt in self.events]
        request.set('startDateTime', self.start)
        request.set('endDateTime', self.end)
        self.set_flag(request, self.include_condition_codes, 'includeConditionCodes')
        self.set_flag(request, self.include_nonplottable_events, 'includeNonPlottableEvents')
        self.set_flag(request, self.include_exchange_codes, 'includeExchangeCodes')
        self.set_flag(request, self.return_eids, 'returnEids')
        self.set_flag(request, self.include_broker_codes, 'includeBrokerCodes')
        self.set_flag(request, self.include_rsp_codes, 'includeRpsCodes')
        self.set_flag(request, self.include_bic_mic_codes, 'includeBicMicCodes')
        return request

    def on_tick_data(self, ticks):
        """Process the incoming tick data array"""
        for tick in XmlHelper.node_iter(ticks):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            tickmap = {n: XmlHelper.get_child_value(tick, n) for n in names}
            self.response.ticks.append(tickmap)

    def on_event(self, evt, is_final):
        for msg in XmlHelper.message_iter(evt):
            tdata = msg.getElement('tickData')
            # tickData will have 0 to 1 tickData[] elements
            if tdata.hasElement('tickData'):
                self.on_tick_data(tdata.getElement('tickData'))


class IntradayBarResponse(object):
    def __init__(self, request):
        self.request = request
        self.bars = []  # array of dicts

    def as_frame(self):
        return pd.DataFrame.from_records(self.bars)


class IntradayBarRequest(Request):
    def __init__(self, sid, start=None, end=None, event='TRADE', interval=None, gap_fill_initial_bar=None,
                 return_eids=None, adjustment_normal=None, adjustment_abnormal=None, adjustment_split=None,
                 adjustment_follow_DPDF=None):
        """
        Parameters
        ----------
        events: [TRADE, BID, ASK, BID_BEST, ASK_BEST, BEST_BID, BEST_ASK]
        interval: int, between 1 and 1440 in minutes. If omitted, defaults to 1 minute
        gap_fill_initial_bar: bool
                            If True, bar contains previous values if not ticks during the interval
        """
        Request.__init__(self, '//blp/refdata')
        self.sid = sid
        self.event = event
        self.interval = interval
        self.gap_fill_initial_bar = gap_fill_initial_bar
        self.return_eids = return_eids
        self.adjustment_normal = adjustment_normal
        self.adjustment_abnormal = adjustment_abnormal
        self.adjustment_split = adjustment_split
        self.adjustment_follow_DPDF = adjustment_follow_DPDF
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + pd.datetools.relativedelta(hours=-1)

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__,
                       sid=self.sid,
                       event=self.event,
                       start=self.start,
                       end=self.end)
        return '<{clz}({sid}, {event}, start={start}, end={end})'.format(**fmtargs)

    def new_response(self):
        self.response = IntradayBarResponse(self)

    def get_bbg_request(self, svc, session):
        # create the bloomberg request object
        request = svc.createRequest('IntradayBarRequest')
        request.set('security', self.sid)
        request.set('eventType', self.event)
        request.set('startDateTime', self.start)
        request.set('endDateTime', self.end)
        request.set('interval', self.interval or 1)
        self.set_flag(request, self.gap_fill_initial_bar, 'gapFillInitialBar')
        self.set_flag(request, self.return_eids, 'returnEids')
        self.set_flag(request, self.adjustment_normal, 'adjustmentNormal')
        self.set_flag(request, self.adjustment_abnormal, 'adjustmentAbnormal')
        self.set_flag(request, self.adjustment_split, 'adjustmentSplit')
        self.set_flag(request, self.adjustment_follow_DPDF, 'adjustmentFollowDPDF')
        return request

    def on_bar_data(self, bars):
        """Process the incoming tick data array"""
        for tick in XmlHelper.node_iter(bars):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            barmap = {n: XmlHelper.get_child_value(tick, n) for n in names}
            self.response.bars.append(barmap)

    def on_event(self, evt, is_final):
        for msg in XmlHelper.message_iter(evt):
            data = msg.getElement('barData')
            # tickData will have 0 to 1 tickData[] elements
            if data.hasElement('barTickData'):
                self.on_bar_data(data.getElement('barTickData'))


class EQSResponse(object):
    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """ :return: Multi-Index DataFrame """
        data = {sid: pd.Series(data) for sid, data in self.response_map.iteritems()}
        return pd.DataFrame.from_dict(data, orient='index')


class EQSRequest(Request):
    def __init__(self, name, type='GLOBAL', group='General', asof=None, language=None):
        super(EQSRequest, self).__init__('//blp/refdata')
        self.name = name
        self.group = group
        self.type = type
        self.asof = asof and pd.to_datetime(asof) or None
        self.language = language

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__,
                       name=self.name,
                       type=self.type,
                       group=self.group,
                       asof=self.asof)
        return '<{clz}({name}, type={type}, group={group}, asof={asof})'.format(**fmtargs)

    def new_response(self):
        self.response = EQSResponse(self)

    def get_bbg_request(self, svc, session):
        # create the bloomberg request object
        request = svc.createRequest('BeqsRequest')
        request.set('screenName', self.name)
        self.type and request.set('screenType', self.type)
        self.group and request.set('Group', self.group)
        overrides = {}
        if self.asof:
            overrides['PiTDate'] = self.asof.strftime('%Y%m%d')
        if self.language:
            overrides['languageId'] = self.language
        overrides and self.apply_overrides(request, overrides)
        return request

    def on_security_node(self, node):
        sid = XmlHelper.get_child_value(node, 'security')
        farr = node.getElement('fieldData')
        fldnames = [str(farr.getElement(_).name()) for _ in range(farr.numElements())]
        fdata = XmlHelper.get_child_values(farr, fldnames)
        self.response.on_security_data(sid, dict(zip(fldnames, fdata)))
        ferrors = XmlHelper.get_field_errors(node)
        ferrors and self.field_errors.extend(ferrors)

    def on_event(self, evt, is_final):
        for msg in XmlHelper.message_iter(evt):
            data = msg.getElement('data')
            for node, error in XmlHelper.security_iter(data.getElement('securityData')):
                if error:
                    self.security_errors.append(error)
                else:
                    self.on_security_node(node)


class Terminal(object):
    """Submits requests to the Bloomberg Terminal and dispatches the events back to the request
    object for processing.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.logger = log.instance_logger(repr(self), self)

    def __repr__(self):
        fmtargs = dict(clz=self.__class__.__name__, host=self.host, port=self.port)
        return '<{clz}({host}:{port})'.format(**fmtargs)

    def _create_session(self):
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        return blpapi.Session(opts)

    def execute(self, request):
        session = self._create_session()
        if not session.start():
            raise Exception('failed to start session')

        try:
            self.logger.info('executing request: %s' % repr(request))
            if not session.openService(request.svcname):
                raise Exception('failed to open service %s' % request.svcname)

            svc = session.getService(request.svcname)
            asbbg = request.get_bbg_request(svc, session)
            # setup response capture
            request.new_response()
            session.sendRequest(asbbg)
            while True:
                evt = session.nextEvent(500)
                if evt.eventType() == blpapi.Event.RESPONSE:
                    request.on_event(evt, is_final=True)
                    break
                elif evt.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    request.on_event(evt, is_final=False)
                else:
                    request.on_admin_event(evt)
            request.has_exception and request.raise_exception()
            return request.response
        finally:
            session.stop()

    def get_historical(self, sids, flds, start=None, end=None, period=None, ignore_security_error=0,
                       ignore_field_error=0, **overrides):
        req = HistoricalDataRequest(sids, flds, start=start, end=end, period=period,
                                    ignore_security_error=ignore_security_error,
                                    ignore_field_error=ignore_field_error,
                                    **overrides)
        return self.execute(req)

    def get_reference_data(self, sids, flds, ignore_security_error=0, ignore_field_error=0, **overrides):
        req = ReferenceDataRequest(sids, flds, ignore_security_error=ignore_security_error,
                                   ignore_field_error=ignore_field_error, **overrides)
        return self.execute(req)

    def get_intraday_tick(self, sids, events=['TRADE'], start=None, end=None, include_condition_codes=None,
                          include_nonplottable_events=None, include_exchange_codes=None, return_eids=None,
                          include_broker_codes=None, include_rsp_codes=None, include_bic_mic_codes=None,
                          **overrides):
        req = IntradayTickRequest(sids, start=start, end=end, events=events,
                                  include_condition_codes=include_condition_codes,
                                  include_nonplottable_events=include_nonplottable_events,
                                  include_exchange_codes=include_exchange_codes,
                                  return_eids=return_eids, include_broker_codes=include_broker_codes,
                                  include_rsp_codes=include_rsp_codes,
                                  include_bic_mic_codes=include_bic_mic_codes, **overrides)
        return self.execute(req)

    def get_intraday_bar(self, sid, event='TRADE', start=None, end=None, interval=None, gap_fill_initial_bar=None,
                         return_eids=None, adjustment_normal=None, adjustment_abnormal=None, adjustment_split=None,
                         adjustment_follow_DPDF=None):
        req = IntradayBarRequest(sid, start=start, end=end, event=event, interval=interval,
                                 gap_fill_initial_bar=gap_fill_initial_bar,
                                 return_eids=return_eids, adjustment_normal=adjustment_normal,
                                 adjustment_split=adjustment_split,
                                 adjustment_abnormal=adjustment_abnormal, adjustment_follow_DPDF=adjustment_follow_DPDF)
        return self.execute(req)

    def get_screener(self, name, group='General', type='GLOBAL', asof=None, language=None):
        req = EQSRequest(name, type=type, group=group, asof=asof, language=language)
        return self.execute(req)


class SyncSubscription(object):
    def __init__(self, tickers, fields, interval=None, host='localhost', port=8194):
        self.fields = isinstance(fields, basestring) and [fields] or fields
        self.tickers = isinstance(tickers, basestring) and [tickers] or tickers
        self.interval = interval
        self.host = host
        self.port = port
        self.session = None
        # build an empty frame
        nrows, ncols = len(self.tickers), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields, index=self.tickers)

    def _init(self):
        # init session
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        self.session = session = blpapi.Session(opts)
        if not session.start():
            raise Exception('failed to start session')

        if not session.openService('//blp/mktdata'):
            raise Exception('failed to open service')

        # init subscriptions
        subs = blpapi.SubscriptionList()
        flds = ','.join(self.fields)
        istr = self.interval and 'interval=%.1f' % self.interval or ''
        for ticker in self.tickers:
            subs.add(ticker, flds, istr, blpapi.CorrelationId(ticker))
        session.subscribe(subs)

    def on_subscription_status(self, evt):
        for msg in XmlHelper.message_iter(evt):
            if msg.messageType() == 'SubscriptionFailure':
                sid = msg.correlationIds()[0].value()
                desc = msg.getElement('reason').getElementAsString('description')
                raise Exception('subscription failed sid=%s desc=%s' % (sid, desc))

    def on_subscription_data(self, evt):
        for msg in XmlHelper.message_iter(evt):
            sid = msg.correlationIds()[0].value()
            ridx = self.tickers.index(sid)
            for cidx, fld in enumerate(self.fields):
                if msg.hasElement(fld.upper()):
                    val = XmlHelper.get_child_value(msg, fld.upper())
                    self.frame.iloc[ridx, cidx] = val

    def check_for_updates(self, timeout=500):
        if self.session is None:
            self._init()

        evt = self.session.nextEvent(timeout)
        if evt.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
            logger.info('next(): subscription data')
            self.on_subscription_data(evt)
        elif evt.eventType() == blpapi.Event.SUBSCRIPTION_STATUS:
            logger.info('next(): subscription status')
            self.on_subscription_status(evt)
            self.check_for_updates(timeout)
        elif evt.eventType() == blpapi.Event.TIMEOUT:
            pass
        else:
            logger.info('next(): ignoring event %s' % evt.eventType())
            self.check_for_updates(timeout)


