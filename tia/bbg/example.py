import pandas as pd
from tia.bbg import LocalTerminal

if __name__ == '__main__':
    d = pd.datetools.BDay(-4).apply(pd.datetime.now())
    m = pd.datetools.BMonthBegin(-2).apply(pd.datetime.now())

    def banner(msg):
        print('*' * 25)
        print(msg)
        print('*' * 25)

    banner('ReferenceDataRequest: single security, single field, frame response')
    response = LocalTerminal.get_reference_data('msft us equity', 'px_last')
    print(response.as_map())
    print(response.as_frame())

    banner('ReferenceDataRequest: single security, multi-field (with bulk), frame response')
    response = LocalTerminal.get_reference_data('eurusd curncy', ['px_last', 'fwd_curve'])
    print(response.as_map())
    rframe = response.as_frame()
    print(rframe.columns)
    # show frame within a frame
    print(rframe.ix[0, 'fwd_curve'].tail())

    banner('ReferenceDataRequest: multi security, multi-field, bad field')
    response = LocalTerminal.get_reference_data(['eurusd curncy', 'msft us equity'], ['px_last', 'fwd_curve'],
                                                ignore_field_error=1)
    print(response.as_frame()['fwd_curve']['eurusd curncy'])

    banner('HistoricalDataRequest: multi security, multi-field, daily data')
    response = LocalTerminal.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=d)
    print(response.as_map())
    print(response.as_frame().head(5))

    banner('HistoricalDataRequest: multi security, multi-field, weekly data')
    response = LocalTerminal.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=m,
                                                 period='WEEKLY')
    print('--------- AS SINGLE TABLE ----------')
    print(response.as_frame().head(5))

    #
    # HOW TO
    #
    # - Retrieve an fx vol surface:  BbgReferenceDataRequest('eurusd curncy', 'DFLT_VOL_SURF_MID')
    # - Retrieve a fx forward curve:  BbgReferenceDataRequest('eurusd curncy', 'FWD_CURVE')
    # - Retrieve dividends:  BbgReferenceDataRequest('csco us equity', 'BDVD_PR_EX_DTS_DVD_AMTS_W_ANN')