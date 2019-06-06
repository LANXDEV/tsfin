import QuantLib as ql
import numpy as np
from collections import OrderedDict
from tsfin.base import to_ql_date, to_datetime, to_ql_quote_handle


class BlackScholesMerton:

    def __init__(self, ts_option_collection, ts_underlying_collection, yield_curve):

        self.ts_options = ts_option_collection
        self.ts_underlying = ts_underlying_collection
        self.yield_curve = yield_curve
        self.risk_free_handle = ql.RelinkableYieldTermStructureHandle()
        self.volatility_handle = ql.RelinkableBlackVolTermStructureHandle()
        self.dividend_handle = ql.RelinkableYieldTermStructureHandle()
        self.spot_price_handle = ql.RelinkableQuoteHandle()
        self.bsm_process = ql.BlackScholesMertonProcess(self.spot_price_handle,
                                                        self.dividend_handle,
                                                        self.risk_free_handle,
                                                        self.volatility_handle)
        self.vol_updated = OrderedDict()

    def spot_price_update(self, date, underlying_name, spot_price=None, last_available=True):

        dt_date = to_datetime(date)
        ts_underlying = self.ts_underlying.get(underlying_name).price
        if spot_price is None:
            spot_price = ts_underlying.get_values(index=dt_date, last_available=last_available)
        else:
            spot_price = spot_price

        self.spot_price_handle.linkTo(ql.SimpleQuote(spot_price))

    def dividend_yield_update(self, date, calendar, day_counter, underlying_name, dividend_yield=None, dvd_tax_adjust=1,
                              last_available=True, compounding=ql.Continuous):

        dt_date = to_datetime(date)
        dvd_ts = self.ts_underlying.get(underlying_name).eqy_dvd_yld_12m
        if dividend_yield is None:
            dividend_yield = dvd_ts.get_values(index=dt_date, last_available=last_available)
        else:
            dividend_yield = dividend_yield

        dividend_yield = dividend_yield * dvd_tax_adjust
        dividend = ql.FlatForward(0, calendar, to_ql_quote_handle(dividend_yield), day_counter, compounding)
        self.dividend_handle.linkTo(dividend)

    def yield_curve_update(self, date, calendar, day_counter, maturity, risk_free=None, compounding=ql.Continuous,
                           frequency=ql.Once):

        ql_date = to_ql_date(date)
        mat_date = to_ql_date(maturity)
        if risk_free is not None:
            zero_rate = risk_free
        else:
            zero_rate = self.yield_curve.zero_rate_to_date(date=ql_date, to_date=mat_date, compounding=compounding,
                                                           frequency=frequency)
        yield_curve = ql.FlatForward(0, calendar, to_ql_quote_handle(zero_rate), day_counter, compounding)
        self.risk_free_handle.linkTo(yield_curve)

    def volatility_update(self, date, calendar, day_counter, ts_option_name, underlying_name, vol_value=None,
                          last_available=False):

        dt_date = to_datetime(date)
        option_vol_ts = self.ts_options.get(ts_option_name)
        vol_updated = True

        if vol_value is not None:
            volatility_value = vol_value
        else:
            volatility_value = option_vol_ts.ivol_mid.get_values(index=dt_date, last_available=last_available)
            if np.isnan(volatility_value):
                volatility_value = 0
                vol_updated = False

        back_constant_vol = ql.BlackConstantVol(0, calendar, to_ql_quote_handle(volatility_value), day_counter)
        self.volatility_handle.linkTo(back_constant_vol)
        self.vol_updated[underlying_name][to_ql_date(date)] = vol_updated

    def update_process(self, date, calendar, day_counter, ts_option_name, maturity, underlying_name,
                       vol_last_available=False, dvd_tax_adjust=1, last_available=True, **kwargs):

        self.vol_updated[underlying_name] = OrderedDict()
        if to_ql_date(date) in self.vol_updated[underlying_name].keys():
            vol_updated = self.vol_updated[underlying_name][to_ql_date(date)]
        else:
            vol_updated = False

        if vol_updated:
            return self.vol_updated[underlying_name][to_ql_date(date)]
        else:
            self.spot_price_update(date=date, underlying_name=underlying_name, last_available=last_available)

            self.dividend_yield_update(date=date, calendar=calendar, day_counter=day_counter,
                                       underlying_name=underlying_name, dvd_tax_adjust=dvd_tax_adjust,
                                       last_available=last_available)

            self.yield_curve_update(date=date, calendar=calendar, day_counter=day_counter, maturity=maturity, **kwargs)

            self.volatility_update(date=date, calendar=calendar, day_counter=day_counter, ts_option_name=ts_option_name,
                                   underlying_name=underlying_name, last_available=vol_last_available)

            return self.vol_updated[underlying_name][to_ql_date(date)]
