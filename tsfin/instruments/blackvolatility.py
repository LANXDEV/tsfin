import QuantLib as ql
import numpy as np
from collections import OrderedDict
from tsfin.base import to_ql_date, to_ql_calendar, to_ql_day_counter, to_datetime, conditional_vectorize, \
    to_ql_quote_handle, to_list
from tsio import TimeSeriesCollection
from tsfin.constants import CALENDAR, MATURITY_DATE, DAY_COUNTER


def adjusted_date_range(initial_date, final_date, calendar):

    schedule = ql.Schedule(to_ql_date(initial_date) - ql.Period("1D"), to_ql_date(final_date),
                           ql.Period("1D"), calendar, ql.Following, ql.Preceding,
                           ql.DateGeneration.Backward, False)
    dates = list()
    for date in schedule:
        if to_datetime(date) < to_datetime(initial_date) or to_datetime(date) in dates:
            continue
        dates.append(to_datetime(date))

    return dates


class BlackScholesMerton:

    def __init__(self, ts_options, ts_underlying, initial_date, final_date,
                 yield_curve, dvd_zero=False, dvd_tax_adjust=1):

        self.ts_options = TimeSeriesCollection(to_list(ts_options))
        self.ts_underlying = ts_underlying
        self.initial_date = initial_date
        self.final_date = final_date
        self.calendar = to_ql_calendar(ts_options[0].ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(ts_options[0].ts_attributes[DAY_COUNTER])
        self.dates = adjusted_date_range(self.initial_date, self.final_date, self.calendar)
        self.dvd_zero = dvd_zero
        self.dvd_tax_adjust = dvd_tax_adjust
        self.yield_curve = yield_curve
        self.risk_free = None
        self.volatility = None
        self.dividend = None
        self.spot_price = None
        self.process = None

    def volatility_from_ts_values(self, ts_vol, update_date=None):

        vol_ts = ts_vol
        volatility_handle = OrderedDict()
        last_available = False
        dates = self.dates
        if update_date is not None:
            dates = to_list(to_datetime(update_date))
            last_available = True

        for date in dates:
            ts_date = to_datetime(date)
            ts_value = vol_ts.get_values(index=ts_date, last_available=last_available)
            if np.isnan(ts_value):
                continue
            ql_sigma = to_ql_quote_handle(ts_value / 100)
            black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql_sigma, self.day_counter)
            volatility_handle[ts_date] = ql.BlackVolTermStructureHandle(black_constant_vol)

        return volatility_handle

    def dividend_yield_from_ts_values(self, update_date=None):

        dvd_series = self.ts_underlying.eqy_dvd_yld_12m
        dvd_series.ts_values *= self.dvd_tax_adjust
        dates = self.dates
        if update_date is not None:
            dates = to_list(to_datetime(update_date))

        dividend_handle = OrderedDict()
        if self.dvd_zero:
            for date in dates:
                ts_date = to_datetime(date)
                dividend_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(
                    ql.FlatForward(0, self.calendar, to_ql_quote_handle(0), self.day_counter, ql.Continuous))
        else:
            for date in dates:
                ts_date = to_datetime(date)
                ts_value = dvd_series.get_values(index=ts_date, last_available=True)
                dividend_handle[ts_date] = ql.YieldTermStructureHandle(
                    ql.FlatForward(0, self.calendar,  to_ql_quote_handle(ts_value / 100), self.day_counter,
                                   ql.Continuous))

        return dividend_handle

    def yield_curve_flat(self, maturity, update_date=None):

        yield_curve = self.yield_curve
        dates = self.dates
        if update_date is not None:
            dates = to_list(to_datetime(update_date))

        maturity = to_ql_date(to_datetime(maturity))
        yield_handle = OrderedDict()
        for date in dates:
            ts_date = to_datetime(date)
            if to_datetime(ts_date) > to_datetime(maturity):
                zero_rate = 0
            else:
                zero_rate = yield_curve.zero_rate_to_date(date=ts_date, to_date=maturity, compounding=ql.Continuous,
                                                          frequency=ql.Annual)
            yield_handle[ts_date] = ql.YieldTermStructureHandle(ql.FlatForward(
                0, self.calendar, to_ql_quote_handle(zero_rate), self.day_counter, ql.Continuous))

        return yield_handle

    def underlying_quote_handler(self, update_date=None):

        underlying_ts = self.ts_underlying.price
        spot_handle = OrderedDict()
        dates = self.dates
        if update_date is not None:
            dates = to_list(update_date)

        for date in dates:
            ts_date = to_datetime(date)
            ts_value = underlying_ts.get_values(index=ts_date, last_available=True)
            spot_handle[ts_date] = to_ql_quote_handle(ts_value)

        return spot_handle

    def update_process(self, start_date=None, final_date=None):

        if start_date is not None:
            self.initial_date = to_datetime(start_date)
        if final_date is not None:
            self.final_date = to_datetime(final_date)

        self.volatility = OrderedDict()
        self.risk_free = OrderedDict()
        for ts in self.ts_options:
            self.volatility[ts.ts_name] = self.volatility_from_ts_values(ts_vol=ts.ivol_mid)
            self.risk_free[ts.ts_name] = self.yield_curve_flat(maturity=ts.ts_attributes[MATURITY_DATE])

        self.dividend = self.dividend_yield_from_ts_values()

        self.spot_price = self.underlying_quote_handler()

        process = OrderedDict()

        for ts in self.ts_options:
            process[ts.ts_name] = OrderedDict()
            for date_value in self.dates:
                date_value = to_datetime(date_value)
                try:
                    process[ts.ts_name][date_value] = ql.BlackScholesMertonProcess(
                        self.spot_price[date_value],
                        self.dividend[date_value],
                        self.risk_free[ts.ts_name][date_value],
                        self.volatility[ts.ts_name][date_value])
                except KeyError:
                    continue
        self.process = process
        return self

    @conditional_vectorize('date, vol_value')
    def update_missing_vol(self, date, vol_value, ts_name, maturity):

        dt_date = to_datetime(date)
        if self.process is None:
            self.update_process()

        if date > self.final_date:
            self.final_date = date

        vol_value = vol_value / 100
        ql_sigma = to_ql_quote_handle(float(vol_value))
        black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql_sigma, self.day_counter)
        self.volatility[ts_name][dt_date] = ql.BlackVolTermStructureHandle(black_constant_vol)

        try:
            dividend = self.dividend[dt_date]
        except KeyError:
            self.dividend[dt_date] = self.dividend_yield_from_ts_values(update_date=dt_date)[dt_date]
            dividend = self.dividend[dt_date]

        try:
            risk_free = self.risk_free[ts_name][dt_date]
        except KeyError:
            self.risk_free[ts_name][dt_date] = self.yield_curve_flat(maturity=maturity, update_date=dt_date)[dt_date]
            risk_free = self.risk_free[ts_name][dt_date]

        try:
            spot_price = self.spot_price[dt_date]
        except KeyError:
            self.spot_price[dt_date] = self.underlying_quote_handler(update_date=dt_date)[dt_date]
            spot_price = self.spot_price[dt_date]

        self.process[ts_name][dt_date] = ql.BlackScholesMertonProcess(
            spot_price,
            dividend,
            risk_free,
            self.volatility[ts_name][dt_date])

        return self

    def update_only_yield_curve(self, ts_name, maturity, update_date=None):

        if self.process is None:
            self.update_process()
        if update_date is not None:
            update_date = to_datetime(update_date)
        dates = self.dates

        risk_free = self.yield_curve_flat(maturity=maturity, update_date=update_date)[update_date]

        process = OrderedDict()
        process[ts_name] = OrderedDict()
        for date in dates:
            process[ts_name][date] = ql.BlackScholesMertonProcess(
                self.spot_price[date],
                self.dividend[date],
                risk_free,
                self.volatility[ts_name][date])

        return process

    @conditional_vectorize('date, spot_price')
    def update_only_spot_price(self, date, spot_price, ts_name):

        dt_date = to_datetime(date)
        if self.process is None:
            self.update_process()

        ql_spot_price = to_ql_quote_handle(spot_price)

        process = ql.BlackScholesMertonProcess(ql_spot_price,
                                               self.dividend[dt_date],
                                               self.risk_free[ts_name][dt_date],
                                               self.volatility[ts_name][dt_date])
        return process

    @conditional_vectorize('date, vol_value')
    def update_only_vol(self, date, vol_value, ts_name):

        dt_date = to_datetime(date)
        if self.process is None:
            self.update_process()

        vol_value = vol_value / 100
        ql_sigma = to_ql_quote_handle(float(vol_value))
        black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql_sigma, self.day_counter)
        volatility_term = ql.BlackVolTermStructureHandle(black_constant_vol)

        process = ql.BlackScholesMertonProcess(
            self.spot_price[dt_date],
            self.dividend[dt_date],
            self.risk_free[ts_name][dt_date],
            volatility_term)

        return process
