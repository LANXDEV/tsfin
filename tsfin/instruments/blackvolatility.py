import QuantLib as ql
from collections import OrderedDict
from tsfin.base import to_ql_date, to_ql_calendar, to_ql_day_counter, filter_series, to_datetime, \
    conditional_vectorize, to_ql_quote_handle, to_upper_list, to_list
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.tools import ts_values_to_dict
from ads.tools.fixedincome import generate_yield_curve
from tsfin.constants import CALENDAR, DAY_COUNTER


def filtered_series(timeseries, initial_date, final_date):
    filter_series(timeseries.ts_values, initial_date=initial_date, final_date=final_date)
    return timeseries


class BlackScholesMerton:

    def __init__(self, ts_option_vol, ts_underlying, ts_underlying_dvd, initial_date, final_date,
                 curve_tag, dvd_zero=False, dvd_tax_adjust=1):

        self.ts_option_vol = TimeSeriesCollection(to_list(ts_option_vol))
        self.ts_underlying_dvd = ts_underlying_dvd
        self.ts_underlying_price = ts_underlying
        self.initial_date = initial_date
        self.final_date = final_date
        self.calendar = to_ql_calendar(ts_option_vol[0].ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(ts_option_vol[0].ts_attributes[DAY_COUNTER])
        self.curve_tag = curve_tag
        self.dvd_zero = dvd_zero
        self.dvd_tax_adjust = dvd_tax_adjust
        self.yield_curve = None
        self.volatility = None
        self.dividend = None
        self.spot_price = None
        self.process = None

    def volatility_from_ts_values(self, ts_vol, initial_date, final_date):

        vol_ts = filtered_series(ts_vol, initial_date=initial_date, final_date=final_date)
        volatility_handle = OrderedDict()

        for ts_date, ts_value in vol_ts.ts_values.iteritems():
            ql_sigma = to_ql_quote_handle(ts_value / 100)
            black_constant_vol = ql.BlackConstantVol(to_ql_date(ts_date), self.calendar, ql_sigma, self.day_counter)
            volatility_handle[to_datetime(ts_date)] = ql.BlackVolTermStructureHandle(black_constant_vol)

        return volatility_handle

    def dividend_yield_from_ts_values(self, initial_date, final_date):

        dvd_series = filtered_series(self.ts_underlying_dvd,
                                     initial_date=initial_date,
                                     final_date=final_date)
        dvd_series.ts_values *= self.dvd_tax_adjust

        dividend_handle = OrderedDict()
        if self.dvd_zero:
            for ts_date, ts_value in dvd_series.ts_values.iteritems():
                dividend_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(
                    ql.FlatForward(to_ql_date(ts_date), to_ql_quote_handle(0), self.day_counter))
        else:
            for ts_date, ts_value in dvd_series.ts_values.iteritems():
                dividend_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(
                    ql.FlatForward(to_ql_date(ts_date),  to_ql_quote_handle(ts_value / 100),
                                   self.day_counter))

        return dividend_handle

    def yield_curve_values(self, initial_date, final_date, curve_tag):

        yield_curve = generate_yield_curve(curve_tag=curve_tag,
                                           initial_date=initial_date,
                                           final_date=final_date)

        return yield_curve

    def underlying_quote_handler(self, initial_date, final_date):

        underlying_ts = filtered_series(self.ts_underlying_price,
                                        initial_date=initial_date,
                                        final_date=final_date)

        spot_handle = OrderedDict()
        for ts_date, ts_value in underlying_ts.ts_values.iteritems():
            spot_handle[to_datetime(ts_date)] = to_ql_quote_handle(ts_value)

        return spot_handle

    def update_process(self, start_date=None, final_date=None):

        if start_date is not None:
            self.initial_date = to_datetime(start_date)
        if final_date is not None:
            self.final_date = to_datetime(final_date)

        vol_collection = OrderedDict()
        for ts in self.ts_option_vol:
            vol_collection[ts.ts_name] = self.volatility_from_ts_values(ts_vol=ts,
                                                                        initial_date=self.initial_date,
                                                                        final_date=self.final_date)

        self.volatility = vol_collection

        self.dividend = self.dividend_yield_from_ts_values(initial_date=self.initial_date,
                                                           final_date=self.final_date)

        self.yield_curve = self.yield_curve_values(curve_tag=self.curve_tag,
                                                   initial_date=self.initial_date,
                                                   final_date=self.final_date)

        self.spot_price = self.underlying_quote_handler(initial_date=self.initial_date,
                                                        final_date=self.final_date)

        process = OrderedDict()
        for ts in self.ts_option_vol:
            process[ts.ts_name] = OrderedDict()
            for date_value in ts.ts_values.index:
                process[ts.ts_name][date_value] = ql.BlackScholesMertonProcess(
                    self.spot_price[date_value],
                    self.dividend[date_value],
                    self.yield_curve.yield_curve_handle(date=date_value),
                    self.volatility[ts.ts_name][date_value])

        self.process = process
        return self

    def update_only_yield_curve(self, curve_tag, overwrite=False):

        if self.process is None:
            self.update_process()

        yield_curve = self.yield_curve_values(curve_tag=curve_tag,
                                              initial_date=self.initial_date,
                                              final_date=self.final_date)

        process = OrderedDict()
        for ts in self.ts_option_vol:
            process[ts.ts_name] = OrderedDict()
            for date_value in ts.ts_values.index:
                process[ts.ts_name][date_value] = ql.BlackScholesMertonProcess(
                    self.spot_price[date_value],
                    self.dividend[date_value],
                    yield_curve.yield_curve_handle(date=date_value),
                    self.volatility[ts.ts_name][date_value])

        if overwrite:
            self.yield_curve = yield_curve
            self.process = process
        return process

    @conditional_vectorize('date, spot_price')
    def update_only_spot_price(self, date, spot_price, overwrite=False):

        if not self.initial_date <= date <= self.final_date:
            raise ValueError('Date not is not on interval')

        if self.process is None:
            self.update_process()

        spot_price_dict = OrderedDict()
        spot_price_dict[date] = to_ql_quote_handle(spot_price)

        process = OrderedDict()
        process[date] = ql.BlackScholesMertonProcess(self.spot_price[date],
                                                     self.dividend[date],
                                                     self.yield_curve.yield_curve_handle(date=date),
                                                     self.volatility[date])

        if overwrite:
            self.spot_price[date] = spot_price_dict[date]
            self.process[date] = process[date]
        return process

    @conditional_vectorize('date, vol_value')
    def update_missing_vol(self, date, vol_value, ts_name):

        if self.process is None:
            self.update_process()

        if date > self.final_date:
            self.final_date = date

        vol_value = vol_value / 100
        ql_sigma = to_ql_quote_handle(float(vol_value))
        black_constant_vol = ql.BlackConstantVol(to_ql_date(date), self.calendar, ql_sigma, self.day_counter)
        self.volatility[ts_name][to_datetime(date)] = ql.BlackVolTermStructureHandle(black_constant_vol)

        try:
            dividend = self.dividend[to_datetime(date)]
        except KeyError:
            self.dividend[to_datetime(date)] = self.dividend_yield_from_ts_values(initial_date=date,
                                                                                  final_date=self.final_date)
            dividend = self.dividend[to_datetime(date)]

        yield_curve = self.yield_curve_values(curve_tag=self.curve_tag,
                                              initial_date=date,
                                              final_date=self.final_date)
        try:
            spot_price = self.spot_price[to_datetime(date)]
        except KeyError:
            self.spot_price[to_datetime(date)] = self.underlying_quote_handler(initial_date=date,
                                                                               final_date=self.final_date)
            spot_price = self.spot_price[to_datetime(date)]

        self.process[ts_name][to_datetime(date)] = ql.BlackScholesMertonProcess(
            spot_price,
            dividend,
            yield_curve.yield_curve_handle(date=date), self.volatility[ts_name][to_datetime(date)])

        return self

    @conditional_vectorize('date, vol_value')
    def update_only_vol(self, date, vol_value, ts_name):

        if self.process is None:
            self.update_process()

        vol_value = vol_value / 100
        ql_sigma = to_ql_quote_handle(float(vol_value))
        black_constant_vol = ql.BlackConstantVol(to_ql_date(date), self.calendar, ql_sigma, self.day_counter)
        self.volatility[ts_name][to_datetime(date)] = ql.BlackVolTermStructureHandle(black_constant_vol)

        self.process[ts_name][to_datetime(date)] = ql.BlackScholesMertonProcess(
            self.spot_price[to_datetime(date)],
            self.dividend[to_datetime(date)],
            self.yield_curve.yield_curve_handle(date=date), self.volatility[ts_name][to_datetime(date)])

        return self
