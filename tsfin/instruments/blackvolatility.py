import QuantLib as ql
from collections import OrderedDict
from tsfin.base import to_ql_date, to_ql_calendar, to_ql_day_counter, filter_series, to_datetime, \
    conditional_vectorize, to_ql_quote_handle, to_upper_list, to_list
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.tools import ts_values_to_dict
from ads.tools.fixedincome import generate_yield_curve
from tsfin.constants import CALENDAR, MATURITY_DATE, DAY_COUNTER


def filtered_series(timeseries, initial_date, final_date):
    filter_series(timeseries.ts_values, initial_date=initial_date, final_date=final_date)
    return timeseries


class BlackScholesMerton:

    def __init__(self, ts_options, ts_underlying, initial_date, final_date,
                 curve_tag, dvd_zero=False, dvd_tax_adjust=1):

        self.ts_options = TimeSeriesCollection(to_list(ts_options))
        self.ts_underlying = ts_underlying
        self.initial_date = initial_date
        self.final_date = final_date
        self.calendar = to_ql_calendar(ts_options[0].ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(ts_options[0].ts_attributes[DAY_COUNTER])
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
            black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql_sigma, self.day_counter)
            volatility_handle[to_datetime(ts_date)] = ql.BlackVolTermStructureHandle(black_constant_vol)

        return volatility_handle

    def dividend_yield_from_ts_values(self, initial_date, final_date):

        dvd_series = filtered_series(self.ts_underlying.eqy_dvd_yld_12m,
                                     initial_date=initial_date,
                                     final_date=final_date)
        dvd_series.ts_values *= self.dvd_tax_adjust

        dividend_handle = OrderedDict()
        if self.dvd_zero:
            for ts_date, ts_value in dvd_series.ts_values.iteritems():
                dividend_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(
                    ql.FlatForward(0, self.calendar, to_ql_quote_handle(0), self.day_counter))
        else:
            for ts_date, ts_value in dvd_series.ts_values.iteritems():
                dividend_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(
                    ql.FlatForward(0, self.calendar,  to_ql_quote_handle(ts_value / 100),
                                   self.day_counter))

        return dividend_handle

    def yield_curve_values(self, initial_date, final_date, curve_tag):

        yield_curve = generate_yield_curve(curve_tag=curve_tag,
                                           initial_date=initial_date,
                                           final_date=final_date)

        return yield_curve

    def yield_curve_flat(self, maturity, initial_date, final_date, curve_tag):

        yield_curve = generate_yield_curve(curve_tag=curve_tag,
                                           initial_date=initial_date,
                                           final_date=final_date)
        underlying_ts = filtered_series(self.ts_underlying.price,
                                        initial_date=initial_date,
                                        final_date=final_date)
        maturity = to_ql_date(to_datetime(maturity))
        yield_handle = OrderedDict()
        for ts_date, ts_value in underlying_ts.ts_values.iteritems():
            zero_rate = yield_curve.zero_rate_to_date(date=ts_date, to_date=maturity, compounding=ql.Continuous,
                                                      frequency=ql.Annual)
            yield_handle[to_datetime(ts_date)] = ql.YieldTermStructureHandle(ql.FlatForward(
                0, self.calendar, to_ql_quote_handle(zero_rate), self.day_counter))

        return yield_handle

    def underlying_quote_handler(self, initial_date, final_date):

        underlying_ts = filtered_series(self.ts_underlying.price,
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
        yield_collection = OrderedDict()
        for ts in self.ts_options:
            vol_collection[ts.ts_name] = self.volatility_from_ts_values(ts_vol=ts.ivol_mid,
                                                                        initial_date=self.initial_date,
                                                                        final_date=self.final_date)
            yield_collection[ts.ts_name] = self.yield_curve_flat(maturity=(ts.ts_attributes[MATURITY_DATE]),
                                                                 initial_date=self.initial_date,
                                                                 final_date=self.final_date,
                                                                 curve_tag=self.curve_tag)

        self.volatility = vol_collection
        self.yield_curve = yield_collection

        self.dividend = self.dividend_yield_from_ts_values(initial_date=self.initial_date,
                                                           final_date=self.final_date)

        # self.yield_curve = self.yield_curve_values(curve_tag=self.curve_tag,
        #                                            initial_date=self.initial_date,
        #                                            final_date=self.final_date)

        self.spot_price = self.underlying_quote_handler(initial_date=self.initial_date,
                                                        final_date=self.final_date)

        process = OrderedDict()
        for ts in self.ts_options:
            process[ts.ts_name] = OrderedDict()
            for date_value in ts.ivol_mid.ts_values.index:
                process[ts.ts_name][date_value] = ql.BlackScholesMertonProcess(
                    self.spot_price[date_value],
                    self.dividend[date_value],
                    self.yield_curve[ts.ts_name][date_value],
                    # self.yield_curve.yield_curve_relinkable_handle(date=date_value),
                    self.volatility[ts.ts_name][date_value])

        self.process = process
        return self

    def update_only_yield_curve(self, curve_tag):

        if self.process is None:
            self.update_process()

        yield_curve = self.yield_curve_values(curve_tag=curve_tag,
                                              initial_date=self.initial_date,
                                              final_date=self.final_date)

        process = OrderedDict()
        for ts in self.ts_options:
            process[ts.ts_name] = OrderedDict()
            for date_value in ts.ts_values.index:
                process[ts.ts_name][date_value] = ql.BlackScholesMertonProcess(
                    self.spot_price[date_value],
                    self.dividend[date_value],
                    yield_curve.yield_curve_handle(date=date_value),
                    self.volatility[ts.ts_name][date_value])

        self.process = process
        return self

    @conditional_vectorize('date, spot_price')
    def update_only_spot_price(self, date, spot_price, ts_name):

        dt_date = to_datetime(date)
        if self.process is None:
            self.update_process()

        ql_spot_price = to_ql_quote_handle(spot_price)

        process = ql.BlackScholesMertonProcess(ql_spot_price,
                                               self.dividend[dt_date],
                                               self.yield_curve[ts_name][dt_date],
                                               self.volatility[ts_name][dt_date])
        return process

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
            self.dividend[dt_date] = self.dividend_yield_from_ts_values(initial_date=dt_date,
                                                                        final_date=self.final_date)
            dividend = self.dividend[dt_date]

        try:
            yield_curve = self.yield_curve[ts_name][dt_date]

        except KeyError:
            self.yield_curve[ts_name][dt_date] = self.yield_curve_flat(maturity=maturity,
                                                                       initial_date=self.initial_date,
                                                                       final_date=self.final_date,
                                                                       curve_tag=self.curve_tag)
            yield_curve = self.yield_curve[ts_name][dt_date]

        try:
            spot_price = self.spot_price[dt_date]

        except KeyError:
            self.spot_price[dt_date] = self.underlying_quote_handler(initial_date=dt_date,
                                                                     final_date=self.final_date)
            spot_price = self.spot_price[dt_date]

        self.process[ts_name][dt_date] = ql.BlackScholesMertonProcess(
            spot_price,
            dividend,
            yield_curve,
            self.volatility[ts_name][dt_date])

        return self

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
            self.yield_curve[ts_name][dt_date],
            volatility_term)

        return process
