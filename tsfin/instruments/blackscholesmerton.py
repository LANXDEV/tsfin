# Copyright (C) 2016-2018 Lanx Capital Investimentos LTDA.
#
# This file is part of Time Series Finance (tsfin).
#
# Time Series Finance (tsfin) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# Time Series Finance (tsfin) is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Time Series Finance (tsfin). If not, see <https://www.gnu.org/licenses/>.
"""
A class for modelling Black Scholes Merton
"""
import QuantLib as ql
from collections import OrderedDict
from tsfin.base import to_ql_date


class BlackScholesMerton:

    def __init__(self, equity_instruments, yield_curve):
        """ Model for the Black Scholes Merton model used to evaluate options.

        :param equity_instruments: :py:obj:Equity
            The instrument class representing an Equity (Stocks and ETFs)
        :param yield_curve: :py:obj:YieldCurveTimeSeries
            The yield curve of the index rate, used to estimate future cash flows.
        """
        self.equity_instruments = equity_instruments
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

    def spot_price_update(self, date, underlying_name, spot_price=None, last_available=False, **kwargs):
        """

        :param date: date-like
            The date.
        :param underlying_name: str
            The underlying Timeseries ts_name
        :param spot_price: float, optional
            An override of the underlying spot price in case you don't wan't to use the timeseries one.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        """

        date = to_ql_date(date)
        if spot_price is None:
            equity_instrument = self.equity_instruments.get(underlying_name)
            spot_price = float(equity_instrument.spot_price(date=date, last_available=last_available))
        else:
            spot_price = spot_price

        self.spot_price_handle.linkTo(ql.SimpleQuote(spot_price))

    def dividend_yield_update(self, date, calendar, day_counter, underlying_name, dividend_yield=None, dvd_tax_adjust=1,
                              last_available=True, compounding=ql.Continuous, **kwargs):
        """

        :param date: date-like
            The date.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param underlying_name: str
            The timeseries ts_name used to query the stored timeseries in self.ts_underlying
        :param dividend_yield: float, optional
            An override of the dividend yield in case you don't wan't to use the timeseries one.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param compounding: QuantLib.Compounding, default=Continuous
            The compounding used to interpolate the curve.
        """
        date = to_ql_date(date)
        if dividend_yield is None:
            equity_instrument = self.equity_instruments.get(underlying_name)
            dividend_yield = float(equity_instrument.dividend_yield(date=date, last_available=last_available))
        else:
            dividend_yield = dividend_yield

        final_dividend = ql.SimpleQuote(dividend_yield * dvd_tax_adjust)

        dividend = ql.FlatForward(0, calendar, ql.QuoteHandle(final_dividend), day_counter, compounding)
        self.dividend_handle.linkTo(dividend)

    def yield_curve_update(self, date, calendar, day_counter, maturity, risk_free=None, compounding=ql.Continuous,
                           frequency=ql.Once, **kwargs):
        """

        :param date: date-like
            The date.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: date-like
            The option maturity.
        :param risk_free: float, optional
            An override of the zero rate in case you don't wan't to use the yield curve implied one.
        :param compounding: QuantLib.Compounding, default=Continuous
            The compounding used to interpolate the curve.
        :param frequency: QuantLib.Frequency, default = Once
            The frequency of the quoted yield.
        """
        date = to_ql_date(date)
        mat_date = to_ql_date(maturity)
        if risk_free is None:
            zero_rate = ql.SimpleQuote(self.yield_curve.zero_rate_to_date(date=date, to_date=mat_date,
                                                                          compounding=compounding,
                                                                          frequency=frequency))
        else:
            zero_rate = ql.SimpleQuote(risk_free)
        yield_curve = ql.FlatForward(0, calendar, ql.QuoteHandle(zero_rate), day_counter, compounding)
        self.risk_free_handle.linkTo(yield_curve)

    def volatility_update(self, vol_value, calendar, day_counter):

        """
        :param vol_value: float, optional
            An override of the volatility value in case you don't wan't to use the timeseries one.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :return:
        """
        volatility_value = ql.SimpleQuote(vol_value)
        back_constant_vol = ql.BlackConstantVol(0, calendar, ql.QuoteHandle(volatility_value), day_counter)
        self.volatility_handle.linkTo(back_constant_vol)

    def update_process(self, date, calendar, day_counter, maturity, underlying_name, vol_value, dvd_tax_adjust=1,
                       last_available=True, **kwargs):
        """

        :param date: date-like
            The date.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: date-like
            The option maturity.
        :param underlying_name: str
            The underlying Timeseries ts_name
        :param vol_value: float
            The volatility value to be used in the model.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :return: bool
            Return True if the volatility timeseries was updated.
        """
        date = to_ql_date(date)
        self.spot_price_update(date=date, underlying_name=underlying_name, last_available=last_available,
                               **kwargs)
        self.dividend_yield_update(date=date, calendar=calendar, day_counter=day_counter,
                                   underlying_name=underlying_name, dvd_tax_adjust=dvd_tax_adjust,
                                   last_available=last_available, **kwargs)
        self.yield_curve_update(date=date, calendar=calendar, day_counter=day_counter, maturity=maturity,
                                **kwargs)
        self.volatility_update(vol_value=vol_value, calendar=calendar, day_counter=day_counter)
