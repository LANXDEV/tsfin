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
A class for modelling volatility processes and its different implementations
"""
import QuantLib as ql
from tsfin.base import to_ql_date


class BaseEquityProcess:

    def __init__(self, yield_curve):
        """ Base Model for QuantLib Stochastic Process for Equities and Equity Options.

        :param yield_curve: :py:obj:YieldCurveTimeSeries
            The yield curve of the index rate, used to estimate future cash flows.
        """
        self.process_name = None
        self.yield_curve = yield_curve
        self.risk_free_handle = ql.RelinkableYieldTermStructureHandle()
        self.dividend_handle = ql.RelinkableYieldTermStructureHandle()
        self.spot_price_handle = ql.RelinkableQuoteHandle()
        self.volatility_handle = None
        # Heston Model parameters
        self.kappa = None  # mean reversion strength
        self.theta = None  # mean reversion variance
        self.sigma = None  # volatility of volatility
        self.rho = None  # correlation between the asset price and its variance
        self.process = None

    def spot_price_update(self, spot_price):
        """
        :param spot_price: float, optional
            An override of the underlying spot price in case you don't wan't to use the timeseries one.
        """
        if isinstance(spot_price, ql.SimpleQuote):
            spot = spot_price
        else:
            spot = ql.SimpleQuote(spot_price)
        self.spot_price_handle.linkTo(spot)

    def dividend_yield_update(self, dividend_yield, calendar, day_counter, dividend_tax,
                              compounding=ql.Continuous):
        """

        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param dividend_tax: float
            The dividend % tax applied.
        :param compounding: QuantLib.Compounding, default=Continuous
            The compounding used to interpolate the curve.
        """
        final_dividend = ql.SimpleQuote(dividend_yield * (1-float(dividend_tax)))
        dividend = ql.FlatForward(0, calendar, ql.QuoteHandle(final_dividend), day_counter, compounding)
        self.dividend_handle.linkTo(dividend)

    def yield_curve_update(self, date, base_date, calendar, day_counter, maturity, compounding=ql.Continuous,
                           frequency=ql.NoFrequency, zero_rate=None):
        """

        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param compounding: QuantLib.Compounding, default=Continuous
            The compounding used to interpolate the curve.
        :param frequency: QuantLib.Frequency
            Frequency convention for the rate.
        :param zero_rate: float
            The risk-free rate to be used by the model (Annualized)
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        if date <= base_date:
            base_date = date
        if zero_rate is None:
            zero_rate = self.yield_curve.forward_rate_date_to_date(date=base_date, to_date1=date, to_date2=maturity,
                                                                   compounding=compounding, frequency=frequency,
                                                                   day_counter=day_counter)
        else:
            zero_rate = float(zero_rate)
        risk_free_curve = ql.FlatForward(0, calendar, ql.QuoteHandle(ql.SimpleQuote(zero_rate)), day_counter,
                                         compounding)
        self.risk_free_handle.linkTo(risk_free_curve)

    def volatility_update(self, volatility, calendar, day_counter):

        """
        :param volatility: float, optional
            An override of the volatility value in case you don't wan't to use the timeseries one.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :return: None
            Defined in the child class.
        """
        return None

    def update_process(self, date, base_date, spot_price, dividend_yield, calendar, day_counter, volatility, maturity,
                       dividend_tax, kappa=None, theta=None, sigma=None, rho=None):
        """

        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            The underlying spot price.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param volatility: float
            The volatility value to be used in the model.
        :param maturity: ql.Date
            The option maturity date.
        :param dividend_tax: float
            The dividend % tax applied.
        :param kappa: float
            The mean reversion strength, used in the Heston Model.
        :param theta: float
            The mean reversion variance, used in the Heston Model.
        :param sigma: float
            The volatility of the volatility, used in the Heston Model
        :param rho: float
            The correlation between the asset price and its variance, used in the Heston Model
        :return: bool
            Return True if the volatility timeseries was updated.
        """
        self.spot_price_update(spot_price=spot_price)
        self.dividend_yield_update(dividend_yield=dividend_yield, calendar=calendar, day_counter=day_counter,
                                   dividend_tax=dividend_tax)
        self.yield_curve_update(date=date, base_date=base_date, maturity=maturity, calendar=calendar,
                                day_counter=day_counter)
        self.volatility_update(volatility=volatility, calendar=calendar, day_counter=day_counter)
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho


class BlackScholesMerton(BaseEquityProcess):
    """
    Model for the Black Scholes Merton model used to evaluate options.

    :param yield_curve: :py:obj:YieldCurveTimeSeries
        The yield curve of the index rate, used to estimate future cash flows.
    """
    def __init__(self, yield_curve):
        super().__init__(yield_curve)
        self.process_name = "BLACK_SCHOLES_MERTON"
        self.volatility_handle = ql.RelinkableBlackVolTermStructureHandle()
        self.process = ql.BlackScholesMertonProcess(self.spot_price_handle,
                                                    self.dividend_handle,
                                                    self.risk_free_handle,
                                                    self.volatility_handle)

    def volatility_update(self, volatility, calendar, day_counter):

        """
        :param volatility: float, optional
            An override of the volatility value in case you don't wan't to use the timeseries one.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :return:
        """
        if isinstance(volatility, ql.SimpleQuote):
            volatility_value = volatility
        else:
            volatility_value = ql.SimpleQuote(volatility)
        black_constant_vol = ql.BlackConstantVol(0, calendar, ql.QuoteHandle(volatility_value), day_counter)
        self.volatility_handle.linkTo(black_constant_vol)


class BlackScholes(BaseEquityProcess):
    """
    Model for the Black Scholes model used to evaluate options.

    :param yield_curve: :py:obj:YieldCurveTimeSeries
        The yield curve of the index rate, used to estimate future cash flows.
    """
    def __init__(self, yield_curve):
        super().__init__(yield_curve)
        self.process_name = "BLACK_SCHOLES"
        self.volatility_handle = ql.RelinkableBlackVolTermStructureHandle()
        self.process = ql.BlackScholesProcess(self.spot_price_handle,
                                              self.risk_free_handle,
                                              self.volatility_handle)

    def volatility_update(self, volatility, calendar, day_counter):

        """
        :param volatility: float, optional
            An override of the volatility value in case you don't wan't to use the timeseries one.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :return:
        """
        if isinstance(volatility, ql.SimpleQuote):
            volatility_value = volatility
        else:
            volatility_value = ql.SimpleQuote(volatility)
        black_constant_vol = ql.BlackConstantVol(0, calendar, ql.QuoteHandle(volatility_value), day_counter)
        self.volatility_handle.linkTo(black_constant_vol)

    def dividend_yield_update(self, dividend_yield, calendar, day_counter, dividend_tax, compounding=ql.Continuous):
        """

        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param dividend_tax: float
            The dividend % tax applied.
        :param compounding: QuantLib.Compounding, default=Continuous
            The compounding used to interpolate the curve.
        """

        self.dividend_handle = None


class HestonProcess(BaseEquityProcess):
    """
    Model for the Heston model used to evaluate options.

    :param yield_curve: :py:obj:YieldCurveTimeSeries
        The yield curve of the index rate, used to estimate future cash flows.
    """
    def __init__(self, yield_curve):
        super().__init__(yield_curve)
        self.process_name = "HESTON"
        self.volatility_handle = 0  # spot variance
        self.process = ql.HestonProcess(self.risk_free_handle,
                                        self.dividend_handle,
                                        self.spot_price_handle,
                                        self.volatility_handle,
                                        self.kappa,
                                        self.theta,
                                        self.sigma,
                                        self.rho)

    def volatility_update(self, volatility, calendar, day_counter):

        """
        :param volatility: float, optional
            An override of the volatility value in case you don't wan't to use the timeseries one.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :return:
        """
        if isinstance(volatility, ql.SimpleQuote):
            volatility_value = volatility.value() * volatility.value()
        else:
            volatility_value = volatility * volatility
        self.volatility_handle = volatility_value

    def heston_parameters_update(self, kappa, theta, sigma, rho):

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
