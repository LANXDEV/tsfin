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


def to_ql_equity_process(process_name):
    if process_name.upper() == 'BLACK_SCHOLES_MERTON':
        return BlackScholesMerton
    elif process_name.upper() == 'BLACK_SCHOLES':
        return BlackScholes
    elif process_name.upper() == 'HESTON':
        return HestonProcess


class BaseEquityProcess:

    def __init__(self, calendar=None, day_counter=None):
        """ Base Model for QuantLib Stochastic Process for Equities and Equity Options.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        """
        self.process_name = None
        self.calendar = calendar
        self.day_counter = day_counter
        self.compounding = ql.Continuous
        self.volatility = ql.SimpleQuote(0)
        self.risk_free_rate = ql.SimpleQuote(0)
        self.dividend_yield = ql.SimpleQuote(0)
        self.spot_price = ql.SimpleQuote(0)
        self.risk_free_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, self.calendar,
                                                                           ql.QuoteHandle(self.risk_free_rate),
                                                                           self.day_counter, self.compounding))
        self.dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, self.calendar,
                                                                          ql.QuoteHandle(self.dividend_yield),
                                                                          self.day_counter, self.compounding))
        self.spot_price_handle = ql.QuoteHandle(self.spot_price)
        self.volatility_handle = None
        # Heston Model parameters
        self.kappa = ql.SimpleQuote(0)  # mean reversion strength
        self.theta = ql.SimpleQuote(0)  # mean reversion variance
        self.sigma = ql.SimpleQuote(0)  # volatility of volatility
        self.rho = ql.SimpleQuote(0)  # correlation between the asset price and its variance
        self.process = None


class BlackScholesMerton(BaseEquityProcess):
    """
    Model for the Black Scholes Merton model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "BLACK_SCHOLES_MERTON"
        self.black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql.QuoteHandle(self.volatility),
                                                      self.day_counter)
        self.volatility_handle = ql.BlackVolTermStructureHandle(self.black_constant_vol)
        self.process = ql.BlackScholesMertonProcess(self.spot_price_handle,
                                                    self.dividend_handle,
                                                    self.risk_free_handle,
                                                    self.volatility_handle)


class BlackScholes(BaseEquityProcess):
    """
    Model for the Black Scholes model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "BLACK_SCHOLES"
        self.black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql.QuoteHandle(self.volatility),
                                                      self.day_counter)
        self.volatility_handle = ql.BlackVolTermStructureHandle(self.black_constant_vol)
        self.process = ql.BlackScholesProcess(self.spot_price_handle,
                                              self.risk_free_handle,
                                              self.volatility_handle)


class HestonProcess(BaseEquityProcess):
    """
    Model for the Heston model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "HESTON"
        self.process = ql.HestonProcess(self.risk_free_handle,
                                        self.dividend_handle,
                                        self.spot_price_handle,
                                        self.volatility.value(),
                                        self.kappa.value(),
                                        self.theta.value(),
                                        self.sigma.value(),
                                        self.rho.value())
