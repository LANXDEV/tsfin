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
import numpy as np


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
        self.dividend_yield = ql.SimpleQuote(0)
        self.spot_price = ql.SimpleQuote(0)
        self.risk_free_handle = ql.RelinkableYieldTermStructureHandle()
        self.dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, self.calendar,
                                                                          ql.QuoteHandle(self.dividend_yield),
                                                                          self.day_counter, self.compounding))
        self.spot_price_handle = ql.QuoteHandle(self.spot_price)
        self._process = dict()


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

    def process(self, volatility, **kwargs):

        black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql.QuoteHandle(volatility), self.day_counter)
        volatility_handle = ql.BlackVolTermStructureHandle(black_constant_vol)
        return ql.BlackScholesMertonProcess(self.spot_price_handle,
                                            self.dividend_handle,
                                            self.risk_free_handle,
                                            volatility_handle)


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

    def process(self, volatility, **kwargs):

        black_constant_vol = ql.BlackConstantVol(0, self.calendar, ql.QuoteHandle(volatility), self.day_counter)
        volatility_handle = ql.BlackVolTermStructureHandle(black_constant_vol)
        return ql.BlackScholesProcess(self.spot_price_handle,
                                      self.risk_free_handle,
                                      volatility_handle)


class Heston(BaseEquityProcess):
    """
    Model for the Heston model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None, variance_ts=None, kappa_ts=None, theta_ts=None,
                 sigma_ts=None, rho_ts=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "HESTON"
        # Heston Model parameters timeseries
        self.variance_ts = variance_ts  # spot variance
        self.kappa_ts = kappa_ts  # mean reversion strength
        self.theta_ts = theta_ts  # mean reversion variance
        self.sigma_ts = sigma_ts  # volatility of volatility
        self.rho_ts = rho_ts  # correlation between the asset price and its variance

    def process(self, variance=None, kappa=None, theta=None, sigma=None, rho=None, **kwargs):
        return ql.HestonProcess(self.risk_free_handle, self.dividend_handle, self.spot_price_handle,
                                variance, kappa, theta, sigma, rho)

    def get_constant_values(self, date, last_available=True, fill_value=np.nan):
        variance = self.variance_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        kappa = self.kappa_ts.get_values(index=date, last_available=last_available,  fill_value=fill_value)
        theta = self.theta_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        sigma = self.sigma_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        rho = self.rho_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        return variance, kappa, theta, sigma, rho


class GJRGARCH(BaseEquityProcess):

    def __init__(self, calendar=None, day_counter=None, omega_ts=None, alpha_ts=None, beta_ts=None, gamma_ts=None,
                 lambda_ts=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = 'GJR_GARCH'
        self.omega_ts = omega_ts
        self.alpha_ts = alpha_ts
        self.beta_ts = beta_ts
        self.gamma_ts = gamma_ts
        self.lambda_ts = lambda_ts

    def process(self, omega=None, alpha=None, beta=None, gamma=None, lambda_value=None, v_zero=None, days_per_year=365,
                **kwargs):
        return ql.GJRGARCHProcess(self.risk_free_handle, self.dividend_handle, self.spot_price_handle, v_zero, omega,
                                  alpha, beta, gamma, lambda_value, days_per_year)

    def get_constant_values(self, date, last_available=True, fill_value=np.nan):
        omega = self.omega_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        alpha = self.alpha_ts.get_values(index=date, last_available=last_available,  fill_value=fill_value)
        beta = self.beta_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        gamma = self.gamma_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        lambda_value = self.lambda_ts.get_values(index=date, last_available=last_available, fill_value=fill_value)
        return omega, alpha, beta, gamma, lambda_value

    @staticmethod
    def v_zero(omega, alpha, beta, gamma, lambda_value):
        normal_cdf_lambda = ql.CumulativeNormalDistribution()(lambda_value)
        n = np.exp(-lambda_value * lambda_value / 2) / np.sqrt(2 * np.pi)
        q2 = 1 + lambda_value * lambda_value
        m1 = beta + (alpha + gamma * normal_cdf_lambda) * q2 + gamma * lambda_value * n
        return omega / (1 - m1)
