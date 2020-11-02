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
from collections import namedtuple


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


heston_constants = namedtuple('heston_constants', ['long_term_variance', 'mean_reversion', 'vol_of_vol', 'correlation',
                                                   'spot_variance'])


class Heston(BaseEquityProcess):
    """
    Model for the Heston model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None, spot_variance_ts=None, mean_reversion_ts=None,
                 long_term_variance_ts=None, volatility_of_volatility_ts=None, correlation_ts=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "HESTON"
        # Heston Model parameters timeseries
        self.spot_variance_ts = spot_variance_ts  # spot variance
        self.mean_reversion_ts = mean_reversion_ts  # mean reversion strength
        self.long_term_variance_ts = long_term_variance_ts  # mean reversion variance
        self.volatility_of_volatility_ts = volatility_of_volatility_ts  # volatility of volatility
        self.correlation_ts = correlation_ts  # correlation between the asset price and its variance

    @staticmethod
    def volatility_zero_test(kappa, theta, sigma):

        if 2*kappa*theta > sigma**2:
            return True
        else:
            return False

    def process(self, spot_variance=None, mean_reversion=None, long_term_variance=None,
                volatility_of_volatility=None, correlation=None, discretization=None, **kwargs):

        if discretization is None:
            discretization = ql.HestonProcess.QuadraticExponentialMartingale

        return ql.HestonProcess(self.risk_free_handle, self.dividend_handle, self.spot_price_handle,
                                spot_variance, mean_reversion, long_term_variance, volatility_of_volatility,
                                correlation, discretization)

    def get_constant_values(self, date, last_available=True, fill_value=np.nan):
        return heston_constants(self.long_term_variance_ts.get_values(date, last_available, fill_value),
                                self.mean_reversion_ts.get_values(date, last_available, fill_value),
                                self.volatility_of_volatility_ts.get_values(date, last_available, fill_value),
                                self.correlation_ts.get_values(date, last_available, fill_value),
                                self.spot_variance_ts.get_values(date, last_available, fill_value))

    def heston_black_vol_surface(self, date, spot_volatility=None, spot_variance=None, last_available=True,
                                 fill_value=np.nan, discretization=None):

        constants = self.get_constant_values(date, last_available, fill_value)
        if spot_volatility is not None:
            variance = spot_volatility * spot_volatility
            long_term_variance = variance
        elif spot_variance is not None:
            variance = spot_variance
            long_term_variance = variance
        else:
            variance = constants.spot_variance
            long_term_variance = constants.long_term_variance

        heston_process = self.process(spot_variance=variance,
                                      mean_reversion=constants.mean_reversion,
                                      long_term_variance=long_term_variance,
                                      volatility_of_volatility=constants.vol_of_vol,
                                      correlation=constants.correlation,
                                      discretization=discretization)
        heston_model = ql.HestonModel(heston_process)
        return ql.HestonBlackVolSurface(ql.HestonModelHandle(heston_model))


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


class Bates(BaseEquityProcess):
    """
    Model for the Heston model used to evaluate options.

    :param calendar: QuantLib.Calendar
        The option calendar used to evaluate the model
    :param day_counter: QuantLib.DayCounter
        The option day count used to evaluate the model
    """
    def __init__(self, calendar=None, day_counter=None, spot_variance_ts=None, mean_reversion_ts=None,
                 long_term_variance_ts=None, volatility_of_volatility_ts=None, correlation_ts=None,
                 bates_lambda_ts=None, bates_nu_ts=None, bates_delta_ts=None):
        super().__init__(calendar=calendar, day_counter=day_counter)
        self.process_name = "BATES"
        # Heston Model parameters timeseries
        self.spot_variance_ts = spot_variance_ts  # spot variance
        self.mean_reversion_ts = mean_reversion_ts  # mean reversion strength
        self.long_term_variance_ts = long_term_variance_ts  # mean reversion variance
        self.volatility_of_volatility_ts = volatility_of_volatility_ts  # volatility of volatility
        self.correlation_ts = correlation_ts  # correlation between the asset price and its variance

    @staticmethod
    def volatility_zero_test(kappa, theta, sigma):

        if 2*kappa*theta > sigma**2:
            return True
        else:
            return False

    def process(self, spot_variance=None, mean_reversion=None, long_term_variance=None,
                volatility_of_volatility=None, correlation=None, bates_lambda=None, bates_nu=None, bates_delta=None,
                **kwargs):

        return ql.BatesProcess(self.risk_free_handle, self.dividend_handle, self.spot_price_handle,
                               spot_variance, mean_reversion, long_term_variance, volatility_of_volatility,
                               correlation, bates_lambda, bates_nu, bates_delta)
