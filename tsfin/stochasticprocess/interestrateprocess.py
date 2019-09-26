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
A class for modelling interest rate processes and its different implementations
"""
import QuantLib as ql
import numpy as np
from tsfin.tools import to_datetime
from tsfin.base import to_ql_short_rate_model


def get_interest_rate_process(model_name, yield_curve):
    if model_name == "HULL_WHITE":
        return HullWhiteProcess(yield_curve=yield_curve)
    elif model_name == "G2":
        return G2Process(yield_curve=yield_curve)
    else:
        raise print("No model available with this name")


class BaseInterestRateProcess:

    def __init__(self, yield_curve):
        """ Base Model for QuantLib Stochastic Process for Interest Rates.

        :param yield_curve: :py:obj:YieldCurveTimeSeries
            The yield curve of the index rate, used to estimate future cash flows.
        """
        self.process_name = None
        self.yield_curve = yield_curve
        self.yield_curve_handle = ql.RelinkableYieldTermStructureHandle()
        self.model = None
        self.process = None
        self.mean = 0
        self.sigma = 0
        self.beta = 0
        self.eta = 0
        self.rho = 0

    def yield_curve_update(self, date=None, yield_curve=None):

        if date is not None and yield_curve is None:
            self.yield_curve_handle.linkTo(self.yield_curve.yield_curve(date=date))
        elif date is None and isinstance(yield_curve, ql.YieldTermStructure):
            self.yield_curve_handle.linkTo(yield_curve)
        elif date is not None and yield_curve is not None:
            self.yield_curve_handle.linkTo(yield_curve.yield_curve(date=date))

    def update_process(self, mean, sigma, beta, eta, rho, date=None, yield_curve=None, **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)
        self.mean = mean
        self.sigma = sigma
        self.beta = beta
        self.eta = eta
        self.rho = rho

    def update_model(self, date=None, yield_curve=None, **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)

    def discount_bond(self, to_time1, to_time2):

        rate = self.yield_curve_handle.zeroRate(to_time1, ql.Continuous, ql.NoFrequency).rate()
        return self.model.discountBond(to_time1, to_time2, rate)


class HullWhiteProcess(BaseInterestRateProcess):

    def __init__(self, yield_curve):
        super().__init__(yield_curve)
        self.process_name = "HULL_WHITE"

    def update_process(self, mean, sigma, beta=0, eta=0, rho=0, date=None, yield_curve=None,
                       **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)
        self.mean = mean
        self.sigma = sigma
        self.beta = beta
        self.eta = eta
        self.rho = rho
        self.process = ql.HullWhiteProcess(self.yield_curve_handle, self.mean, self.sigma)

    def update_process_from_ts(self, date, mean_timeseries, sigma_timeseries, yield_curve=None, last_available=True,
                               fill_value=np.nan):

        date = to_datetime(date)
        mean = mean_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        sigma = sigma_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        self.update_process(date=date, yield_curve=yield_curve, mean=mean, sigma=sigma)
        return self.process

    def update_model(self, date=None, yield_curve=None, **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)
        mean = kwargs.get('mean', self.mean)
        sigma = kwargs.get('sigma', self.sigma)
        self.model = to_ql_short_rate_model(self.process_name, self.yield_curve_handle, mean, sigma)


class G2Process(BaseInterestRateProcess):

    def __init__(self, yield_curve):
        super().__init__(yield_curve)
        self.process_name = "G2"
        self.model = ql.G2

    def update_process(self, mean, sigma, beta, eta, rho, date=None, yield_curve=None, **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)
        self.mean = mean
        self.sigma = sigma
        self.beta = beta
        self.eta = eta
        self.rho = rho
        self.process = ql.G2Process(self.mean, self.sigma, self.beta, self.eta, self.rho)

    def update_process_from_ts(self, date, mean_timeseries, sigma_timeseries, beta_timeseries, eta_timeseries,
                               rho_timeseries, yield_curve=None, last_available=True, fill_value=np.nan):

        date = to_datetime(date)
        mean = mean_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        sigma = sigma_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        beta = beta_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        eta = eta_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        rho = rho_timeseries.get_values(index=date, last_available=last_available, fill_value=fill_value)
        self.update_process(date=date, yield_curve=yield_curve, mean=mean, sigma=sigma, beta=beta, eta=eta, rho=rho)
        return self.process

    def update_model(self, date=None, yield_curve=None, **kwargs):

        self.yield_curve_update(date=date, yield_curve=yield_curve)
        mean = kwargs.get('mean', self.mean)
        sigma = kwargs.get('sigma', self.sigma)
        beta = kwargs.get('beta', self.beta)
        eta = kwargs.get('eta', self.eta)
        rho = kwargs.get('rho', self.rho)
        self.model = to_ql_short_rate_model(self.process_name, self.yield_curve_handle, mean, sigma, beta, eta, rho)

