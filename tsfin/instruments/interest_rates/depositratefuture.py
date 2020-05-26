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
DepositRate class, to represent deposit rates.
"""
import QuantLib as ql
import numpy as np
from tsfin.constants import CALENDAR, MATURITY_DATE, BUSINESS_CONVENTION, DAY_COUNTER, FIXING_DAYS, SPREAD_TAG
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_business_convention, to_ql_calendar, to_ql_day_counter, to_ql_date, to_ql_ibor_index


class DepositRateFuture(BaseInterestRate):
    def __init__(self, timeseries, telescopic_value_dates=True):
        super().__init__(timeseries, telescopic_value_dates=telescopic_value_dates)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self._maturity = to_ql_date(self.ts_attributes[MATURITY_DATE])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])
        self.month_end = False
        self.index = to_ql_ibor_index(index=self.ts_attributes[SPREAD_TAG], fixing_days=self.fixing_days,
                                      currency=self.currency, calendar=self.calendar, day_counter=self.day_counter,
                                      overnight_index=True)
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)

    def rate_helper(self, date, last_available=True, spread=None, sigma=None, mean=None, **other_args):
        """Helper for yield curve construction.

        :param date: QuantLib.Date
            Reference date.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :param spread: float
            Rate Spread
        :param sigma: :py:obj:`TimeSeries`
            The timeseries of the sigma, used for convexity calculation
        :param mean: :py:obj:`TimeSeries`
            The timeseries of the mean, used for convexity calculation
        :return QuantLib.RateHelper
            Rate helper for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series
        date = to_ql_date(date)
        if self.is_expired(date, **other_args):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        time = self.day_counter.yearFraction(date, self.maturity(date))
        rate = ql.InterestRate(rate, self.day_counter, self.compounding, self.frequency)
        self.helper_rate.setValue(rate.equivalentRate(ql.Simple, ql.Annual, time).rate())
        return ql.DepositRateHelper(ql.QuoteHandle(self.helper_rate),
                                    ql.Period(self.day_counter.dayCount(date, self._maturity), ql.Days),
                                    self.fixing_days,
                                    self.calendar,
                                    self.business_convention,
                                    self.month_end,
                                    self.day_counter)
