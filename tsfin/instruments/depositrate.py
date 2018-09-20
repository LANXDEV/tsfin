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
import numpy as np
import QuantLib as ql
from tsfin.constants import CALENDAR, TENOR_PERIOD, MATURITY_DATE, BUSINESS_CONVENTION, \
    COMPOUNDING, FREQUENCY, DAY_COUNTER, FIXING_DAYS
from tsfin.base.instrument import default_arguments
from tsfin.base import Instrument, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, \
    to_ql_business_convention, to_ql_calendar, to_ql_compounding, to_ql_day_counter


class DepositRate(Instrument):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        try:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        except KeyError:
            # If the deposit rate has no tenor, it must have a maturity.
            self._maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.compounding = to_ql_compounding(self.ts_attributes[COMPOUNDING])
        self.frequency = to_ql_frequency(self.ts_attributes[FREQUENCY])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])

    def is_expired(self, date, *args, **kwargs):
        """Check if the deposit rate is expired.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        bool
            Whether the instrument is expired at `date`.
        """
        try:
            date = to_ql_date(date)
            if date >= self._maturity:
                return True
        except AttributeError:
            pass
        return False

    @conditional_vectorize('date')
    def value(*args, **kwargs):
        """Returns zero.
        """
        return 0

    def _get_fixing_maturity_dates(self, start_date, end_date):
        start_date = self.calendar.adjust(start_date)
        end_date = self.calendar.adjust(end_date)
        fixing_dates = list()
        maturity_dates = list()
        fixing_date = self.calendar.adjust(start_date)
        maturity_date = self.calendar.advance(fixing_date, self.tenor(start_date))
        while maturity_date < end_date:
            fixing_dates.append(fixing_date)
            maturity_dates.append(maturity_date)
            fixing_date = maturity_date
            maturity_date = self.calendar.advance(fixing_date, self.tenor(start_date))
        fixing_dates.append(fixing_date)
        maturity_dates.append(end_date)
        return fixing_dates, maturity_dates

    def tenor(self, date, *args, **kwargs):
        """Get tenor of the deposit rate.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        QuantLib.Period
            The tenor (period) to maturity of the deposit rate.
        """
        try:
            # If the object has a tenor attribute, return it.
            assert self._tenor
            return self._tenor
        except (AttributeError, AssertionError):
            # If no tenor, then it must have a maturity. Use it to calculate the tenor.
            date = to_ql_date(date)
            if self.is_expired(date):
                raise ValueError("The requested date is equal or higher than the instrument's maturity: {}".format(
                    self.name))
            # TODO: Check if the below calculation yields correct results when creating deposit rate helpers.
            # return ql.Period(self.maturity - date, ql.Days)
            return ql.Period(self.calendar.businessDaysBetween(date, self.maturity), ql.Days)

    @default_arguments
    @conditional_vectorize('date')
    def performance(self, start_date=None, date=None, **kwargs):
        """Performance of investment in the interest rate, taking tenor into account.

        If the period between start_date and date is larger the the deposit rate's tenor, considers the investment
        is rolled at the prevailing rate at each maturity.

        Parameters
        ----------
        start_date: QuantLib.Date
            Start date of the investment period.
        date: QuantLib.Date, c-vectorized
            End date of the investment period.

        Returns
        -------
        scalar
            Performance of the investment.
        """
        first_available_date = self.quotes.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if date < start_date:
            return np.nan
        start_date = to_ql_date(start_date)
        date = to_ql_date(date)
        fixing_dates, maturity_dates = self._get_fixing_maturity_dates(start_date, date)
        fixings = self.timeseries.get_values(index=[to_datetime(fixing_date) for fixing_date in fixing_dates])
        return np.prod([ql.InterestRate(fixing, self.day_counter, self.compounding,
                                        self.frequency).compoundFactor(fixing_date, maturity_date, start_date, date)
                       for fixing, fixing_date, maturity_date in zip(fixings, fixing_dates, maturity_dates)]) - 1

    def rate_helper(self, date, last_available=True, **other_args):
        """Helper for yield curve construction.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool, optional
            Whether to use last available quotes if missing data.

        Returns
        -------
        QuantLib.RateHelper
            Rate helper for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series
        if self.is_expired(date):
            return None
        rate = self.get_values(index=date, last_available=last_available, fill_value=np.nan)
        # print("{0} is returning a rate helper with rate {1}".format(self.ts_name, rate))
        if np.isnan(rate):
            return None
        date = to_ql_date(date)
        try:
            tenor = self.tenor(date)
        except ValueError:
            # Return none if the deposit rate can't retrieve a tenor (i.e. is expired).
            return None
        # Convert rate to simple compounding because DepositRateHelper expects simple rates.
        time = self.day_counter.yearFraction(date, self.calendar.advance(date, tenor))
        rate = ql.InterestRate(rate, self.day_counter, self.compounding,
                               self.frequency).equivalentRate(ql.Simple, ql.Annual, time).rate()
        return ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), tenor, self.fixing_days, self.calendar,
                                    self.business_convention, False, self.day_counter)
