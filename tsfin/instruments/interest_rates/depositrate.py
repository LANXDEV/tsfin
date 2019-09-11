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
from tsfin.constants import TENOR_PERIOD, COMPOUNDING, FREQUENCY, ISSUE_DATE, INDEX, INDEX_TENOR
from tsfin.base.instrument import default_arguments
from tsfin.base import Instrument, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, \
    to_ql_compounding, to_ql_rate_index


DEFAULT_ISSUE_DATE = ql.Date.minDate()


class DepositRate(Instrument):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries)
        try:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        except KeyError:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._tenor)
        self.calendar = self.index.fixingCalendar()
        self.day_counter = self.index.dayCounter()
        self.compounding = to_ql_compounding(self.ts_attributes[COMPOUNDING])
        self.frequency = to_ql_frequency(self.ts_attributes[FREQUENCY])
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        try:
            self.issue_date = to_ql_date(to_datetime(self.ts_attributes[ISSUE_DATE]))
        except KeyError:
            self.issue_date = DEFAULT_ISSUE_DATE

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
        start_date = self.calendar.adjust(start_date, self.business_convention)
        end_date = self.calendar.adjust(end_date, self.business_convention)
        fixing_dates = list()
        maturity_dates = list()
        fixing_date = self.index.fixingDate(start_date)
        maturity_date = self.index.maturityDate(self.index.valueDate(fixing_date))
        while maturity_date < end_date:
            fixing_dates.append(fixing_date)
            maturity_dates.append(maturity_date)
            fixing_date = self.index.fixingDate(maturity_date)
            maturity_date = self.index.maturityDate(self.index.valueDate(fixing_date))
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
        return self._tenor

    def maturity(self, date, *args, **kwargs):
        """Get maturity based on a date and tenor of the deposit rate.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        QuantLib.Date
            The maturity based on the reference date and tenor of the deposit rate.
        """
        date = to_ql_date(date)
        maturity = self.index.maturityDate(date)
        return maturity

    @default_arguments
    @conditional_vectorize('date')
    def performance(self, start_date=None, date=None, spread=None, **kwargs):
        """Performance of investment in the interest rate, taking tenor into account.

        If the period between start_date and date is larger the the deposit rate's tenor, considers the investment
        is rolled at the prevailing rate at each maturity.

        Parameters
        ----------
        start_date: QuantLib.Date
            Start date of the investment period.
        date: QuantLib.Date, c-vectorized
            End date of the investment period.
        spread: float
            rate to be added to the return calculation.

        Returns
        -------
        scalar
            Performance of the investment.
        """
        first_available_date = self.quotes.ts_values.first_valid_index()
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

        if spread is not None:
            fixings += spread
        return np.prod([ql.InterestRate(
            fixing, self.day_counter, self.compounding, self.frequency).compoundFactor(
                self.index.valueDate(fixing_date), maturity_date, start_date, date)
                    for fixing, fixing_date, maturity_date in zip(fixings, fixing_dates, maturity_dates)
                    if self.index.valueDate(fixing_date) <= maturity_date]) - 1

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
        rate = self.timeseries.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        date = to_ql_date(date)
        try:
            tenor = self.tenor(date)
        except ValueError:
            # Return none if the deposit rate can't retrieve a tenor (i.e. is expired).
            return None
        # Convert rate to simple compounding because DepositRateHelper expects simple rates.
        time = self.day_counter.yearFraction(date, self.index.maturityDate(date))
        rate = ql.InterestRate(rate, self.day_counter, self.compounding,
                               self.frequency).equivalentRate(ql.Simple, ql.Annual, time).rate()
        final_rate = ql.SimpleQuote(rate)
        return ql.DepositRateHelper(ql.QuoteHandle(final_rate), tenor, self.fixing_days, self.calendar,
                                    self.business_convention, self.month_end, self.day_counter)
