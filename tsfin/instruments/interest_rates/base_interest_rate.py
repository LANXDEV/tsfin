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
Base class for interest rates classes
"""
import numpy as np
import QuantLib as ql
from tsfin.constants import TENOR_PERIOD, COMPOUNDING, FREQUENCY, ISSUE_DATE, INDEX_TENOR, CURRENCY
from tsfin.base import Instrument, default_arguments, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, \
    to_ql_compounding, to_ql_currency


DEFAULT_ISSUE_DATE = ql.Date(1, 1, 2000)


class BaseInterestRate(Instrument):
    """ Base class to model interest rates.

    :param timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, is_deposit_rate=False, calculate_convexity=False, telescopic_value_dates=False):
        super().__init__(timeseries=timeseries)
        # Class Flags
        self.is_deposit_rate = is_deposit_rate
        self.calculate_convexity = calculate_convexity
        self.telescopic_value_dates = telescopic_value_dates
        # Database Attributes
        try:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        except KeyError:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        try:
            self.issue_date = to_ql_date(to_datetime(self.ts_attributes[ISSUE_DATE]))
        except KeyError:
            self.issue_date = DEFAULT_ISSUE_DATE
        self.compounding = to_ql_compounding(self.ts_attributes[COMPOUNDING])
        self.frequency = to_ql_frequency(self.ts_attributes[FREQUENCY])
        self.currency = to_ql_currency(self.ts_attributes[CURRENCY])
        # Other Interest Rate Class start from here
        self._index_tenor = None
        self._maturity = None
        # QuantLib Objects
        self.term_structure = ql.RelinkableYieldTermStructureHandle()
        self.index = None
        # QuantLib Attributes
        self.calendar = None
        self.day_counter = None
        self.business_convention = None
        self.fixing_days = None
        self.month_end = None
        self.interest_maturity_date = None
        # Rate Helper
        self.helper_rate = None
        self.helper_convexity = None
        # Defined
        self._rate_helper = None

    def link_to_term_structure(self, date, yield_curve):
        """ link a yield curve to self.term_structure

        :param date: QuantLib.Date
            The yield curve base date
        :param yield_curve: :py:obj:`YieldCurveTimeSeries"
            The yield curve to be used by the class
        :return:
        """
        date = to_ql_date(date)
        self.term_structure.linkTo(yield_curve.yield_curve(date=date))

    def set_rate(self, date, rate, spread=None, sigma=None, mean=None, **kwargs):
        """  Set Simple Quote rates to be used by the helper_rate, helper_spread or helper_convexity

        :param date: QuantLib.Date
            The reference date
        :param rate: float
            The interest rate
        :param spread: float, optional
            The spread rate to be used if any is given
        :param sigma: float
            The volatility value to be used in the convexity adjustment
        :param mean: float
            The mean reversion value to be used in the convexity adjustment
        :param kwargs:
        :return:
        """

        if self.is_deposit_rate:
            time = self.day_counter.yearFraction(date, self.maturity(date))
            rate = ql.InterestRate(rate, self.day_counter, self.compounding,
                                   self.frequency).equivalentRate(ql.Simple, ql.Annual, time).rate()
        self.helper_rate.setValue(float(rate))
        if self.calculate_convexity:
            self.convexity_bias(date=date, future_price=rate, sigma=sigma, mean=mean)

    def fixing_date(self, date):
        """The index fixing date

        :param date: ql.Date
            Reference date for calculation
        :return: ql.Date
        """
        date = to_ql_date(date)
        if self.index is not None:
            return self.index.fixingDate(date)
        else:
            return self.calendar.advance(date, -self.fixing_days, ql.Days, self.business_convention)

    def value_date(self, date):
        """The index value date

        :param date: ql.Date
            Reference date for calculation
        :return: ql.Date
        """
        date = to_ql_date(date)
        if self.index is not None:
            return self.index.valueDate(date)
        else:
            return self.calendar.advance(date, self.fixing_days, ql.Days, self.business_convention)

    def _convexity(self, future_price, date, sigma, mean):
        """ Helper function to calculate the convexity bias

        :param future_price: float
            The future price
        :param date: QuantLib.Date
            The reference date
        :param sigma: float
            The volatility of the rate
        :param mean: float
            The mean reversion value
        :return: float
            The convexity bias value
        """

        if future_price <= 0:
            return 0
        elif sigma <= 0:
            return 0
        elif mean <= 0:
            return 0

        initial_t = self.day_counter.yearFraction(date, self._maturity)
        final_t = self.day_counter.yearFraction(date, self.interest_maturity_date)
        convex = ql.HullWhite.convexityBias(future_price, initial_t, final_t, sigma, mean)
        return convex

    def convexity_bias(self, date, future_price, sigma=None, mean=None, last_available=True):
        """

        :param date: QuantLib.Date
            Reference date.
        :param future_price: float
            The future price at date
        :param sigma: float, optional
            The volatility value from the Hull White Model
        :param mean: float, optional
            The mean value from the Hull White Model
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return: float
            The convexity rate adjustment
        """

        date = to_ql_date(date)
        if sigma is None:
            return 0
        elif mean is None:
            return 0

        sigma_value = sigma.get_values(index=date, last_available=last_available, fill_value=np.nan)
        mean_value = mean.get_values(index=date, last_available=last_available, fill_value=np.nan)

        if np.isnan(sigma_value) or np.isnan(mean_value):
            convexity_bias = 0
        else:
            convexity_bias = self._convexity(future_price=future_price, date=date, sigma=sigma_value, mean=mean_value)
        self.helper_convexity.setValue(float(convexity_bias))

    def tenor(self, date, *args, **kwargs):
        """Get tenor of the deposit rate.

        :param date: QuantLib.Date
            Reference date.
        :return QuantLib.Period
            The tenor (period) to maturity of the deposit rate.
        """
        return self._tenor

    def maturity(self, date, *args, **kwargs):
        """Get maturity based on a date and tenor of the deposit rate.

        :param date: QuantLib.Date
            Reference date.
        :return QuantLib.Date
            The maturity based on the reference date and tenor of the deposit rate.
        """
        if self._maturity is not None:
            return self._maturity
        date = self.calendar.adjust(to_ql_date(date), self.business_convention)
        if (self.index is not None) and self.is_deposit_rate:
            return self.index.maturityDate(date)
        else:
            return self.calendar.advance(date, self._tenor, self.business_convention, self.month_end)

    def is_expired(self, date, *args, **kwargs):
        """Check if the deposit rate is expired.

        :param date: QuantLib.Date
            Reference date.
        :return bool
            Whether the instrument is expired at `date`.
        """
        maturity = self.maturity(date=date)
        date = to_ql_date(date)
        if date >= maturity:
            return True
        return False

    @conditional_vectorize('date')
    def value(*args, **kwargs):
        """Returns zero
        Implemented in the child class
        """
        return 0

    def _get_fixing_maturity_dates(self, start_date, end_date, fixing_at_start_date=False):
        """ Get a list fixing dates and a list maturity dates between two dates.

        :param start_date: QuantLib.Date
            The start date of the period
        :param end_date: QuantLib.Date
            The end date of the period
        :param fixing_at_start_date: bool
            If True the first fixing will be the start date instead of a date in relation to the start date
        :return: list of QuantLib.Date, list QuantLib.Date
            The list of fixing dates, the list of maturity dates
        """
        start_date = self.calendar.adjust(start_date, self.business_convention)
        end_date = self.calendar.adjust(end_date, self.business_convention)
        fixing_dates = list()
        maturity_dates = list()
        if fixing_at_start_date:
            fixing_date = start_date
        else:
            fixing_date = self.fixing_date(start_date)

        value_date = self.value_date(fixing_date)
        maturity_date = self.maturity(value_date)
        while maturity_date < end_date:
            fixing_dates.append(fixing_date)
            maturity_dates.append(maturity_date)
            fixing_date = self.fixing_date(maturity_date)
            value_date = self.value_date(fixing_date)
            maturity_date = self.maturity(value_date)

        fixing_dates.append(fixing_date)
        maturity_dates.append(end_date)
        return fixing_dates, maturity_dates

    @default_arguments
    @conditional_vectorize('date')
    def performance(self, start_date=None, date=None, spread=None, fixing_at_start_date=False, **kwargs):
        """Performance of investment in the interest rate, taking tenor into account.

        If the period between start_date and date is larger the the deposit rate's tenor, considers the investment
        is rolled at the prevailing rate at each maturity.

        :param start_date: QuantLib.Date
            Start date of the investment period.
        :param date: QuantLib.Date, c-vectorized
            End date of the investment period.
        :param spread: float
            rate to be added to the return calculation.
        :param fixing_at_start_date: bool
            Whether to use the start date as the first fixing date or not.
        :return scalar
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
        fixing_dates, maturity_dates = self._get_fixing_maturity_dates(start_date, date, fixing_at_start_date)
        fixings = self.quotes.get_values(index=[to_datetime(fixing_date) for fixing_date in fixing_dates])

        if spread is not None:
            fixings += spread

        return np.prod([
            ql.InterestRate(fixing, self.day_counter, self.compounding, self.frequency).compoundFactor(
                self.value_date(fixing_date), maturity_date, start_date, date)
            for fixing, fixing_date, maturity_date in zip(fixings, fixing_dates, maturity_dates)
            if self.value_date(fixing_date) <= maturity_date
        ]) - 1

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
        return None

    def spread_rate(self, date, last_available=True, **kwargs):
        """ Returns the QuantLib.InterestRate representing the spread at date

        This will return the rate with the specific characteristics of the rate as Day Counter, Compounding and
        Frequency. This way when we ask for a rate to be used as spread we can be sure that the rate being used
        is correctly adjusted to the base rate characteristics
        :param date: QuantLib.Date
            Reference date.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :param kwargs:
        :return:QuantLib.InterestRate
            The QuantLib object with the rate and characteristics
        """
        date = to_ql_date(date)
        # Returns None if impossible to obtain a rate helper from this time series
        if self.is_expired(date):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        return ql.InterestRate(rate, self.day_counter, self.compounding, self.frequency)
