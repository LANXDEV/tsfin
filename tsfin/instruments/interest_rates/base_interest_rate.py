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
from tsfin.constants import TENOR_PERIOD, COMPOUNDING, FREQUENCY, ISSUE_DATE, INDEX_TENOR, CURRENCY
from tsfin.base import Instrument, default_arguments, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, \
    to_ql_compounding, to_ql_currency


DEFAULT_ISSUE_DATE = ql.Date(1, 1, 2000)


class BaseInterestRate(Instrument):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries)
        # Class Flags
        self.is_deposit_rate = False
        self.calculate_convexity = False
        self.telescopic_value_dates = False
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
        self.helper_spread = None
        self.helper_convexity = None
        # Defined
        self._rate_helper = None

    def set_rate_helper(self):

        if self._rate_helper is None:
            self._rate_helper = None

    def link_to_term_structure(self, date, yield_curve):

        """
        :param date: QuantLib.Date
            The yield curve base date
        :param yield_curve: :py:obj:`YieldCurveTimeSeries"
        :return:
        """
        date = to_ql_date(date)
        self.term_structure.linkTo(yield_curve.yield_curve(date=date))

    def set_rate(self, date, rate, spread=None, sigma=None, mean=None, **kwargs):

        if self.is_deposit_rate:
            time = self.day_counter.yearFraction(date, self.maturity(date))
            rate = ql.InterestRate(rate, self.day_counter, self.compounding,
                                   self.frequency).equivalentRate(ql.Simple, ql.Annual, time).rate()
        self.helper_rate.setValue(float(rate))
        if spread is not None:
            self.helper_spread.setValue(float(spread))
        if self.calculate_convexity:
            self.convexity_bias(date=date, future_price=rate, sigma=sigma, mean=mean)

    def _convexity(self, future_price, date, sigma, mean):

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
        if self._maturity is not None:
            return self._maturity
        if self.is_deposit_rate and self.index is not None:
            return self.index.maturityDate(date)
        else:
            return self.calendar.advance(date, self._tenor, self.business_convention, self.month_end)

    def is_expired(self, date, min_future_date=None, max_future_date=None, *args, **kwargs):
        """Check if the deposit rate is expired.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        min_future_date: QuantLib.Date
            The minimum maturity date to be used.
        max_future_date: QuantLib.Date
            The maximum maturity date to be used.
        Returns
        -------
        bool
            Whether the instrument is expired at `date`.
        """
        min_future_date = min_future_date
        max_future_date = max_future_date
        maturity = self.maturity(date=date)
        date = to_ql_date(date)
        if date >= maturity:
            return True
        if min_future_date is None and max_future_date is None:
            return False
        elif min_future_date is None and max_future_date is not None:
            if maturity > max_future_date:
                return True
        elif min_future_date is not None and max_future_date is None:
            if maturity < min_future_date:
                return True
        else:
            if not min_future_date < maturity < max_future_date:
                return True
        return False

    @conditional_vectorize('date')
    def value(*args, **kwargs):
        """Returns zero.
        """
        return 0

    def _get_fixing_maturity_dates(self, start_date, end_date, fixing_at_start_date=False):
        start_date = self.calendar.adjust(start_date, self.business_convention)
        end_date = self.calendar.adjust(end_date, self.business_convention)
        fixing_dates = list()
        maturity_dates = list()
        if fixing_at_start_date:
            fixing_date = start_date
        else:
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

    @default_arguments
    @conditional_vectorize('date')
    def performance(self, start_date=None, date=None, spread=None, fixing_at_start_date=False, **kwargs):
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
        fixing_at_start_date: bool
            Whether to use the start date as the first fixing date or not.

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
        fixing_dates, maturity_dates = self._get_fixing_maturity_dates(start_date, date, fixing_at_start_date)
        fixings = self.timeseries.get_values(index=[to_datetime(fixing_date) for fixing_date in fixing_dates])

        if spread is not None:
            fixings += spread
        return np.prod([ql.InterestRate(
            fixing, self.day_counter, self.compounding, self.frequency).compoundFactor(
                self.index.valueDate(fixing_date), maturity_date, start_date, date)
                    for fixing, fixing_date, maturity_date in zip(fixings, fixing_dates, maturity_dates)
                    if self.index.valueDate(fixing_date) <= maturity_date]) - 1

    def rate_helper(self, date, last_available=True, spread=None, sigma=None, mean=None, **other_args):
        """Helper for yield curve construction.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool, optional
            Whether to use last available quotes if missing data.
        spread: float
            Rate Spread
        sigma: :py:obj:`TimeSeries`
            The timeseries of the sigma, used for convexity calculation
        mean: :py:obj:`TimeSeries`
            The timeseries of the mean, used for convexity calculation

        Returns
        -------
        QuantLib.RateHelper
            Rate helper for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series

        if self.is_expired(date, **other_args):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        self.set_rate_helper()
        date = to_ql_date(date)
        self.set_rate(date=date, rate=rate, spread=spread, sigma=sigma, mean=mean, **other_args)
        return self._rate_helper

    def spread_rate(self, date, last_available=True, **kwargs):

        date = to_ql_date(date)
        if self.is_expired(date):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        return ql.InterestRate(rate, self.day_counter, self.compounding, self.frequency)
