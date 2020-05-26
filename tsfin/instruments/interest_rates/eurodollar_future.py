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
EurodollarFuture class, to represent eurodollar futures.
"""
import numpy as np
import QuantLib as ql
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_rate_index, to_ql_date, to_datetime, to_ql_time_unit, conditional_vectorize
from tsfin.constants import SETTLEMENT_DAYS, INDEX, CONTRACT_SIZE, TICK_SIZE, TICK_VALUE, TERM_NUMBER, TERM_PERIOD,\
    INDEX_TENOR, LAST_DELIVERY


class EurodollarFuture(BaseInterestRate):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, calculate_convexity=True):
        super().__init__(timeseries, calculate_convexity=calculate_convexity)
        # Database Attributes
        self.last_delivery = to_ql_date(to_datetime(self.ts_attributes[LAST_DELIVERY]))
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.contract_size = float(self.ts_attributes[CONTRACT_SIZE])
        self.tick_size = float(self.ts_attributes[TICK_SIZE])
        self.tick_value = float(self.ts_attributes[TICK_VALUE])
        self.term_number = int(self.ts_attributes[TERM_NUMBER])
        self.term_period = to_ql_time_unit(self.ts_attributes[TERM_PERIOD])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        # QuantLib Objects
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor)
        # QuantLib Attributes
        self._maturity = self.interest_start_date = self.index.valueDate(self.last_delivery)
        self.calendar = self.index.fixingCalendar()
        self.day_counter = self.index.dayCounter()
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        self.interest_maturity_date = self.index.maturityDate(self.interest_start_date)
        self.futures_type = ql.Futures.IMM
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(100)
        self.helper_convexity = ql.SimpleQuote(0)

    def is_expired(self, date, min_future_tenor=None, max_future_tenor=None, *args, **kwargs):
        """Check if the deposit rate is expired.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        min_future_tenor: str
            The minimum maturity date to be used.
        max_future_tenor: str
            The maximum maturity date to be used.
        Returns
        -------
        bool
            Whether the instrument is expired at `date`.
        """

        date = to_ql_date(date)
        maturity = self.maturity(date=date)

        if min_future_tenor is not None:
            min_future_date = self.calendar.advance(date, ql.PeriodParser.parse(min_future_tenor))
        else:
            min_future_date = date
        if max_future_tenor is not None:
            max_future_date = self.calendar.advance(date, ql.PeriodParser.parse(max_future_tenor))
        else:
            max_future_date = self.calendar.advance(maturity, 1, ql.Days)

        if date >= maturity:
            return True
        if not min_future_date < maturity < max_future_date:
            return True
        return False

    @conditional_vectorize('date', 'start_quote', 'quote')
    def value(self, date, start_quote, quote, *args, **kwargs):
        """Returns zero.
        """

        price_change = quote - start_quote
        margin_value = price_change/self.tick_size*self.tick_value

        return margin_value

    @conditional_vectorize('start_date', 'start_quote', 'date', 'quote')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, *args, **kwargs):
        """
        Performance of investment in the interest rate, taking tenor into account.

        If the period between start_date and date is larger the the deposit rate's tenor, considers the investment
        is rolled at the prevailing rate at each maturity.

        Parameters
        ----------
        start_date: datetime-like, optional
            The starting date of the period.
        date: datetime-like, optional
            The ending date of the period.
        start_quote: float, optional
            The quote of the instrument in `start_date`. Defaults to the quote in `start_date`.
        quote
            The quote of the instrument in `date`. Defaults to the quote in `date`.

        Returns
        -------
        scalar, None
            Performance of a unit of the instrument.
        """
        quotes = self.quotes
        first_available_date = quotes.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = quotes.get_values(index=start_date)
        if date < start_date:
            return np.nan
        start_value = self.value(quote=start_quote, date=start_date)
        value = self.value(quote=quote, date=date)

        return (value / start_value) - 1

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
        if self.is_expired(date, **other_args):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        date = to_ql_date(date)
        self.helper_rate.setValue(float(rate))
        self.convexity_bias(date=date, future_price=rate, sigma=sigma, mean=mean)
        return ql.FuturesRateHelper(ql.QuoteHandle(self.helper_rate),
                                    self.interest_start_date,
                                    self.index,
                                    ql.QuoteHandle(self.helper_convexity),
                                    self.futures_type)
