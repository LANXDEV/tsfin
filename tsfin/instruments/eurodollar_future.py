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
from tsfin.constants import MATURITY_DATE, FUTURE_CONTRACT_SIZE, TICK_SIZE, TICK_VALUE, TERM_NUMBER, \
    TERM_PERIOD, SETTLEMENT_DAYS
from tsfin.instruments.depositrate import DepositRate
from tsfin.base import conditional_vectorize, to_datetime, to_ql_date, to_ql_time_unit


class EurodollarFuture(DepositRate):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self._maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.contract_size = float(self.ts_attributes[FUTURE_CONTRACT_SIZE])
        self.tick_size = float(self.ts_attributes[TICK_SIZE])
        self.tick_value = float(self.ts_attributes[TICK_VALUE])
        self.term_number = int(self.ts_attributes[TERM_NUMBER])
        self.term_period = to_ql_time_unit(self.ts_attributes[TERM_PERIOD])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.convexity_adjustment = dict()

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
        initial_t = self.day_counter.yearFraction(date, self._maturity)
        interest_maturity_date = self.calendar.advance(self._maturity, ql.Period(self.term_number, self.term_period),
                                                       self.business_convention, self.month_end)
        final_t = self.day_counter(date, interest_maturity_date)
        convexity_bias = ql.HullWhite.convexityBias(future_price, initial_t, final_t, sigma_value, mean_value)
        return convexity_bias

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
        price = self.price.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(price):
            return None
        date = to_ql_date(date)
        final_price = ql.SimpleQuote(price)
        sigma = other_args.get('sigma', None)
        mean = other_args.get('mean', None)
        convexity = ql.SimpleQuote(self.convexity_bias(date, future_price=price, sigma=sigma, mean=mean,
                                                       last_available=last_available))
        return ql.FuturesRateHelper(ql.QuoteHandle(final_price), self.maturity(date), self.term_number,
                                    self.term_period, self.calendar, self.business_convention, self.month_end,
                                    self.day_counter, ql.QuoteHandle(convexity))
