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
from tsfin.constants import MATURITY_DATE, FUTURE_CONTRACT_SIZE, TICK_SIZE, TICK_VALUE, TERM_NUMBER, \
    TERM_PERIOD, SETTLEMENT_DAYS
from tsfin.base.instrument import default_arguments
from tsfin.instruments.depositrate import DepositRate
from tsfin.base import Instrument, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, \
    to_ql_business_convention, to_ql_calendar, to_ql_compounding, to_ql_day_counter, to_ql_time_unit


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

    def set_convexity_adjustment(self, date, convexity_adjustment=None):

        date = to_ql_date(date)
        if convexity_adjustment is None:
            convexity_adjustment = ql.SimpleQuote(0)
        else:
            convexity_adjustment = ql.SimpleQuote(convexity_adjustment)

        self.convexity_adjustment[date] = convexity_adjustment

        return self.convexity_adjustment[date]

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
        price = self.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(price):
            return None
        date = to_ql_date(date)
        final_price = ql.SimpleQuote(price)
        convexity = self.set_convexity_adjustment(date)
        return ql.FuturesRateHelper(ql.QuoteHandle(final_price), self.maturity(date), self.term_number,
                                    self.term_period, self.calendar, self.business_convention, self.month_end,
                                    self.day_counter, ql.QuoteHandle(convexity))
