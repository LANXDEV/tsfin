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
Equity class, to represent Equities and Exchange Traded Funds.
"""
import numpy as np
import QuantLib as ql
from tsfin.constants import CALENDAR, UNDERLYING_INSTRUMENT, TICKER, QUOTES, UNADJUSTED_PRICE, DIVIDEND_YIELD, DIVIDENDS
from tsfin.base import Instrument, to_datetime, to_ql_date, to_ql_calendar, conditional_vectorize


class Equity(Instrument):
    """ Model for Equities and ETFs.

    :param timeseries: :py:class:`TimeSeries`
        The TimeSeries representing the Equity or ETF.
    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """

    def __init__(self, timeseries):
        super().__init__(timeseries=timeseries)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        try:
            self.underlying_name = self.ts_attributes[UNDERLYING_INSTRUMENT]
        except KeyError:
            self.underlying_name = self.ts_attributes[TICKER]

    def spot_prices(self, dividend_adjusted=False):
        """ Return a the dividend adjusted or unadjusted price series.

        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :return pandas.Series
        """
        if dividend_adjusted:
            return getattr(self.timeseries, QUOTES)
        else:
            try:
                return getattr(self.timeseries, UNADJUSTED_PRICE)
            except AttributeError:
                return getattr(self.timeseries, QUOTES)

    def _dividends_to_date(self, start_date, date, *args, **kwargs):
        """ Cash amount paid by a unit of the instrument between `start_date` and `date`.

        :param start_date: Date-like
            Start date of the range
        :param date: Date-like
            Final date of the range
        :return: pandas.Series
        """
        start_date = to_datetime(start_date)
        date = to_datetime(date)
        ts_dividends = getattr(self.timeseries, DIVIDENDS).ts_values
        if start_date >= date:
            start_date = date
        ts_dividends = ts_dividends[ts_dividends != 0]
        if date == start_date:
            return ts_dividends[(ts_dividends.index == date)]
        else:
            return ts_dividends[(ts_dividends.index >= start_date) & (ts_dividends.index <= date)]

    def security(self, date, last_available=False, dividend_adjusted=False, *args, **kwargs):
        """ Return the QuantLib Object representing a Stock

        :param date: date-like, optional
            The date.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing
        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :param args:
        :param kwargs:
        :return: QuantLib.Stock
        """
        price = self.spot_prices(dividend_adjusted=dividend_adjusted)(index=date, last_available=last_available)
        return ql.Stock(ql.QuoteHandle(ql.SimpleQuote(price)))

    @conditional_vectorize('start_date', 'date')
    def cash_flow_to_date(self, start_date, date, tax_adjust=0, *args, **kwargs):
        """ Cash amount paid by a unit of the instrument between `start_date` and `date`.

        :param start_date: Date-like
            Start date of the range
        :param date: Date-like
            Final date of the range
        :param tax_adjust: float
            The tax value to adjust the dividends received
        :return list of tuples with (date, date, value)
            Return the ex-date, pay-date and dividend value
        """
        ts_dividends = self._dividends_to_date(start_date=start_date, date=date, *args, **kwargs)
        return list((to_ql_date(date), to_ql_date(to_datetime(value[0])), value[1]*(1-float(tax_adjust)))
                    for date, value in ts_dividends.iteritems())

    @conditional_vectorize('start_date', 'date')
    def cash_to_date(self, start_date, date, tax_adjust=0, *args, **kwargs):
        """ Cash amount paid by a unit of the instrument between `start_date` and `date`.

        :param start_date: Date-like
            Start date of the range
        :param date: Date-like
            Final date of the range
        :param tax_adjust: float
            The tax value to adjust the dividends received
        :return float
        """
        ts_dividends = self._dividends_to_date(start_date=start_date, date=date, *args, **kwargs)
        dividends = 0
        for date, value in ts_dividends.iteritems():
            dividends += value[1]
        return dividends*(1-float(tax_adjust))

    @conditional_vectorize('date')
    def value(self, date, last_available=False, dividend_adjusted=False, *args, **kwargs):
        """Try to deduce dirty value for a unit of the time series (as a financial instrument).

        :param date: date-like, optional
            The date.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :return scalar, None
            The unit dirty value of the instrument.
        """
        quotes = self.spot_prices(dividend_adjusted=dividend_adjusted)
        date = to_datetime(date)
        if last_available:
            return quotes(index=date, last_available=last_available)
        if date > quotes.ts_values.last_valid_index():
            return np.nan
        if date < quotes.ts_values.first_valid_index():
            return np.nan
        return quotes(index=date)

    @conditional_vectorize('date')
    def risk_value(self, date, last_available=False, dividend_adjusted=False, *args, **kwargs):
        """Try to deduce risk value for a unit of the time series (as a financial instrument).

        :param date: date-like, optional
            The date.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :return scalar, None
            The unit risk value of the instrument.
        """
        return self.value(date=date, last_available=last_available, dividend_adjusted=dividend_adjusted, *args,
                          **kwargs)

    @conditional_vectorize('quote', 'date')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, tax_adjust=0,
                    dividend_adjusted=False, *args, **kwargs):

        """  Performance of a unit of the instrument.

        :param start_date: Date-like
            Start date of the range
        :param start_quote: float, optional
            The quote of the instrument in `start_date`. Defaults to the quote in `start_date`.
        :param date: Date-like
            Final date of the range
        :param quote: float, optional
            The quote of the instrument at `date`. Defaults to the quote at `date`.
        :param tax_adjust: float
            The tax value to adjust the dividends received
        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :return scalar, None
        """
        quotes = self.spot_prices(dividend_adjusted=dividend_adjusted)

        first_available_date = quotes.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = quotes(index=start_date)
        if quote is None:
            quote = quotes(index=date)
        if date < start_date:
            return np.nan

        if dividend_adjusted:
            dividends = 0
        else:
            start_date = self.calendar.advance(to_ql_date(start_date), ql.Period(1, ql.Days), ql.Following)
            dividends = self.cash_to_date(start_date=start_date, date=date, tax_adjust=tax_adjust)

        return (quote + dividends) / start_quote - 1

    @conditional_vectorize('date')
    def spot_price(self, date, last_available=True, fill_value=np.nan, dividend_adjusted=False):
        """ Return the daily series of unadjusted price at date(s).

        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :param dividend_adjusted: bool
            If true it will use the adjusted price for calculation.
        :return pandas.Series
        """
        date = to_datetime(date)
        prices = self.spot_prices(dividend_adjusted=dividend_adjusted)
        return prices(index=date, last_available=last_available, fill_value=fill_value)

    @conditional_vectorize('date')
    def dividend_yield(self, date, last_available=True, fill_value=np.nan):
        """ 12 month dividend yield at date(s).

        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :return pandas.Series
        """
        date = to_datetime(date)
        try:
            dividend_yield = getattr(self.timeseries, DIVIDEND_YIELD)
            return dividend_yield(index=date, last_available=last_available, fill_value=fill_value)
        except KeyError:
            return 0

    @conditional_vectorize('date')
    def volatility(self, date, n_days=252, annual_factor=252, log_returns=False):
        """ The converted volatility value series at date.

        :param date: Date-like
            Date or dates to be return the dividend values.
        :param n_days: int
            Rolling window for the volatility calculation.
        :param annual_factor: int, default 252
            The number of days used for period transformation, default is 252, or 1 year.
        :param log_returns: bool
            If True it will use the log returns of prices
        :return pandas.Series
        """
        date = to_datetime(date)
        prices = self.price()
        if log_returns:
            returns = np.log(prices[prices.index <= date].tail(n_days)).diff()
        else:
            returns = prices[prices.index <= date].tail(n_days).pct_change()
        return np.std(returns)*np.sqrt(annual_factor)
