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
import pandas as pd
import QuantLib as ql
from tsio.tools import at_index
from tsfin.constants import CALENDAR
from tsfin.base import Instrument, to_datetime, to_ql_date, to_ql_calendar, ql_holiday_list, conditional_vectorize


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


class Equity(Instrument):
    """ Model for Equities and ETFs.

    :param timeseries: :py:class:`TimeSeries`
        The TimeSeries representing the Equity or ETF.
    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])

    def ts_prices(self, dpdf=False):
        """

        :param dpdf: bool
            If true it will use the adjusted price for calculation.
        :return: pandas.Series
        """
        if dpdf:
            return self.price
        else:
            return self.unadjusted_price

    def ts_returns(self):
        """
        Daily returns from trading days.
        :return: pandas.Series
        """
        price = self.price.ts_values
        start_date = to_ql_date(price.first_valid_index())
        end_date = to_ql_date(price.last_valid_index())
        holiday_list = to_datetime(ql_holiday_list(start_date, end_date, self.calendar))
        price.drop(pd.Index(np.where(price.index.isin(holiday_list))[0]), inplace=True, errors='ignore')
        daily_returns = price / price.shift(1) - 1
        return daily_returns

    def ts_volatility(self, n_days=252):
        """
        Daily rolling volatility.
        :param n_days: int
            The rolling window, will default to 252 if no value is passed.
        :return: pandas.Series
        """

        daily_returns = self.ts_returns()
        vol = daily_returns.rolling(window=n_days, min_periods=1).std()
        vol.dropna(axis='index', inplace=True)
        vol.name = "({})(VOLATILITY)".format(self.ts_name)
        return vol

    @conditional_vectorize('date')
    def cash_to_date(self, start_date, date, tax_adjust=0, *args, **kwargs):
        """
        Cash amount paid by a unit of the instrument between `start_date` and `date`.
        :param start_date: Date-like
            Start date of the range
        :param date: Date-like
            Final date of the range
        :param tax_adjust: float
            The tax value to adjust the dividends received
        :return: float
        """
        start_date = to_ql_date(start_date)
        date = to_ql_date(date)
        if start_date >= date:
            dates = [to_datetime(date)]
        else:
            ql_dates = ql.Schedule(start_date, date, ql.Period(1, ql.Days), self.calendar, ql.Following, ql.Following,
                                   ql.DateGeneration.Forward, False)
            dates = [to_datetime(date) for date in ql_dates]
        dividends = self.dividend_values(date=dates, fill_value=0)
        dividends *= (1 - float(tax_adjust))
        return sum(dividends)

    @conditional_vectorize('quote', 'date')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, tax_adjust=0, dpdf=False,
                    *args, **kwargs):

        """
        Performance of a unit of the instrument.
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
        :param dpdf: bool
            If true it will use the adjusted price for calculation.
        :return: scalar, None
        """
        quotes = self.ts_prices(dpdf=dpdf)

        first_available_date = quotes.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = self.spot_price(date=start_date, dpdf=dpdf)
        if quote is None:
            quote = self.spot_price(date=date, dpdf=dpdf)
        if date < start_date:
            return np.nan

        start_value = start_quote
        value = quote
        if dpdf:
            dividends = 0
        else:
            start_date = self.calendar.advance(to_ql_date(start_date), ql.Period(1, ql.Days), ql.Following)
            dividends = self.cash_to_date(start_date=start_date, date=date, tax_adjust=tax_adjust)

        return (value + dividends) / start_value - 1

    @conditional_vectorize('date')
    def spot_price(self, date, last_available=True, fill_value=np.nan, dpdf=False):
        """
        Return the daily series of unadjusted price at date(s).
        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :param dpdf: bool
            If true it will use the adjusted price for calculation.
        :return: pandas.Series
        """
        date = to_datetime(date)
        prices = self.ts_prices(dpdf=dpdf)
        return prices.get_values(index=date, last_available=last_available, fill_value=fill_value)

    @conditional_vectorize('date')
    def dividend_values(self, date, last_available=True, fill_value=np.nan):
        """
        Daily dividend values at date.
        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :return: pandas.Series
        """
        date = to_datetime(date)
        try:
            return self.dividends.get_values(index=date, last_available=last_available, fill_value=fill_value)
        except KeyError:
            return 0

    @conditional_vectorize('date')
    def dividend_yield(self, date, last_available=True, fill_value=np.nan):
        """
        12 month dividend yield at date(s).
        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :return: pandas.Series
        """
        date = to_datetime(date)
        try:
            return self.eqy_dvd_yld_12m.get_values(index=date, last_available=last_available, fill_value=fill_value)
        except KeyError:
            return 0

    @conditional_vectorize('date')
    def volatility(self, date, last_available=True, fill_value=np.nan, n_days=None, annual_factor=252):
        """
        The converted volatility value series at date.
        :param date: Date-like
            Date or dates to be return the dividend values.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing.
        :param fill_value: scalar
            Default value in case `date` can't be found.
        :param n_days: int
            Rolling window for the volatility calculation.
        :param annual_factor: int, default 252
            The number of days used for period transformation, default is 252, or 1 year.
        :return: pandas.Series
        """
        date = to_datetime(date)
        vol = at_index(df=self.ts_volatility(n_days=n_days), index=date, last_available=last_available,
                       fill_value=fill_value)

        return vol*np.sqrt(annual_factor)
