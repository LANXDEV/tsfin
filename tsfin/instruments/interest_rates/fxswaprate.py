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
A class for modelling FX swaps.
"""
import QuantLib as ql
import numpy as np
from tsio.constants import COMPONENTS
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_calendar, to_ql_day_counter, to_ql_business_convention, to_ql_date, to_datetime, \
    conditional_vectorize, Instrument
from tsfin.constants import CALENDAR, DAY_COUNTER, BUSINESS_CONVENTION, TENOR_PERIOD, FIXING_DAYS, COUNTRY, \
    BASE_CURRENCY, BASE_CALENDAR, MATURITY_DATE, QUANTITY, TRADE_PRICE, ISSUE_DATE


class FxSwapRate(BaseInterestRate):
    """ Model for rolling interest rate swap rates (fixed tenor, like the ones quoted in Bloomberg).

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, currency_ts):
        super().__init__(timeseries)
        self._maturity = None
        # Swap Database Attributes
        self.currency_ts = currency_ts
        try:
            self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        except AttributeError:
            self._tenor = None
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])
        self.base_currency = self.ts_attributes[BASE_CURRENCY]
        self.base_calendar = to_ql_calendar(self.ts_attributes[BASE_CALENDAR])
        self.country = self.ts_attributes[COUNTRY]
        self.month_end = False
        # Rate Helper
        self.currency_spot_rate = ql.SimpleQuote(1)
        self.currency_spot_handle = ql.RelinkableQuoteHandle(self.currency_spot_rate)
        self.helper_rate = ql.SimpleQuote(0)
        self.helper_spread = ql.SimpleQuote(0)
        self.helper_convexity = ql.SimpleQuote(0)

    def is_expired(self, date, *args, **kwargs):
        """ Returns False.

        Parameters
        ----------
        date: QuantLib.Date
            The date.

        Returns
        -------
        bool
            Always False.
        """
        return False

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
        fx_spot = self.currency_ts.price.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(fx_spot):
            return None
        self.currency_spot_rate.setValue(fx_spot)
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)/10000
        if np.isnan(rate):
            return None
        self.helper_rate.setValue(float(rate))
        return ql.FxSwapRateHelper(ql.QuoteHandle(self.helper_rate),
                                   self.currency_spot_handle,
                                   self._tenor,
                                   self.fixing_days,
                                   self.calendar,
                                   self.business_convention,
                                   self.month_end,
                                   True,
                                   self.term_structure)


class NonDeliverableForward(Instrument):
    """
    Non Deliverable Forward
    Class still being tested
    """

    def __init__(self, timeseries, currency_ts, fx_swap_curve=None):
        super().__init__(timeseries=timeseries)
        self.currency_ts = currency_ts
        self.fx_swap_curve = fx_swap_curve
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])
        self.base_currency = self.ts_attributes[BASE_CURRENCY]
        self.base_calendar = to_ql_calendar(self.ts_attributes[BASE_CALENDAR])
        self.country = self.ts_attributes[COUNTRY]
        self._maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.issue_date = to_ql_date(to_datetime(self.ts_attributes[ISSUE_DATE]))
        self.ts_quantity = self.ts_attributes[COMPONENTS][QUANTITY]
        self.ts_trade_price = self.ts_attributes[COMPONENTS][TRADE_PRICE]

    def set_fx_swap_curve(self, fx_swap_curve):

        self.fx_swap_curve = fx_swap_curve

    @conditional_vectorize('date')
    def maturity(self, date, *args, **kwargs):

        return self._maturity

    @conditional_vectorize('date')
    def value(self, date, fx_swap_curve=None, *args, **kwargs):

        date = to_ql_date(date)

        maturity = self.maturity(date=date)
        if date > maturity:
            return 0

        if fx_swap_curve is None:
            fx_swap_curve = self.fx_swap_curve

        fx_rate = fx_swap_curve.implied_fx_rate_to_date(date=date, to_date=self.maturity(date=date))
        filtered_quantity = self.ts_quantity.ts_values[self.ts_quantity.ts_values.index <= to_datetime(date)].values
        filtered_trade_price = self.ts_trade_price.ts_values[
            self.ts_trade_price.ts_values.index <= to_datetime(date)].values

        total_value = 0
        for quantity, trade_price in zip(filtered_quantity, filtered_trade_price):
            total_value += quantity/fx_rate - quantity/trade_price

        return total_value

    @conditional_vectorize('date', 'quote')
    def risk_value(self, date, quote=None, fx_swap_curve=None, *args, **kwargs):

        date = to_ql_date(date)

        maturity = self.maturity(date=date)
        if date > maturity:
            return 0

        if fx_swap_curve is None and quote is None:
            fx_swap_curve = self.fx_swap_curve

        if quote is None and fx_swap_curve is not None:
            quote = fx_swap_curve.implied_fx_rate_to_date(date=date, to_date=self.maturity(date=date))

        filtered_quantity = self.ts_quantity.ts_values[self.ts_quantity.ts_values.index <= to_datetime(date)].values
        filtered_trade_price = self.ts_trade_price.ts_values[
            self.ts_trade_price.ts_values.index <= to_datetime(date)].values

        total_value = 0
        base_value = 0

        for quantity, trade_price in zip(filtered_quantity, filtered_trade_price):
            base_value += -quantity/trade_price
            total_value += quantity/quote

        if np.round(total_value, 2) == 0:
            return 0
        elif np.round(base_value, 2) == 0:
            return 0
        else:
            return total_value / base_value
