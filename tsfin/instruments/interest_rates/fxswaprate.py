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
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_calendar, to_ql_day_counter, to_ql_business_convention, to_ql_date
from tsfin.constants import CALENDAR, DAY_COUNTER, BUSINESS_CONVENTION, TENOR_PERIOD, FIXING_DAYS, COUNTRY, \
    BASE_CURRENCY, BASE_CALENDAR


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
        self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
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
