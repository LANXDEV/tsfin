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
A class for modelling interest rate swaption.
"""
import QuantLib as ql
import numpy as np
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_calendar, to_ql_day_counter, to_ql_business_convention, to_ql_rate_index, to_ql_date
from tsfin.constants import CALENDAR, DAY_COUNTER, BUSINESS_CONVENTION, SETTLEMENT_DAYS, FIXED_LEG_TENOR, INDEX, \
    MATURITY_TENOR, INDEX_TENOR


class Swaption(BaseInterestRate):

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self._maturity = None
        # Swaption Database Attributes
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.fixed_business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.fixed_calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.fixed_leg_tenor = ql.PeriodParser.parse(self.ts_attributes[FIXED_LEG_TENOR])
        self.fixed_leg_day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.maturity_tenor = ql.PeriodParser.parse(self.ts_attributes[MATURITY_TENOR])
        # QuantLib Objects
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor, self.term_structure)
        # QuantLib Attributes
        self.calendar = ql.JointCalendar(self.fixed_calendar, self.index.fixingCalendar())
        self.day_counter = self.index.dayCounter()
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)

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
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        self.helper_rate.setValue(float(rate))
        return ql.SwaptionHelper(self.maturity_tenor,
                                 self._tenor,
                                 ql.QuoteHandle(self.helper_rate),
                                 self.index,
                                 self.fixed_leg_tenor,
                                 self.fixed_leg_day_counter,
                                 self.index.dayCounter(),
                                 self.term_structure)
