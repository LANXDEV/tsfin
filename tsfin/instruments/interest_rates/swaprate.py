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
A class for modelling interest rate swaps.
"""
import QuantLib as ql
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_calendar, to_ql_day_counter, to_ql_business_convention, to_ql_rate_index
from tsfin.constants import CALENDAR, DAY_COUNTER, BUSINESS_CONVENTION, SETTLEMENT_DAYS, FIXED_LEG_TENOR, INDEX, \
    INDEX_TENOR


class SwapRate(BaseInterestRate):
    """ Model for rolling interest rate swap rates (fixed tenor, like the ones quoted in Bloomberg).

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self._maturity = None
        # Swap Database Attributes
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.fixed_business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.fixed_calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.fixed_leg_tenor = ql.PeriodParser.parse(self.ts_attributes[FIXED_LEG_TENOR])
        self.fixed_leg_day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        # QuantLib Objects
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor)
        # QuantLib Attributes
        self.calendar = ql.JointCalendar(self.fixed_calendar, self.index.fixingCalendar())
        self.day_counter = self.index.dayCounter()
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        # Swap Index
        self.swap_index = ql.SwapIndex(self.ts_name, self._tenor, self.settlement_days, self.currency, self.calendar,
                                       self.fixed_leg_tenor, self.fixed_business_convention, self.fixed_leg_day_counter,
                                       self.index)
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)
        self.helper_spread = ql.SimpleQuote(0)
        self.helper_convexity = ql.SimpleQuote(0)

    def set_rate_helper(self):

        if self._rate_helper is None:
            self._rate_helper = ql.SwapRateHelper(ql.QuoteHandle(self.helper_rate), self.swap_index,
                                                  ql.QuoteHandle(self.helper_spread))

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
