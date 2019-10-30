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
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_calendar, to_ql_day_counter, to_ql_business_convention, to_ql_rate_index
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
        self.term_structure = ql.RelinkableYieldTermStructureHandle()
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor)
        # QuantLib Attributes
        self.calendar = ql.JointCalendar(self.fixed_calendar, self.index.fixingCalendar())
        self.day_counter = self.index.dayCounter()
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)
        self.helper_spread = ql.SimpleQuote(0)
        self.helper_convexity = ql.SimpleQuote(0)
        self._rate_helper = ql.SwaptionHelper(self.maturity_tenor, self._tenor, ql.QuoteHandle(self.helper_rate),
                                              self.index, self.fixed_leg_tenor, self.fixed_leg_day_counter,
                                              self.index.dayCounter(), self.term_structure)

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
