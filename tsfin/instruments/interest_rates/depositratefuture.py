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
import QuantLib as ql
from tsfin.constants import CALENDAR, MATURITY_DATE, BUSINESS_CONVENTION, DAY_COUNTER, FIXING_DAYS
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_business_convention, to_ql_calendar, to_ql_day_counter, to_ql_date


class DepositRateFuture(BaseInterestRate):
    def __init__(self, timeseries, is_deposit_rate=True):
        super().__init__(timeseries, is_deposit_rate=is_deposit_rate)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self._maturity = to_ql_date(self.ts_attributes[MATURITY_DATE])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])
        self.month_end = False
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)
        self.helper_spread = ql.SimpleQuote(0)
        self.helper_convexity = ql.SimpleQuote(0)

    def set_rate_helper(self):
        if self._rate_helper is None:
            self._rate_helper = ql.DepositRateHelper(ql.QuoteHandle(self.helper_rate), self._tenor, self.fixing_days,
                                                     self.calendar, self.business_convention, self.month_end,
                                                     self.day_counter)

    def _get_fixing_maturity_dates(self, start_date, end_date, fixing_at_start_date=False):
        start_date = self.calendar.adjust(start_date, self.business_convention)
        end_date = self.calendar.adjust(end_date, self.business_convention)
        fixing_dates = list()
        maturity_dates = list()
        if fixing_at_start_date:
            fixing_date = start_date
        else:
            fixing_date = self.calendar.advance(start_date, -self.fixing_days, ql.Days, self.business_convention,
                                                self.month_end)
        value_date = self.calendar.advance(fixing_date, self.fixing_days, ql.Days, self.business_convention,
                                           self.month_end)
        maturity_date = self.calendar.advance(value_date, self._tenor, self.business_convention, self.month_end)
        while maturity_date < end_date:
            fixing_dates.append(fixing_date)
            maturity_dates.append(maturity_date)
            fixing_date = self.calendar.advance(maturity_date, -self.fixing_days, ql.Days, self.business_convention,
                                                self.month_end)
            maturity_date = self.calendar.advance(maturity_date, self._tenor, self.business_convention, self.month_end)
        fixing_dates.append(fixing_date)
        maturity_dates.append(end_date)
        return fixing_dates, maturity_dates
