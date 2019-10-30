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
from tsfin.constants import INDEX, INDEX_TENOR
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import conditional_vectorize, to_ql_rate_index


class DepositRate(BaseInterestRate):
    """Class to model deposit rates.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries representing the deposit rate.
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries)
        self.is_deposit_rate = True
        self._maturity = None
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        # QuantLib Objects
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor)
        # QuantLib Attributes
        self.calendar = self.index.fixingCalendar()
        self.day_counter = self.index.dayCounter()
        self.business_convention = self.index.businessDayConvention()
        self.fixing_days = self.index.fixingDays()
        self.month_end = self.index.endOfMonth()
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)
        self.helper_spread = ql.SimpleQuote(0)
        self.helper_convexity = ql.SimpleQuote(0)
        # Defined
        self._rate_helper = ql.DepositRateHelper(ql.QuoteHandle(self.helper_rate), self._tenor, self.fixing_days,
                                                 self.calendar, self.business_convention, self.month_end,
                                                 self.day_counter)

    @conditional_vectorize('date')
    def value(*args, **kwargs):
        """Returns zero.
        """
        return 0

    def _get_fixing_maturity_dates(self, start_date, end_date, fixing_at_start_date=False):
        start_date = self.calendar.adjust(start_date, self.business_convention)
        end_date = self.calendar.adjust(end_date, self.business_convention)
        fixing_dates = list()
        maturity_dates = list()
        if fixing_at_start_date:
            fixing_date = start_date
        else:
            fixing_date = self.index.fixingDate(start_date)
        maturity_date = self.index.maturityDate(self.index.valueDate(fixing_date))
        while maturity_date < end_date:
            fixing_dates.append(fixing_date)
            maturity_dates.append(maturity_date)
            fixing_date = self.index.fixingDate(maturity_date)
            maturity_date = self.index.maturityDate(self.index.valueDate(fixing_date))
        fixing_dates.append(fixing_date)
        maturity_dates.append(end_date)
        return fixing_dates, maturity_dates
