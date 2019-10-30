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
Helper Classes for different Quantlib functions
"""

import QuantLib as ql
from tsfin.constants import TENOR_PERIOD, MATURITY_DATE, DAY_COUNTER, COMPOUNDING, FREQUENCY
from tsfin.base import to_ql_date, to_ql_day_counter, to_ql_calendar, to_ql_frequency, to_ql_compounding, \
    conditional_vectorize


class SpreadHandle:

    def __init__(self, ts_collection=None):
        self.ts_collection = ts_collection
        self.spreads = dict()

    @conditional_vectorize('date')
    def update_spread_from_timeseries(self, date, last_available=True):

        date = to_ql_date(date)
        self.spreads[date] = dict()
        for ts in self.ts_collection:
            spread = ts.get_values(date, last_available=last_available)
            try:
                spread_date_or_tenor = ts.calendar.advance(date, ql.PeriodParser.parse(ts.ts_attributes[TENOR_PERIOD]))
            except AttributeError:
                spread_date_or_tenor = ts.calendar.adjust(to_ql_date(ts.ts_attributes[MATURITY_DATE]))
            day_counter = to_ql_day_counter(ts.ts_attributes[DAY_COUNTER])
            compounding = to_ql_compounding(ts.ts_attributes[COMPOUNDING])
            frequency = to_ql_frequency(ts.ts_attributes[FREQUENCY])
            self.spreads[date][spread_date_or_tenor] = ql.InterestRate(spread, day_counter, compounding, frequency)

    @conditional_vectorize('date')
    def spread_handle(self, date, last_available=True):

        date = to_ql_date(date)
        try:
            return self.spreads[date]

        except KeyError:
            self.update_spread_from_timeseries(date=date, last_available=last_available)
            return self.spreads[date]

    @conditional_vectorize('date', 'spread', 'tenor_date')
    def update_spread_from_value(self, date, spread, tenor_date, day_counter, frequency, compounding):

        date = to_ql_date(date)
        tenor_date = to_ql_date(tenor_date)
        self.spreads[date] = dict()
        self.spreads[date][tenor_date] = ql.InterestRate(spread, day_counter, compounding, frequency)
        return self
