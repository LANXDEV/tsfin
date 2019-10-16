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
from tsfin.base.qlconverters import to_ql_date, to_ql_day_counter, to_ql_calendar, to_ql_frequency, to_ql_compounding
from tsfin.base.basetools import conditional_vectorize


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
                spread_date_or_tenor = date + ql.Period(ts.ts_attributes[TENOR_PERIOD])
            except AttributeError:
                spread_date_or_tenor = to_ql_date(ts.ts_attributes[MATURITY_DATE])
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
    def update_spread_from_value(self, date, spread, tenor_date):

        date = to_ql_date(date)
        self.spreads[date] = dict()
        self.spreads[date][tenor_date] = ql.SimpleQuote(spread)
        return self


# class for hosting schedule-related information (dates, times)
class TimeGrid:

    def __init__(self, start_date, end_date, tenor, calendar, day_counter):
        self.start_date = to_ql_date(start_date)
        self.end_date = to_ql_date(end_date)
        self.tenor = ql.PeriodParser.parse(tenor)
        self.calendar = to_ql_calendar(calendar)
        self.day_counter = to_ql_day_counter(day_counter)
        self.schedule = ql.Schedule(self.start_date, self.end_date, self.tenor, self.calendar, ql.Unadjusted,
                                    ql.Unadjusted, ql.DateGeneration.Forward, False)

    def get_times(self):
        # get list of scheduled times
        return [self.day_counter.yearFraction(self.schedule.startDate(), date) for date in self.schedule]

    def get_dates(self):
        # get list of scheduled dates
        return [date for date in self.schedule]

    def get_maturity(self):
        # get maturity in time units
        return self.day_counter.yearFraction(self.schedule.startDate(), self.schedule.endDate())

    def get_steps(self):
        # get number of steps in schedule
        return self.get_size() - 1

    def get_size(self):
        # get total number of items in schedule
        return len(self.schedule)

    def get_time_grid(self):
        # get QuantLib TimeGrid object, constructed by using list of scheduled times
        return ql.TimeGrid(self.get_times(), self.get_size())

    def get_dt(self):
        # get constant time step
        return self.get_maturity() / self.get_steps()
