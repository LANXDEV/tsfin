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
Custom Class wrapper for QuantLib Schedule Class
TODO: More testing needed
"""
import QuantLib as ql


def to_bus_day_name(bus_day_convention):
    if bus_day_convention == ql.Following:
        return 'Following'
    elif bus_day_convention == ql.ModifiedFollowing:
        return 'Modified Following'
    elif bus_day_convention == ql.Preceding:
        return 'Preceding'
    elif bus_day_convention == ql.ModifiedPreceding:
        return 'Modified Preceding'
    elif bus_day_convention == ql.HalfMonthModifiedFollowing:
        return 'Half Month Modified Following'
    elif bus_day_convention == ql.Unadjusted:
        return 'Unadjusted'
    else:
        return None


def to_date_generation_name(date_generation):
    if date_generation == ql.DateGeneration.Backward:
        return 'Backward'
    elif date_generation == ql.DateGeneration.Forward:
        return 'Forward'
    elif date_generation == ql.DateGeneration.Zero:
        return 'Zero'
    elif date_generation == ql.DateGeneration.ThirdWednesday:
        return 'ThirdWednesday'
    elif date_generation == ql.DateGeneration.Twentieth:
        return 'Twentieth'
    elif date_generation == ql.DateGeneration.TwentiethIMM:
        return 'TwentiethIMM'
    elif date_generation == ql.DateGeneration.CDS2015:
        return 'CDS2015'
    elif date_generation == ql.DateGeneration.CDS:
        return 'CDS'
    else:
        return None


class ScheduleIterator:

    def __init__(self, schedule):
        self._schedule = schedule
        self._index = 0
        self._index_len = len(self._schedule)

    def __next__(self):
        if self._index < self._index_len:
            result = self._schedule.__getitem__(i=self._index)
            self._index += 1
            return result
        else:
            raise StopIteration


class Schedule(ql.Schedule):
    """QuantLib Schedule class with some additions

    Add iter method and a more readable repr method.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.dates = args[0]
        self.repr_constructor = False
        if isinstance(args[0], ql.Date) and isinstance(args[1], ql.Date):
            self.repr_constructor = True

    def __iter__(self):
        return ScheduleIterator(self)

    def __repr__(self):
        if self.repr_constructor:
            return "{}({!r})".format(self.__class__.__name__,
                                     [self.startDate(), self.endDate(), self.tenor(), self.calendar().name(),
                                      to_bus_day_name(self.businessDayConvention()),
                                      to_bus_day_name(self.terminationDateBusinessDayConvention()),
                                      to_date_generation_name(self.rule()),
                                      'End Of Month={}'.format(self.endOfMonth())])
        else:
            return "{}({!r})".format(self.__class__.__name__,
                                     ["{}({!r})".format('Date Vector',
                                                        [self.dates[0], self.dates[-1],
                                                         'len={}'.format(len(self.dates))]),
                                      self.calendar().name(), to_bus_day_name(self.businessDayConvention()),
                                      to_bus_day_name(self.terminationDateBusinessDayConvention()), self.tenor(),
                                      to_date_generation_name(self.rule()),
                                      'End Of Month={}'.format(self.endOfMonth())])
