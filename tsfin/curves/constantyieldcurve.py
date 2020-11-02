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
ConstantYieldCurve, a yield curve that will be based on a pre-defined list of forward rates and dates
"""

import QuantLib as ql
from tsfin.curves.yieldcurve import SimpleYieldCurve
from tsfin.base import to_list, to_ql_date


class ConstantYieldCurve(SimpleYieldCurve):

    def __init__(self, forward_rates, forward_dates, calendar, day_counter, compounding,
                 business_convention=ql.Following, enable_extrapolation=True, ignore_errors=True,
                 frozen_curve_interpolation_type='monotonic_cubic_zero'):
        super().__init__(calendar=calendar, day_counter=day_counter, enable_extrapolation=enable_extrapolation,
                         ignore_errors=ignore_errors)
        self.business_convention = business_convention
        self.compounding = compounding
        self.forward_dates = [
            self.calendar.adjust(forward_date, self.business_convention) for forward_date in forward_dates]
        if self.compounding == ql.Continuous:
            self.forward_rates = forward_rates
        else:
            self.forward_rates = list()
            for t, (forward_rate, forward_date) in enumerate(zip(forward_rates, self.forward_dates)):
                interest_rate = ql.InterestRate(forward_rate, self.day_counter, self.compounding, ql.Annual)
                if t ==0:
                    time = 0
                else:
                    time = self.day_counter.yearFraction(self.forward_dates[t-1], forward_date)
                rate = interest_rate.equivalentRate(ql.Continuous, ql.NoFrequency, time).rate()
                self.forward_rates.append(rate)
        self.forward_curve = ql.ForwardCurve(self.forward_dates, self.forward_rates, self.day_counter, self.calendar)
        self.frozen_curve_interpolation_type = frozen_curve_interpolation_type

    def update_curves(self, dates):

        dates = to_list(dates)

        for date in dates:
            date = to_ql_date(date)
            ql.Settings.instance().evaluationDate = date
            yield_curve = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(self.forward_curve), date)
            if self.enable_extrapolation:
                yield_curve.enableExtrapolation()
            self.yield_curves[date] = yield_curve
