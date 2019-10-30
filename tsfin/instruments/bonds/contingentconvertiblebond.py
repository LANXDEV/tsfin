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

import QuantLib as ql
from tsfin.base import to_datetime, to_list, to_ql_date
from tsfin.instruments.bonds._basebond import create_schedule_for_component
from tsfin.instruments.bonds.floatingratebond import FloatingRateBond
from tsfin.constants import COUPON_TYPE_RESET_DATE, CALLED_DATE, COUPONS


def create_call_component(call_date, call_price, main_bond_schedule, calendar, business_convention, tenor,
                          date_generation, month_end, settlement_days, face_amount, day_counter, issue_date,
                          index, fixed_coupon_len, fixed_coupon, floating_coupon, caps, floors, in_arrears):
    """Create a QuantLib fixed-rate or zero-coupon bond instance for the ``components`` dict of callable bond.

    Parameters
    ----------
    call_date: QuantLib.Date
        Maturity of the component bond.
    call_price: scalar
        Redemption price of the component bond.
    main_bond_schedule: QuantLib.Schedule
        Cash-flow schedule of the parent bond.
    calendar: QuantLib.Calendar
        Calendar of the parent bond.
    business_convention: int
        Business-day convention of the parent bond.
    tenor: QuantLib.Period
        Coupon payment frequency of the parent bond.
    date_generation: QuantLib.DateGeneration
        Date-generation pattern of the parent bond's schedule.
    month_end: bool
        End of month parameter of the parent bond's schedule.
    settlement_days: int
        Default settlement days for trades in the parent bond.
    face_amount: scalar
        Face amount of the parent bond.
    day_counter: QuantLib.DayCounter
        DayCounter of the parent bond.
    issue_date: QuantLib.Date
        Issue date of the parent bond.
    index: QuantLib.IborIndex
        Index used for fixings
    fixed_coupon_len: int
        The number of coupon payments of the fixed leg
    fixed_coupon: list
        A list with the fixed coupon value(s)
    floating_coupon: list
        A list with the floating coupon value(s)
    caps: list
        A list with the coupon max value
    floors: list
        A list with the coupon min value
    in_arrears: bool

    Returns
    -------
    QuantLib.FloatingRateBond
        Bond representing a component of the parent bond, with the passed call date and price.

    """
    fixed_coupon = to_list(fixed_coupon)
    floating_coupon = to_list(floating_coupon)
    schedule = create_schedule_for_component(call_date, main_bond_schedule, calendar, business_convention, tenor,
                                             date_generation, month_end)
    floating_coupon_len = len(schedule) - fixed_coupon_len - 1
    gearings = [0]*fixed_coupon_len + [1]*floating_coupon_len
    spreads = fixed_coupon*fixed_coupon_len + floating_coupon*floating_coupon_len
    return ql.FloatingRateBond(settlement_days, face_amount, schedule, index, day_counter, business_convention,
                               index.fixingDays(), gearings, spreads, caps, floors, in_arrears, call_price, issue_date)


class ContingentConvertibleBond(FloatingRateBond):
    """ Callable fixed rate bond.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        The TimeSeries representing the bond.

    Note
    ----
    The `timeseries` attribute needs a component TimeSeries `call_schedule`, containing the call dates and call prices
    schedule in its `ts_values`.

    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, reference_curve=None, index_timeseries=None):
        super().__init__(timeseries, reference_curve, index_timeseries)
        self.coupon_reset_date = to_ql_date(to_datetime(self.ts_attributes[COUPON_TYPE_RESET_DATE]))
        self.fixed_coupons = [float(self.ts_attributes[COUPONS])]
        self.fixed_schedule = ql.Schedule(self.first_accrual_date, self.coupon_reset_date, self.coupon_frequency,
                                          self.calendar, self.business_convention, self.business_convention,
                                          self.date_generation, self.month_end)
        fixed_schedule_len = len(self.fixed_schedule) - 1
        float_schedule_len = len(self.schedule) - fixed_schedule_len - 1
        self.gearings = [0]*fixed_schedule_len + [1]*float_schedule_len
        self.total_spreads = self.fixed_coupons*fixed_schedule_len + self.spreads*float_schedule_len
        called_date = self.ts_attributes[CALLED_DATE]
        if called_date:
            self.expire_date = to_ql_date(to_datetime(called_date))
        self.callability_schedule = ql.CallabilitySchedule()
        # noinspection PyCompatibility
        for call_date, call_price in self.call_schedule.ts_values.iteritems():
            # The original bond (with maturity at self.maturity will be added to the components after its
            # instantiation below.
            call_date = to_ql_date(to_datetime(call_date))
            callability_price = ql.CallabilityPrice(call_price, ql.CallabilityPrice.Clean)
            self.callability_schedule.append(ql.Callability(callability_price, ql.Callability.Call, call_date))
            self.bond_components[call_date] = create_call_component(call_date, call_price, self.schedule,
                                                                    self.calendar, self.business_convention,
                                                                    self.coupon_frequency, self.date_generation,
                                                                    self.month_end, self.settlement_days,
                                                                    self.face_amount, self.day_counter, self.issue_date,
                                                                    self.index, fixed_schedule_len, self.fixed_coupons,
                                                                    self.spreads, self.caps, self.floors,
                                                                    self.in_arrears)

        self.bond = ql.FloatingRateBond(self.settlement_days, self.face_amount, self.schedule, self.index,
                                        self.day_counter, self.business_convention, self.index.fixingDays(),
                                        self.gearings, self.total_spreads, self.caps, self.floors, self.in_arrears,
                                        self.redemption, self.issue_date)
        self.fixing_schedule = ql.Schedule(self.first_accrual_date, self.maturity_date, self._index_tenor,
                                           self.calendar, self.business_convention, self.business_convention,
                                           ql.DateGeneration.Forward, self.month_end)
        self.reference_schedule = list(self.fixing_schedule)[:-1]
        self.fixing_dates = [self.index.fixingDate(dt) for dt in self.reference_schedule]
        # Coupon pricers
        self.pricer = ql.BlackIborCouponPricer()
        self.volatility = 0.0
        self.vol = ql.ConstantOptionletVolatility(self.settlement_days, self.index_calendar,
                                                  self.index_bus_day_convention, self.volatility, self.day_counter)
        self.pricer.setCapletVolatility(ql.OptionletVolatilityStructureHandle(self.vol))
        self.bond_components[self.maturity_date] = self.bond  # Add the original bond to bond_components.
        for bond in self.bond_components.values():
            ql.setCouponPricer(bond.cashflows(), self.pricer)
        self._bond_components_backup = self.bond_components.copy()
