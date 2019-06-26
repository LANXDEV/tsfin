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
from tsfin.base.qlconverters import to_ql_date, to_ql_short_rate_model
from tsfin.base.basetools import conditional_vectorize, to_datetime
from tsfin.instruments.bonds._basebond import _BaseBond, default_arguments, create_call_component


class CallableFixedRateBond(_BaseBond):
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
    def __init__(self, timeseries):
        super().__init__(timeseries=timeseries)
        # TODO: Add support for puttable bonds.
        # TODO: Here we assume that the call prices are always clean prices. Fix this!
        # TODO: Implement an option to reduce (some kind of 'telescopic') call dates.
        # if there are too much. This is useful in case we are treating a callable perpetual bond, for example.
        self.callability_schedule = ql.CallabilitySchedule()
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
                                                                    self.face_amount, self.coupons,
                                                                    self.day_counter, self.issue_date)

        self.bond = ql.CallableFixedRateBond(self.settlement_days, self.face_amount, self.schedule, self.coupons,
                                             self.day_counter, self.business_convention, self.redemption,
                                             self.issue_date, self.callability_schedule)
        self.bond_components[self.maturity_date] = self.bond  # Add the original bond to bond_components.
        self._bond_components_backup = self.bond_components.copy()

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def oas(self, yield_curve_timeseries, model, model_params, last, quote, date, day_counter, compounding, frequency,
            settlement_days, **kwargs):
        """
        Warning
        -------
        This method has only been tested with ``model=QuantLib.HullWhite``.

        Parameters
        ----------
        yield_curve_timeseries: :py:func:`YieldCurveTimeSeries`
            The yield curve object against which the z-spreads will be calculated.
        model: str
            A string representing one of QuantLib Short Rate models, for simulating evolution of rates.
            **Currently only tested with QuantLib.HullWhite.**
        model_params: tuple, dict
            Parameter set for the model.
            * tuple format: (param1, param2, ...)
                If a tuple is passed, assumes the model parameters are fixed for all possibly vectorized calculation
                dates.
            * dict format: {date1: (param1, param2, ...), date2: (param1, param2, ...), ... }
                If a dict is passed, assumes it contains a parameter set for each date of the possibly vectorized
                calculation dates.

        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Bond's option-adjusted spread relative to `yield_curve_timeseries`.
        """

        bond = self.bond
        date = to_ql_date(date)
        yield_curve_relinkable_handle = yield_curve_timeseries.yield_curve_relinkable_handle(date=date)
        ql_model = to_ql_short_rate_model(model)
        ql.Settings.instance().evaluationDate = date
        if isinstance(model_params, dict):
            # Assumes model parameters are given for each date.
            ql_model = ql_model(yield_curve_relinkable_handle, *model_params[date])
        else:
            # Only one set of model parameters are given (calibrated for, say, a specific date).
            ql_model = ql_model(yield_curve_relinkable_handle, *model_params)
        engine = ql.TreeCallableFixedRateBondEngine(ql_model, 40)
        bond.setPricingEngine(engine)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        return bond.OAS(quote, yield_curve_relinkable_handle, day_counter, compounding, frequency, settlement_date)
