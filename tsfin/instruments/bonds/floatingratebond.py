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

from functools import wraps
import numpy as np
import QuantLib as ql
from operator import itemgetter
from datetime import timedelta
from tsfin.base.qlconverters import to_ql_date, to_ql_calendar, to_ql_currency, to_ql_ibor_index
from tsfin.base.basetools import conditional_vectorize, to_datetime
from tsfin.instruments.bonds._basebond import _BaseBond, default_arguments, create_schedule_for_component
from tsfin.constants import INDEX_TENOR, FIXING_DAYS, CALENDAR, YIELD, CLEAN_PRICE, DIRTY_PRICE


def set_floating_rate_index(f):
    """ Decorator to set values of the floating rate bond's index.

    This is needed for correct cash flow projection. Does nothing if the bond has no floating coupon.

    Parameters
    ----------
    f: method
        A method that needs the bond to have its past index values set.

    Returns
    -------
    function
        `f` itself. Only the bond instance is modified with this decorator.
    """
    @wraps(f)
    def new_f(self, *args, **kwargs):
        try:
            date = to_ql_date(kwargs['date'][0])
        except:
            date = to_ql_date(kwargs['date'])
        ql.Settings.instance().evaluationDate = date
        self.add_fixings(date=date)
        self.link_to_curves(date=date)
        return f(self, *args, **kwargs)
    new_f._decorated_by_floating_rate_index_ = True
    return new_f


class FloatingRateBond(_BaseBond):
    """ Floating rate bond.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        The TimeSeries representing the bond.
    index_timeseries: :py:obj:`TimeSeries`
        The TimeSeries containing the quotes of the index used to calculate the bond's coupon.
    reference_curve: :py:obj:YieldCurveTimeSeries
        The yield curve of the index rate, used to estimate future cash flows.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, reference_curve, index_timeseries):
        super().__init__(timeseries)
        self.reference_curve = reference_curve
        self.index_timeseries = index_timeseries
        self.forecast_curve = ql.RelinkableYieldTermStructureHandle()
        self.index_reference_curve = ql.RelinkableYieldTermStructureHandle()
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.fixing_days = self.ts_attributes[FIXING_DAYS]
        self.index = to_ql_ibor_index('{0}_{1}_Libor'.format(self.ts_name, self.currency), self._index_tenor,
                                      self.fixing_days, to_ql_currency(self.currency), self.calendar,
                                      self.business_convention, self.month_end, self.day_counter,
                                      self.index_reference_curve)
        self.gearings = [1]  # TODO: Make it possible to have a list of different gearings.
        self.spreads = [float(self.ts_attributes["SPREAD"])]  # TODO: Make it possible to have different spreads.
        self.caps = []  # TODO: Make it possible to have caps.
        self.floors = []  # TODO: Make it possible to have floors.
        self.in_arrears = False  # TODO: Check if this can be made variable.
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.schedule = ql.Schedule(self.first_accrual_date, self.maturity_date, self.coupon_frequency, self.calendar,
                                    self.business_convention, self.business_convention, self.date_generation,
                                    self.month_end)
        self.bond = ql.FloatingRateBond(self.settlement_days, self.face_amount, self.schedule, self.index,
                                        self.day_counter, self.business_convention, self.index.fixingDays(),
                                        self.gearings, self.spreads, self.caps, self.floors, self.in_arrears,
                                        self.redemption, self.issue_date)
        self.coupon_dates = [cf.date() for cf in self.bond.cashflows()]
        # Store the fixing dates of the coupon. These will be useful later.
        self.index_calendar = self.index.fixingCalendar()
        self.index_bus_day_convention = self.index.businessDayConvention()
        self.reference_schedule = list(self.schedule)[:-1]
        self.fixing_dates = [self.index.fixingDate(dt) for dt in self.reference_schedule]
        # Coupon pricers
        self.pricer = ql.BlackIborCouponPricer()
        self.volatility = 0.0
        self.vol = ql.ConstantOptionletVolatility(self.settlement_days, self.index_calendar,
                                                  self.index_bus_day_convention, self.volatility, self.day_counter)
        self.pricer.setCapletVolatility(ql.OptionletVolatilityStructureHandle(self.vol))
        ql.setCouponPricer(self.bond.cashflows(), self.pricer)
        self.bond_components[self.maturity_date] = self.bond
        self._bond_components_backup = self.bond_components.copy()

    def add_fixings(self, date, **kwargs):

        """
        Create the coupon fixings before the evaluation date
        :param date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
        bypass_set_floating_rate_index: bool
             If true it avoids recalculating the index. Useful for nested calls.
        :return:
        """
        if kwargs.get('bypass_set_floating_rate_index'):
            return

        ql.IndexManager.instance().clearHistory(self.index.name())
        for dt in self.fixing_dates:
            if dt <= self.calendar.advance(date, self.fixing_days, ql.Days):
                rate = self.index_timeseries.get_values(index=dt)
                self.index.addFixing(dt, rate)

    def link_to_curves(self, date, **kwargs):

        """

        :param date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
        bypass_set_floating_rate_index: bool
             If true it avoids recalculating the index. Useful for nested calls.
        :return:
        """
        if kwargs.get('bypass_set_floating_rate_index'):
            return

        reference_curve = self.reference_curve.yield_curve(date)
        self.forecast_curve.linkTo(reference_curve)
        self.index_reference_curve.linkTo(reference_curve)

    @set_floating_rate_index
    def minor_price_change(self, last, quote, date, day_counter, compounding, frequency, settlement_days, dy,
                           to_worst=False, rolling_call=False, **kwargs):

        date = to_ql_date(date)
        forecast_curve_timeseries = self.reference_curve
        self.forecast_curve.linkTo(forecast_curve_timeseries.yield_curve(date=date))
        P = self.dirty_price(last=last, quote=quote, date=date, settlement_days=settlement_days,
                             quote_type='YIELD_CURVE', yield_curve=forecast_curve_timeseries)
        if to_worst:
            if rolling_call:
                ytw, worst_date = self.ytw_and_worst_date_rolling_call(last=last, quote=P, date=date,
                                                                       day_counter=day_counter, compounding=compounding,
                                                                       frequency=frequency,
                                                                       settlement_days=settlement_days,
                                                                       quote_type='DIRTY_PRICE',
                                                                       **kwargs)
            else:
                ytw, worst_date = self.ytw_and_worst_date(last=last, quote=P, date=date, day_counter=day_counter,
                                                          compounding=compounding, frequency=frequency,
                                                          settlement_days=settlement_days,
                                                          quote_type='DIRTY_PRICE', **kwargs)
            ytm = ytw
        else:
            ytm = self.ytm(last=last, quote=P, date=date, day_counter=day_counter, compounding=compounding,
                           frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                           quote_type='DIRTY_PRICE', **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan

        lower_shifted_curve = forecast_curve_timeseries.spreaded_curve(date=date, spread=-dy,
                                                                       compounding=self.yield_quote_compounding,
                                                                       frequency=self.yield_quote_frequency)
        upper_shifted_curve = forecast_curve_timeseries.spreaded_curve(date=date, spread=dy,
                                                                       compounding=self.yield_quote_compounding,
                                                                       frequency=self.yield_quote_frequency)

        self.forecast_curve.linkTo(lower_shifted_curve)
        P_m = ql.CashFlows.npv(self.bond.cashflows(),
                               ql.InterestRate(ytm - dy,
                                               self.day_counter,
                                               self.yield_quote_compounding,
                                               self.yield_quote_frequency),
                               True, settlement_date, settlement_date)
        self.forecast_curve.linkTo(upper_shifted_curve)
        P_p = ql.CashFlows.npv(self.bond.cashflows(),
                               ql.InterestRate(ytm + dy,
                                               self.day_counter,
                                               self.yield_quote_compounding,
                                               self.yield_quote_frequency),
                               True, settlement_date, settlement_date)

        return P, P_m, P_p, ytm

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def duration_to_mat(self, duration_type, last, quote, date, day_counter, compounding, frequency, settlement_days,
                        **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
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
            Bond's duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """
        date = to_ql_date(date)
        dy = 0.0001
        P, P_m, P_p, ytm = self.minor_price_change(last=last, quote=quote, date=date, day_counter=day_counter,
                                                   compounding=compounding, frequency=frequency,
                                                   settlement_days=settlement_days, dy=dy, to_worst=False, **kwargs)

        mac_duration = -(1 / P) * (P_p - P_m)/(2 * dy)
        if duration_type == ql.Duration.Modified:
            duration = (1 + ytm / self.yield_quote_frequency) * mac_duration
        else:
            duration = mac_duration
        return duration

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def duration_to_worst(self, duration_type, last, quote, date, day_counter, compounding, frequency, settlement_days,
                          **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
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
            Same as duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """

        date = to_ql_date(date)
        dy = 0.0001
        P, P_m, P_p, ytm = self.minor_price_change(last=last, quote=quote, date=date, day_counter=day_counter,
                                                   compounding=compounding, frequency=frequency,
                                                   settlement_days=settlement_days, dy=dy, to_worst=True, **kwargs)

        mac_duration = -(1 / P) * (P_p - P_m)/(2 * dy)
        if duration_type == ql.Duration.Modified:
            duration = (1 + ytm / self.yield_quote_frequency) * mac_duration
        else:
            duration = mac_duration
        return duration

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def duration_to_worst_rolling_call(self, duration_type, last, quote, date, day_counter, compounding, frequency,
                                       settlement_days, **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
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
            Same as duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """
        return self.duration_to_worst(duration_type=duration_type, last=last, quote=quote, day_counter=day_counter,
                                      compounding=compounding, frequency=frequency, settlement_days=settlement_days,
                                      rolling_call=True, **kwargs)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def convexity_to_mat(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
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
            Bond's convexity to maturity.
        TODO: Needs more testing
        """
        date = to_ql_date(date)
        dy = 0.0001
        P, P_m, P_p, ytm = self.minor_price_change(last=last, quote=quote, date=date, day_counter=day_counter,
                                                   compounding=compounding, frequency=frequency,
                                                   settlement_days=settlement_days, dy=dy, to_worst=False, **kwargs)
        convexity = (P_m + P_p - 2*P)/(2 * P * (dy*dy))
        return convexity

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def convexity_to_worst(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
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
            Bond's convexity to worst.
        TODO: Needs more testing
        """
        date = to_ql_date(date)
        dy = 0.0001
        P, P_m, P_p, ytm = self.minor_price_change(last=last, quote=quote, date=date, day_counter=day_counter,
                                                   compounding=compounding, frequency=frequency,
                                                   settlement_days=settlement_days, dy=dy, to_worst=True, **kwargs)
        convexity = (P_m + P_p - 2*P)/(2 * P * (dy*dy))
        return convexity

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def convexity_to_worst_rolling_call(self, last, quote, date, day_counter, compounding, frequency, settlement_days,
                                        **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
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
            Bond's convexity to worst.
        TODO: Needs more testing
        """
        return self.convexity_to_worst(last=last, quote=quote, day_counter=day_counter, compounding=compounding,
                                       frequency=frequency, settlement_days=settlement_days, rolling_call=True,
                                       **kwargs)


'''
###########################################################################################
Methods of the FloatingRateBond class need to be decorated with set_floating_rate_index.
So we unwrap the base class methods from their wrappers and include set_floating_rate_index 
in the wrapper chain.
###########################################################################################
'''

# Methods to redecorate with set_floating_rate_index.
methods_to_redecorate = ['accrued_interest',
                         'cash_to_date',
                         'clean_price',
                         'dirty_price',
                         'value',
                         'performance',
                         'ytm',
                         'ytw_and_worst_date',
                         'ytw',
                         'ytw_and_worst_date_rolling_call',
                         'ytw_rolling_call',
                         'yield_to_date',
                         'worst_date',
                         'worst_date_rolling_call',
                         'clean_price_from_ytm',
                         'clean_price_from_yield_to_date',
                         'dirty_price_from_ytm',
                         'dirty_price_from_yield_to_date',
                         'zspread_to_mat',
                         'zspread_to_worst',
                         'zspread_to_worst_rolling_call',
                         'oas',
                         ]


def redecorate_with_set_floating_rate_index(cls, method):
    if not getattr(method, '_decorated_by_floating_rate_index_', None):
        if getattr(method, '_conditional_vectorized_', None) and \
                getattr(method, '_decorated_by_default_arguments_', None):
            vectorized_args = method._conditional_vectorize_args_
            setattr(cls, method.__name__, default_arguments(conditional_vectorize(vectorized_args)(
                set_floating_rate_index(method))))
        elif getattr(method, '_conditional_vectorized_', None):
            vectorized_args = method._conditional_vectorize_args_
            setattr(cls, method.__name__, conditional_vectorize(vectorized_args)(set_floating_rate_index(method)))
        elif getattr(method, '_decorated_by_default_arguments_', None):
            setattr(cls, method.__name__, default_arguments(set_floating_rate_index(method)))


# Redecorating methods.
for method_name in methods_to_redecorate:
    redecorate_with_set_floating_rate_index(FloatingRateBond, getattr(FloatingRateBond, method_name))
