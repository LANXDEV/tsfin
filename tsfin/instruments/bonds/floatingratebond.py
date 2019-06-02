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
from tsfin.base.basetools import conditional_vectorize, to_datetime, adjust_rate_from_quote_type
from tsfin.instruments.bonds._basebond import _BaseBond, default_arguments, create_schedule_for_component
from tsfin.constants import INDEX_TENOR, FIXING_DAYS, CALENDAR, YIELD, CLEAN_PRICE, DIRTY_PRICE


def clear_history_and_set_date(date, index, **kwargs):

    date = to_ql_date(date)
    if kwargs.get('bypass_set_floating_rate_index'):
        ql.Settings.instance().evaluationDate = date
    else:
        ql.IndexManager.instance().clearHistory(index.name())
        ql.Settings.instance().evaluationDate = date


# def set_floating_rate_index(f):
#     """ Decorator to set values of the floating rate bond's index.
#
#     This is needed for correct cash flow projection. Does nothing if the bond has no floating coupon.
#
#     Parameters
#     ----------
#     f: method
#         A method that needs the bond to have its past index values set.
#
#     Returns
#     -------
#     function
#         `f` itself. Only the bond instance is modified with this decorator.
#
#     Note
#     ----
#     If the wrapped function is called with a True-like optional argument 'bypass_set_floating_rate_index',
#     the effects of this wrapper is bypassed. This is useful for nested calling of methods that are wrapped by
#     function.
#     """
#     @wraps(f)
#     def new_f(self, **kwargs):
#         if kwargs.get('bypass_set_floating_rate_index'):
#             return f(self, **kwargs)
#         try:
#             date = to_ql_date(kwargs['date'][0])
#         except:
#             date = to_ql_date(kwargs['date'])
#         ql.Settings.instance().evaluationDate = date
#         index_timeseries = getattr(self, 'index_timeseries')
#         calendar = getattr(self, 'calendar')
#         index = getattr(self, 'index')
#         forecast_curve = getattr(self, 'forecast_curve')
#         reference_curve = getattr(self, 'reference_curve')
#         index_reference_curve = getattr(self, 'index_reference_curve')
#         forecast_curve.linkTo(reference_curve.yield_curve(date))
#         index_reference_curve.linkTo(reference_curve.yield_curve(date))
#         fixing_days = getattr(self, 'fixing_days')
#         for dt in getattr(self, 'fixing_dates'):
#             if dt <= calendar.advance(date, fixing_days, ql.Days):
#                 rate = index_timeseries.get_values(index=dt)
#                 quote_type_dict = {'BPS': float(10000), 'PERCENT': float(100)}
#                 quote_type = index_timeseries.get_attribute(QUOTE_TYPE)
#                 rate /= float(quote_type_dict.get(quote_type, 100))
#                 index.addFixing(dt, rate)
#         result = f(self, **kwargs)
#         ql.IndexManager.instance().clearHistory(index.name())
#         return result
#     new_f._decorated_by_floating_rate_index_ = True
#     return new_f


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

    @conditional_vectorize('date')
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

        date = to_ql_date(date)
        ql.IndexManager.instance().clearHistory(self.index.name())
        for dt in self.fixing_dates:
            if dt <= self.calendar.advance(date, self.fixing_days, ql.Days):
                rate = adjust_rate_from_quote_type(self.index_timeseries, dates=date)
                self.index.addFixing(dt, rate)

    @conditional_vectorize('date')
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

        date = to_ql_date(date)
        reference_curve = self.reference_curve.yield_curve(date)
        self.forecast_curve.linkTo(reference_curve)
        self.index_reference_curve.linkTo(reference_curve)

    def rate_helper(self, date, last_available=True, yield_type='ytm', curve_type='zero', max_inactive_days=3,
                    reference_date_for_worst_date=None, **kwargs):
        """
        Warnings
        --------
        This method is not implemented for this class yet. It returns None by default.

        Parameters
        ----------
        date: QuantLib.Date
            Date of the rate helper.
        last_available: bool, optional
            Whether to use last available quotes if missing data.
        yield_type: {'ytm', 'ytw'}, optional
            Which yield to use for the rate helper. Default is 'ytm'.
        curve_type: {'zero', 'par'}, optional
            Which type of yield curve will be built. Default is 'zero'.
        max_inactive_days: int, optional
            After how many days of missing data will this method return None. Default is 3.
        reference_date_for_worst_date: QuantLib.Date
            Date of the quote to be used to calculate the worst date of the bond.

        Returns
        -------
        QuantLib.RateHelper
            Object used to build yield curves.
        """

        if self.is_expired(date):
            return None
        if self.quotes.ts_values.last_valid_index() is None:
            return None
        if self.quotes.ts_values.last_valid_index() <= to_datetime(date) - timedelta(max_inactive_days):
            # To avoid returning bond helper for bonds that have been called, exchanged, etc...
            # print("Returning none..")
            return None
        quote = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(quote):
            return None
        date = to_ql_date(date)
        if reference_date_for_worst_date is not None:
            reference_date_for_worst_date = to_ql_date(reference_date_for_worst_date)
            reference_quote_for_worst_date = self.price.get_values(index=reference_date_for_worst_date,
                                                                   last_available=last_available,
                                                                   fill_value=np.nan)
        else:
            reference_date_for_worst_date = date
            reference_quote_for_worst_date = quote
        if curve_type == 'zero':
            if yield_type == 'ytw':
                maturity = self.worst_date(date=reference_date_for_worst_date, quote=reference_quote_for_worst_date)
                schedule = create_schedule_for_component(maturity, self.schedule, self.calendar,
                                                         self.business_convention, self.coupon_frequency,
                                                         self.date_generation, self.month_end)
            elif yield_type == 'ytw_rolling_call':
                maturity = self.worst_date_rolling_call(date=reference_date_for_worst_date,
                                                        quote=reference_quote_for_worst_date)
                schedule = create_schedule_for_component(maturity, self.schedule, self.calendar,
                                                         self.business_convention, self.coupon_frequency,
                                                         self.date_generation, self.month_end)
            else:
                schedule = self.schedule
            clean_price = self.clean_price(date=date, quote=quote)
            ql.Settings.instance().evaluationDate = date
            return ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(clean_price)), self.settlement_days,
                                          self.face_amount, schedule, self.coupons, self.day_counter,
                                          self.business_convention, self.redemption, self.issue_date)
        elif curve_type == 'par':
            if yield_type == 'ytw':
                maturity = self.worst_date(date=reference_date_for_worst_date,
                                           quote=reference_quote_for_worst_date)
                yld = self.yield_to_date(to_date=maturity,
                                         date=date,
                                         quote=quote)

            elif yield_type == 'ytw_rolling_call':
                maturity = self.worst_date_rolling_call(date=reference_date_for_worst_date,
                                                        quote=reference_quote_for_worst_date)
                yld = self.yield_to_date(to_date=maturity,
                                         date=date,
                                         quote=quote)
            else:
                yld, maturity = self.ytm(date=date, quote=quote), self.maturity_date
            # Convert rate to simple compounding because DepositRateHelper expects simple rates.
            time = self.day_counter.yearFraction(date, maturity)
            tenor = ql.Period(self.calendar.businessDaysBetween(date, maturity), ql.Days)
            simple_yld = ql.InterestRate(yld,
                                         self.day_counter,
                                         self.yield_quote_compounding,
                                         self.yield_quote_frequency).equivalentRate(ql.Simple,
                                                                                    ql.Annual,
                                                                                    time).rate()
            ql.Settings.instance().evaluationDate = date
            return ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(simple_yld)), tenor, 0, self.calendar,
                                        self.business_convention, False, self.day_counter)
        else:
            raise ValueError("Bond class rate_helper method does not support curve_type = {}".format(curve_type))

    @default_arguments
    @conditional_vectorize('date')
    def accrued_interest(self, last, date, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            Date of the calculation.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The accrued interest of the bond.
        """

        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        if date >= self.maturity_date:
            return np.nan
        return self.bond.accruedAmount(date)

    @default_arguments
    @conditional_vectorize('date')
    def cash_to_date(self, start_date, last, date, **kwargs):
        """
        Parameters
        ----------
        start_date: QuantLib.Date
            The start date.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The last date of the computation.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Amount of cash received between `start_date` and `date`.


        """

        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        start_date = to_ql_date(start_date)
        return sum(cf.amount() for cf in self.bond.cashflows() if start_date <= cf.date() <= date) / self.face_amount

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def clean_price(self, last, quote, date, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's clean price.

        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        if self.quote_type == CLEAN_PRICE:
            return quote
        elif self.quote_type == DIRTY_PRICE:
            # TODO: This part needs testing.
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return quote - self.accrued_interest(last=last, date=settlement_date, **kwargs)
        elif self.quote_type == YIELD:
            # TODO: This part needs testing.
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            ql.Settings.instance().evaluationDate = settlement_date
            return self.bond.cleanPrice(quote, self.day_counter, self.yield_quote_compounding,
                                        self.yield_quote_frequency)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def dirty_price(self, last, quote, date, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's dirty price.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        if self.quote_type == CLEAN_PRICE:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return quote + self.accrued_interest(last=last, date=settlement_date, **kwargs)
        elif self.quote_type == DIRTY_PRICE:
            return quote
        elif self.quote_type == YIELD:
            # TODO: This part needs testing.
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return self.bond.dirtyPrice(quote, self.day_counter, self.yield_quote_compounding,
                                        self.yield_quote_frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def value(self, last, quote, date, last_available=False, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        last_available: bool, optional
            Whether to use last available data.
            Default: False.

        Returns
        -------
        scalar
            The dirty value of the bond as of `date`.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        if last_available:
            quote = self.price.get_values(index=date, last_available=last_available)
            return self.dirty_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount
        if date > self.quotes().last_valid_index():
            return np.nan
        if date < self.quotes().first_valid_index():
            return np.nan
        return self.dirty_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, last=False, **kwargs):
        """
        Parameters
        ----------
        start_date: datetime-like, optional
            The starting date of the period.
            Default: The first date in ``self.quotes``.
        date: datetime-like, optional, (c-vectorized)
            The ending date of the period.
            Default: see :py:func:`default_arguments`.
        start_quote: scalar, optional, (c-vectorized)
            The quote of the instrument in `start_date`.
            Default: the quote in `start_date`.
        quote: scalar, optional, (c-vectorized)
            The quote of the instrument in `date`.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar, None
            Performance of a unit of the bond.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        first_available_date = self.quotes().first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = self.price.get_values(index=start_date)
        if date < start_date:
            return np.nan
        start_value = self.value(quote=start_quote, date=start_date)
        value = self.value(quote=quote, date=date)
        paid_interest = self.cash_to_date(start_date=start_date, date=date)
        return (value + paid_interest) / start_value - 1

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytm(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's yield to maturity.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        bond = kwargs.get('bond', self.bond)  # Useful to pass bonds other than self as arguments.
        date = to_ql_date(date)
        ql.Settings.instance().evaluationDate = date
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan
        if self.quote_type == CLEAN_PRICE:
            return bond.bondYield(quote, day_counter, compounding, frequency, settlement_date)
        elif self.quote_type == DIRTY_PRICE:
            # TODO: This part needs testing.
            clean_quote = quote - self.accrued_interest(last=last, date=settlement_date, **kwargs)
            return bond.bondYield(clean_quote, day_counter, compounding, frequency, settlement_date)
        elif self.quote_type == YIELD:
            # TODO: This part needs testing.
            interest_rate = ql.InterestRate(quote, self.day_counter, self.yield_quote_compounding,
                                            self.yield_quote_frequency)
            return interest_rate.equivalentRate(compounding, frequency, 1)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytw_and_worst_date(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        tuple (scalar, QuantLib.Date)
            The bond's yield to worst and worst date.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan, np.nan
        return min(((self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                              frequency=frequency, settlement_days=settlement_days, bond=bond,
                              bypass_set_floating_rate_index=True), key)
                    for key, bond in self.bond_components.items() if key > settlement_date), key=itemgetter(0))

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytw(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's yield to worst.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan
        return min((self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                             frequency=frequency, settlement_days=settlement_days, bond=bond,
                             bypass_set_floating_rate_index=True)
                    for key, bond in self.bond_components.items() if key > settlement_date))

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytw_and_worst_date_rolling_call(self, last, quote, date, day_counter, compounding, frequency, settlement_days,
                                        **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        tuple (scalar, QuantLib.Date)
            The bond's yield to worst and worst date, considering that, anytime after the first date in the call
            schedule, the issuer can call the bond with a 30 days notice.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        if date < list(self.bond_components.keys())[0]:
            # If date is before the first call dates, rolling calls are NOT possible.
            return self.ytw_and_worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                           compounding=compounding, frequency=frequency,
                                           settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                                           **kwargs)
        else:
            # Then rolling calls are possible.
            rolling_call_date = self.calendar.advance(date, 1, ql.Months, self.business_convention)
            rolling_call_component = self._create_call_component(to_date=rolling_call_date)
            self._insert_bond_component(rolling_call_date, rolling_call_component)
            yield_value, worst_date = self.ytw_and_worst_date(last=last, quote=quote, date=date,
                                                              day_counter=day_counter, compounding=compounding,
                                                              frequency=frequency, settlement_days=settlement_days,
                                                              **kwargs)
            self._restore_bond_components()
            return yield_value, worst_date

    @default_arguments
    def ytw_rolling_call(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's yield to worst, considering that, anytime after the first date in the call
            schedule, the issuer can call the bond with a 30 days notice.
        """
        return self.ytw_and_worst_date_rolling_call(last=last, quote=quote, date=date, day_counter=day_counter,
                                                    compounding=compounding, frequency=frequency,
                                                    settlement_days=settlement_days, **kwargs)[0]

    @default_arguments
    def yield_to_date(self, to_date, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        to_date: QuantLib.Date
            The maturity date under consideration.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            The bond's yield if its maturity were the `to_date` argument.
        """
        date = to_ql_date(date)
        to_date = to_ql_date(to_date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        self.bond_components = {to_date: self._create_call_component(to_date=to_date)}
        result = self.ytw(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                          frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                          **kwargs)
        self._restore_bond_components()
        return result

    @default_arguments
    def worst_date(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         QuantLib.Date
            The bond's worst date.
        """
        return self.ytw_and_worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                       compounding=compounding, frequency=frequency, settlement_days=settlement_days,
                                       **kwargs)[1]

    @default_arguments
    def worst_date_rolling_call(self, last, quote, date, day_counter, compounding, frequency, settlement_days,
                                **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional, (c-vectorized)
            The bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         QuantLib.Date
            The bond's worst date, considering that, anytime after the first date in the call
            schedule, the issuer can call the bond with a 30 days notice.
        """
        return self.ytw_and_worst_date_rolling_call(last=last, quote=quote, date=date, day_counter=day_counter,
                                                    compounding=compounding, frequency=frequency,
                                                    settlement_days=settlement_days, **kwargs)[1]

    @default_arguments
    @conditional_vectorize('ytm', 'date')
    def clean_price_from_ytm(self, last, ytm, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        ytm: scalar, (c-vectorized)
            The bond's yield to maturity.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         scalar
            The bond's clean price given its yield to maturity.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        return self.bond.cleanPrice(ytm, day_counter, compounding, frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('yield_to_date', 'date')
    def clean_price_from_yield_to_date(self, to_date, last, yield_to_date, date, day_counter, compounding, frequency,
                                       settlement_days, **kwargs):
        """
        Parameters
        ----------
        to_date: QuantLib.Date
            The maturity date considered for the `yield_to_date` parameter.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        yield_to_date: scalar, (c-vectorized)
            The bond's yield to `to_date`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         scalar
            The bond's clean price given its yield to `to_date`.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        to_date = to_ql_date(to_date)
        bond = self._create_call_component(to_date=to_date)
        return bond.cleanPrice(yield_to_date, day_counter, compounding, frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('ytm', 'date')
    def dirty_price_from_ytm(self, last, ytm, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        ytm: scalar, (c-vectorized)
            Yield to maturity.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         scalar
            The bond's clean price given its yield to maturity.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(self.settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        return self.bond.dirtyPrice(ytm, day_counter, compounding, frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('yield_to_date', 'date')
    def dirty_price_from_yield_to_date(self, to_date, last, yield_to_date, date, day_counter, compounding, frequency,
                                       settlement_days, **kwargs):
        """
        Parameters
        ----------
        to_date: QuantLib.Date
            The maturity date considered in the `yield_to_date` parameter.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments`.
        yield_to_date: scalar, optional, (c-vectorized)
            The yield to `to_date`.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional, (c-vectorized)
            The date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            The compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            The compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
         QuantLib.Date
            The bonds' implied dirty price.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        to_date = to_ql_date(to_date)
        bond = self._create_call_component(to_date=to_date)
        return bond.dirtyPrice(yield_to_date, day_counter, compounding, frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
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

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        ytm = self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                       frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                       **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        P = self.dirty_price(last=last, quote=quote, date=date, day_counter=day_counter,
                             compounding=compounding, frequency=frequency, settlement_days=settlement_days,
                             bypass_set_floating_rate_index=True)
        dy = 1e-5
        forecast_curve_timeseries = self.reference_curve
        node_dates = forecast_curve_timeseries.yield_curve(date=date).dates()
        node_rates = forecast_curve_timeseries.zero_rate_to_date(date=date, to_date=node_dates,
                                                                 compounding=ql.Simple,
                                                                 frequency=ql.Annual)
        lower_shifted_curve = ql.LogLinearZeroCurve(list(node_dates), [r - dy for r in node_rates],
                                                    forecast_curve_timeseries.day_counter,
                                                    forecast_curve_timeseries.calendar,
                                                    ql.LogLinear(),
                                                    ql.Simple,
                                                    )
        upper_shifted_curve = ql.LogLinearZeroCurve(list(node_dates), [r + dy for r in node_rates],
                                                    forecast_curve_timeseries.day_counter,
                                                    forecast_curve_timeseries.calendar,
                                                    ql.LogLinear(),
                                                    ql.Simple,
                                                    )
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
        return -(1/P)*(P_p - P_m)/(2*dy)

    @default_arguments
    @conditional_vectorize('quote', 'date')
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

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        ytm = self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                       frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                       **kwargs)

        mac_duration = self.duration_to_mat(duration_type=duration_type, last=last, quote=quote, date=date,
                                            day_counter=day_counter, compounding=compounding, frequency=frequency,
                                            settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                                            **kwargs)

        return (1 + ytm / self.yield_quote_frequency) * mac_duration

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
        return self.duration_to_mat(duration_type=duration_type, last=last, quote=quote, day_counter=day_counter,
                                    compounding=compounding, frequency=frequency, settlement_days=settlement_days)

    @default_arguments
    @conditional_vectorize('quote', 'date')
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
        """
        # TODO: Implement this.
        return np.nan

    @default_arguments
    @conditional_vectorize('quote', 'date')
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
        """
        # TODO: Implement this.
        return np.nan

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
        """
        # TODO: Implement this.
        return np.nan

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def zspread_to_mat(self, yield_curve_timeseries, last, quote, date, day_counter, compounding, frequency,
                       settlement_days, **kwargs):
        """
        Parameters
        ----------
        yield_curve_timeseries: :py:func:`YieldCurveTimeSeries`
            The yield curve object against which the z-spreads will be calculated.
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
            Bond's z-spread to maturity relative to `yield_curve_timeseries`.
        """
        bond = self.bond

        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        yield_curve = yield_curve_timeseries.yield_curve(date=date)
        ql.Settings.instance().evaluationDate = date
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        if self.quote_type == CLEAN_PRICE:
            return ql.BondFunctions_zSpread(bond, quote, yield_curve, day_counter, compounding, frequency,
                                            settlement_date)
        elif self.quote_type == DIRTY_PRICE:
            # TODO: This part needs testing.
            clean_quote = quote - self.accrued_interest(last=last, date=settlement_date,
                                                        bypass_set_floating_rate_index=True, **kwargs)
            return ql.BondFunctions_zSpread(bond, clean_quote, yield_curve, day_counter, compounding, frequency,
                                            settlement_date)
        elif self.quote_type == YIELD:
            # TODO: This part needs testing.
            clean_quote = self.clean_price_from_ytm(last=last, quote=quote, date=date, day_counter=day_counter,
                                                    compounding=compounding, frequency=frequency,
                                                    bypass_set_floating_rate_index=True,
                                                    settlement_days=settlement_days)
            return ql.BondFunctions_zSpread(bond, clean_quote, yield_curve, day_counter, compounding, frequency,
                                            settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def zspread_to_worst(self, yield_curve_timeseries, last, quote, date, day_counter, compounding, frequency,
                         settlement_days, **kwargs):
        """
        Parameters
        ----------
        yield_curve_timeseries: :py:func:`YieldCurveTimeSeries`
            The yield curve object against which the z-spreads will be calculated.
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
            Bond's z-spread to worst, relative to `yield_curve_timeseries`.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        worst_date = self.worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                     compounding=compounding, frequency=frequency,
                                     settlement_days=settlement_days, bypass_set_floating_rate_index=True, **kwargs)
        bond = self.bond_components[worst_date]
        return self.zspread_to_mat(yield_curve_timeseries=yield_curve_timeseries, last=last, quote=quote, date=date,
                                   day_counter=day_counter, compounding=compounding, frequency=frequency,
                                   settlement_days=settlement_days, bond=bond, bypass_set_floating_rate_index=True,
                                   **kwargs)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def zspread_to_worst_rolling_call(self, yield_curve_timeseries, last, quote, date, day_counter, compounding,
                                      frequency, settlement_days, **kwargs):
        """
        Parameters
        ----------
        yield_curve_timeseries: :py:func:`YieldCurveTimeSeries`
            The yield curve object against which the z-spreads will be calculated.
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
            Bond's z-spread to worst, considering rolling calls, relative to `yield_curve_timeseries`.
        """
        date = to_ql_date(date)

        ql.Settings.instance().evaluationDate = date
        self.link_to_curves(date=date, **kwargs)
        self.add_fixings(date=date, **kwargs)

        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        worst_date = self.worst_date_rolling_call(last=last, quote=quote, date=date,
                                                  day_counter=day_counter, compounding=compounding,
                                                  frequency=frequency, settlement_days=settlement_days,
                                                  bypass_set_floating_rate_index=True,
                                                  **kwargs)
        bond = self._create_call_component(to_date=worst_date)
        return self.zspread_to_mat(yield_curve_timeseries=yield_curve_timeseries, last=last, quote=quote, date=date,
                                   day_counter=day_counter, compounding=compounding, frequency=frequency,
                                   settlement_days=settlement_days, bond=bond, bypass_set_floating_rate_index=True,
                                   **kwargs)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def oas(self, yield_curve_timeseries, model, model_params, last, quote, date, day_counter, compounding, frequency,
            settlement_days, **kwargs):
        """
        Warning
        -------
        This method merely returns the z-spread of the bond. See :py:obj:`FixedRateCallableBond` for true
        option-adjusted spread.

        Parameters
        ----------
        yield_curve_timeseries: :py:func:`YieldCurveTimeSeries`
            The yield curve object against which the z-spreads will be calculated.
        model: QuantLib.ShortRateModel
            Used in :py:obj:`CallableFixedRateBond`.
        model_params: tuple, dict
            Used in :py:obj:`CallableFixedRateBond`.
        last: bool, optional
            Whether to last data.
            Default: see :py:obj:`default_arguments`.
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
            Bond's z-spread relative to `yield_curve_timeseries`.
        """
        return self.zspread_to_mat(yield_curve_timeseries=yield_curve_timeseries, last=last, quote=quote,
                                   date=date, day_counter=day_counter, compounding=compounding,
                                   frequency=frequency, settlement_days=settlement_days, **kwargs)



'''
###########################################################################################
Methods of the FloatingRateBond class need to be decorated with set_floating_rate_index.
So we unwrap the base class methods from their wrappers and include set_floating_rate_index 
in the wrapper chain.
###########################################################################################
'''

# Methods to redecorate with set_floating_rate_index.
# methods_to_redecorate = ['accrued_interest',
#                          'cash_to_date',
#                          'clean_price',
#                          'dirty_price',
#                          'value',
#                          'performance',
#                          'ytm',
#                          'ytw_and_worst_date',
#                          'ytw',
#                          'ytw_and_worst_date_rolling_call',
#                          'ytw_rolling_call',
#                          'yield_to_date',
#                          'worst_date',
#                          'worst_date_rolling_call',
#                          'clean_price_from_ytm',
#                          'clean_price_from_yield_to_date',
#                          'dirty_price_from_ytm',
#                          'dirty_price_from_yield_to_date',
#                          'zspread_to_mat',
#                          'zspread_to_worst',
#                          'zspread_to_worst_rolling_call',
#                          'oas',
#                          ]


# def redecorate_with_set_floating_rate_index(cls, method):
#     if not getattr(method, '_decorated_by_floating_rate_index_', None):
#         if getattr(method, '_conditional_vectorized_', None) and \
#                 getattr(method, '_decorated_by_default_arguments_', None):
#             vectorized_args = method._conditional_vectorize_args_
#             setattr(cls, method.__name__, default_arguments(conditional_vectorize(vectorized_args)(
#                 set_floating_rate_index(method))))
#         elif getattr(method, '_conditional_vectorized_', None):
#             vectorized_args = method._conditional_vectorize_args_
#             setattr(cls, method.__name__, conditional_vectorize(vectorized_args)(set_floating_rate_index(method)))
#         elif getattr(method, '_decorated_by_default_arguments_', None):
#             setattr(cls, method.__name__, default_arguments(set_floating_rate_index(method)))
#
#
# # Redecorating methods.
# for method_name in methods_to_redecorate:
#     redecorate_with_set_floating_rate_index(FloatingRateBond, getattr(FloatingRateBond, method_name))



