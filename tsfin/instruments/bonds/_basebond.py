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
from collections import OrderedDict
from operator import itemgetter
from datetime import timedelta
import numpy as np
import pandas as pd
import QuantLib as ql
from tsfin.base import Instrument, to_ql_date, to_ql_frequency, to_ql_business_convention, to_ql_calendar, \
    to_ql_compounding, to_ql_date_generation, to_ql_day_counter, conditional_vectorize, find_le, to_datetime
from tsfin.constants import BOND_TYPE, QUOTE_TYPE, CURRENCY, YIELD_QUOTE_COMPOUNDING, \
    YIELD_QUOTE_FREQUENCY, ISSUE_DATE, FIRST_ACCRUAL_DATE, MATURITY_DATE, CALENDAR, \
    BUSINESS_CONVENTION, DATE_GENERATION, SETTLEMENT_DAYS, FACE_AMOUNT, COUPONS, DAY_COUNTER, REDEMPTION, DISCOUNT, \
    YIELD, CLEAN_PRICE, DIRTY_PRICE, COUPON_FREQUENCY, EXPIRE_DATE_OVRD, YIELD_CURVE


def default_arguments(f):
    """ Decorator to set default arguments for :py:class:`_BaseBond` and subclasses.

    Parameters
    ----------
    f: method
        A method to be increased with default arguments.

    Returns
    -------
    function
        `f`, increased with default arguments.

    Note
    ----

    +----------------------------+------------------------------------------+
    | Missing Attribute(s)       | Default Value(s)                         |
    +============================+==========================================+
    | date and quote             | dates and quotes in                      |
    |                            | self.quotes.ts_values                    |
    +----------------------------+------------------------------------------+
    | date                       | dates in self.quotes.ts_values           |
    +----------------------------+------------------------------------------+
    | quote                      | corresponding quotes at passed dates     |
    +----------------------------+------------------------------------------+
    | day_counter                | self.day_counter                         |
    +----------------------------+------------------------------------------+
    | compounding                | self.yield_quote_compounding             |
    +----------------------------+------------------------------------------+
    | frequency                  | self.yield_quote_frequency               |
    +----------------------------+------------------------------------------+
    | settlement_days            | self.settlement_days                     |
    +----------------------------+------------------------------------------+

    """
    @wraps(f)
    def new_f(self, **kwargs):
        if kwargs.get('day_counter', None) is None:
            kwargs['day_counter'] = getattr(self, 'day_counter')
        if kwargs.get('compounding', None) is None:
            kwargs['compounding'] = getattr(self, 'yield_quote_compounding')
        if kwargs.get('frequency', None) is None:
            kwargs['frequency'] = getattr(self, 'yield_quote_frequency')
        if kwargs.get('settlement_days', None) is None:
            kwargs['settlement_days'] = getattr(self, 'settlement_days')
        if kwargs.get('last', None) is None:
            kwargs['last'] = False

        # If last True, use last available date and value for yield calculation.
        if kwargs.get('last', None) is True:
            kwargs['date'] = getattr(self, 'quotes').ts_values.last_valid_index()
            kwargs['quote'] = getattr(self, 'quotes').ts_values[kwargs['date']]
            return f(self, **kwargs)
        # If not, use all the available dates and values.
        if 'date' not in kwargs.keys():
            kwargs['date'] = getattr(self, 'quotes').ts_values.index
            if 'quote' not in kwargs.keys():
                kwargs['quote'] = getattr(self, 'quotes').ts_values.values
        elif 'quote' not in kwargs.keys():
            kwargs['quote'] = getattr(self, 'quotes').get_values(index=kwargs['date'])
        return f(self, **kwargs)
    new_f._decorated_by_default_arguments_ = True
    return new_f


def transform_ts_values(timeseries):
    """ Transform inplace ``ts_values`` of the quotes sub-TimeSeries into clean price format.

    Parameters
    ----------
    timeseries: TimeSeries
        The object whose ``quotes.ts_values`` will be converted to clean prices.
    """
    # Define transformations to make on the ts_values of a TimeSeries depending on the subtype, category, etc. of the
    # bond it defines.
    quote_type = timeseries.get_attribute(QUOTE_TYPE)
    if quote_type == DISCOUNT:
        maturity_date = to_ql_date(to_datetime(timeseries.get_attribute(MATURITY_DATE)))
        face_amount = int(timeseries.get_attribute(FACE_AMOUNT))
        day_counter = to_ql_day_counter(timeseries.get_attribute(DAY_COUNTER))
        calendar = to_ql_calendar(timeseries.get_attribute(CALENDAR))
        settlement_days = int(timeseries.get_attribute(SETTLEMENT_DAYS))
        date_series = pd.Series(index=timeseries.quotes.ts_values.index, data=timeseries.quotes.ts_values.index)
        timeseries.quotes.ts_values = face_amount - timeseries.quotes.ts_values * \
            date_series.apply(lambda x: day_counter.yearFraction(calendar.advance(to_ql_date(x),
                                                                                  settlement_days,
                                                                                  ql.Days), maturity_date))
        timeseries.set_attribute(QUOTE_TYPE, CLEAN_PRICE)


def create_schedule_for_component(call_date, main_bond_schedule, calendar, business_convention, tenor,
                                  date_generation, month_end):
    """Create a schedule object for component bond, adjusting for an eventually irregular last coupon payment date.

    Parameters
    ----------
    call_date: QuantLib.Date
        Maturity of the component bond.
    main_bond_schedule: QuantLib.Schedule
        Cash-flow schedule of the parent bond.
    calendar: QuantLib.Calendar
        Calendar of the parent bond.
    business_convention: int
        Business day convention of the parent bond.
    tenor: QuantLib.Period
        Coupon frequency of the parent bond.
    date_generation: QuantLib.DateGeneration
        Date-generation pattern of the parent bond's schedule.
    month_end: bool
        endOfMonth parameter of the parent bond's schedule.

    Returns
    -------
    QuantLib.Schedule
        A schedule representing the cash-flow dates of the component bond.

    """
    regularity_payment_dates = OrderedDict()  # Just using an OrderedSet to keep order and prevent
    # duplicates.

    for date in main_bond_schedule:
        if date <= call_date:
            regularity_payment_dates[date] = True

    if call_date not in regularity_payment_dates:
        # Check if the call date is an interest payment date. If this is false, the last coupon accrual
        # period is irregular.
        regularity_payment_dates[call_date] = False

    payment_dates = list(regularity_payment_dates.keys())
    regular_periods = list(regularity_payment_dates.values())[1:]
    return ql.Schedule(payment_dates, calendar, business_convention, business_convention, tenor, date_generation,
                       month_end, regular_periods)


def create_call_component(call_date, call_price, main_bond_schedule, calendar, business_convention, tenor,
                          date_generation, month_end, settlement_days, face_amount, coupons, day_counter, issue_date):
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
    coupons: list-like of scalars
        Coupon rates of the parent bond.
    day_counter: QuantLib.DayCounter
        DayCounter of the parent bond.
    issue_date: QuantLib.Date
        Issue date of the parent bond.

    Returns
    -------
    QuantLib.FixedRateBond, QuantLib.ZeroCouponBond
        Bond representing a component of the parent bond, with the passed call date and price.

    Warning
    -------
    This functions currently creates correct components only for FixedRateBond and CallableFixedRateBond with fixed
    coupon rates (i.e. the ``coupons`` list contains a single scalar).

    """
    schedule = create_schedule_for_component(call_date, main_bond_schedule, calendar, business_convention, tenor,
                                             date_generation, month_end)
    if len(schedule) <= 1:  # Means that the bond pays only principal, i.e., should be treated as a zero-coupon bond.
        maturity_date = schedule[0]
        return ql.ZeroCouponBond(settlement_days, calendar, face_amount, maturity_date, business_convention, call_price,
                                 issue_date)
    return ql.FixedRateBond(settlement_days, face_amount, schedule, coupons, day_counter, business_convention,
                            call_price, issue_date)


class _BaseBond(Instrument):
    """ Base class for bonds.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        The TimeSeries representing the bond.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries=timeseries)
        # If quotes are in discount format, just convert them to clean prices.
        transform_ts_values(self)
        # The bond_components attribute holds a {maturity: FixedRateBond} dict, with a FixedRateBond for each
        # possible maturity of the represented bond. This is necessary for calculation of yield to worst.
        self.bond_components = OrderedDict()  # Filled later by the child class.

        self.bond_type = self.ts_attributes[BOND_TYPE]
        self.quote_type = self.ts_attributes[QUOTE_TYPE]
        self.currency = self.ts_attributes[CURRENCY]
        self.yield_quote_compounding = to_ql_compounding(self.ts_attributes[YIELD_QUOTE_COMPOUNDING])
        self.yield_quote_frequency = to_ql_frequency(self.ts_attributes[YIELD_QUOTE_FREQUENCY])

        self.issue_date = to_ql_date(to_datetime(self.ts_attributes[ISSUE_DATE]))
        self.first_accrual_date = to_ql_date(to_datetime(self.ts_attributes[FIRST_ACCRUAL_DATE]))
        self.maturity_date = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.coupon_frequency = ql.Period(to_ql_frequency(self.ts_attributes[COUPON_FREQUENCY]))
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.date_generation = to_ql_date_generation(self.ts_attributes[DATE_GENERATION])
        self.month_end = False  # TODO: Add support for this variable.
        self.expire_date = self.maturity_date
        try:
            expire_date_ovrd = self.ts_attributes[EXPIRE_DATE_OVRD]
            if expire_date_ovrd:
                self.expire_date = to_ql_date(to_datetime(expire_date_ovrd))
        except KeyError:
            pass
        self.schedule = ql.Schedule(self.first_accrual_date, self.maturity_date, self.coupon_frequency, self.calendar,
                                    self.business_convention, self.business_convention, self.date_generation,
                                    self.month_end)

        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.face_amount = float(self.ts_attributes[FACE_AMOUNT])
        self.coupons = [float(self.ts_attributes[COUPONS])]
        # TODO: Make it possible to have a list of different coupons.
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.redemption = float(self.ts_attributes[REDEMPTION])
        # rate Helpers
        self._clean_price = OrderedDict()
        # Bond with coupons
        self._bond_rate_helper = OrderedDict()
        # Zero Coupon bond
        self._zero_coupon_rate_helper = OrderedDict()
        self.bond = None  # Assigned later by the child bond class.
        self._bond_components_backup = None

        '''
        ######################################################################
        Each bond subclass continues initiation from here.

        The subclass should at least:
         1. Assign a QuantLib bond instance to self.bond.
         2. Fill self.bond_components.
         3. Use the following code to make a backup of self.bond_components:
            self._bond_components_backup = self.bond_components.copy()
        ######################################################################
        '''

    def _create_call_component(self, to_date, redemption=None):
        """Create a QuantLib bond with the same attributes as the original bond but different maturity and redemption.

        Parameters
        ----------
        to_date: QuantLib.Date
            Maturity date of the new bond.
        redemption: scalar
            Redemption value of the new bond.

        Returns
        -------
        QuantLib.Bond
            The new bond.
        """
        to_date = to_ql_date(to_date)
        if to_date in self.bond_components and redemption is None:
            return self.bond_components[to_date]
        elif redemption is None:
            redemption = self.implied_redemption(redemption_date=to_date)
        return create_call_component(to_date, redemption, self.schedule, self.calendar,
                                     self.business_convention, self.coupon_frequency,
                                     self.date_generation, self.month_end,
                                     self.settlement_days, self.face_amount, self.coupons,
                                     self.day_counter, self.issue_date)

    def _insert_bond_component(self, date, bond_component):
        """Insert a bond component in the bond_components dictionary.

        Parameters
        ----------
        date: QuantLib.Date
            Maturity date of the component.
        bond_component: QuantLib.Bond
            The component.
        """
        self.bond_components[date] = bond_component

    def _restore_bond_components(self):
        """Restore the bond_components dictionary to its starting state.

        """
        self.bond_components = self._bond_components_backup.copy()

    def implied_redemption(self, redemption_date):
        """
        Parameters
        ----------
        redemption_date: QuantLib.Date
            The redemption date.

        Returns
        -------
        scalar
            The redemption value of the bond if it was redeemed at `redemption_date`.
        """
        redemption_date = to_ql_date(redemption_date)
        if redemption_date == self.maturity_date:
            return self.redemption
        if self.is_expired(redemption_date):
            return np.nan
        if redemption_date < list(self.bond_components.keys())[0]:
            return self.redemption
        last_call_date = find_le(list(self.bond_components.keys()), redemption_date)
        return self.call_schedule.ts_values.loc[to_datetime(last_call_date)]

    def security(self, date, maturity=None, yield_curve_time_series=None, *args, **kwargs):
        """
        Parameters
        ----------
        date: Date-Like
            The evaluation date
        maturity: QuantLib.Date
            The maturity date or call date to retrieve from self.bond_components
        yield_curve_time_series: :py:class:`YieldCurveTimeSeries`
            The yield curve used for discounting the bond to retrieve the NPV
        Returns
        -------
        The python object representing a Bond
        """

        date = to_ql_date(date)
        if maturity is None:
            maturity = self.maturity_date
        yield_curve = yield_curve_time_series.yield_curve(date=date)
        self.bond_components[maturity].setPricingEngine(ql.DiscountingBondEngine(yield_curve))
        return self.bond_components[maturity]

    def is_expired(self, date, *args, **kwargs):
        """
        Parameters
        ----------
        date: QuantLib.Date
            The date.

        Returns
        -------
        bool
            True if the bond is past maturity date, False otherwise.
        """
        date = to_ql_date(date)
        if date >= self.expire_date:
            return True
        elif date >= self.maturity_date:
            return True
        return False

    def maturity(self, *args, **kwargs):
        """
        Returns
        -------
        QuantLib.Date
            Maturity date of the bond.
        """
        return self.maturity_date

    def set_pricing_engine(self, pricing_engine):
        """Set pricing engine of the QuantLib bond object.

        Parameters
        ----------
        pricing_engine: QuantLib.PricingEngine
        """
        self.bond.setPricingEngine(pricing_engine)

    def set_coupon_rate_helper(self):
        """Set Rate Helper if None has been defined yet

        Returns
        -------
        QuantLib.RateHelper
        """
        self._clean_price[self.maturity_date] = ql.SimpleQuote(100)
        self._bond_rate_helper[self.maturity_date] = ql.FixedRateBondHelper(ql.QuoteHandle(
            self._clean_price[self.maturity_date]), self.settlement_days, self.face_amount, self.schedule, self.coupons,
            self.day_counter, self.business_convention, self.redemption, self.issue_date)

    def set_zero_rate_helper(self):
        """Set Rate Helper if None has been defined yet

        Returns
        -------
        QuantLib.RateHelper
        """
        self._clean_price[self.maturity_date] = ql.SimpleQuote(100)
        self._zero_coupon_rate_helper[self.maturity_date] = ql.FixedRateBondHelper(ql.QuoteHandle(
            self._clean_price[self.maturity_date]), self.settlement_days, self.face_amount, self.schedule, [0],
            self.day_counter, self.business_convention, self.redemption, self.issue_date)

    def rate_helper(self, date, last_available=True, yield_type='ytm', curve_type='zero', max_inactive_days=3,
                    reference_date_for_worst_date=None, **kwargs):
        """
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
            elif yield_type == 'ytw_rolling_call':
                maturity = self.worst_date_rolling_call(date=reference_date_for_worst_date,
                                                        quote=reference_quote_for_worst_date)
            else:
                maturity = self.maturity_date
            clean_price = self.clean_price(date=date, quote=quote)
            try:
                self._clean_price[maturity].setValue(clean_price)
                return self._bond_rate_helper[maturity]
            except KeyError:
                self.set_coupon_rate_helper()
                self._clean_price[maturity].setValue(clean_price)
                return self._bond_rate_helper[maturity]

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
            time = self.day_counter.yearFraction(date, maturity)
            discount = ql.InterestRate(yld, self.day_counter, self.yield_quote_compounding,
                                       self.yield_quote_frequency).discountFactor(time)
            try:
                self._clean_price[maturity].setValue(discount * 100)
                return self._zero_coupon_rate_helper[maturity]
            except KeyError:
                self.set_zero_rate_helper()
                self._clean_price[maturity].setValue(discount * 100)
                return self._zero_coupon_rate_helper[maturity]
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
        if date >= self.maturity_date:
            return np.nan
        return self.bond.accruedAmount(date)

    @default_arguments
    @conditional_vectorize('date')
    def cash_flow_to_date(self, start_date, date, **kwargs):
        """
        Parameters
        ----------
        start_date: QuantLib.Date
            The start date.
        date: QuantLib.Date, optional, (c-vectorized)
            The last date of the computation.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
           List of Tuples with ex-date, pay-date and the amount paid between the period.


        """
        start_date = to_ql_date(start_date)
        date = to_ql_date(date)
        return list((cf.date(), cf.date(), cf.amount() / self.face_amount) for cf in self.bond.cashflows()
                    if start_date <= cf.date() <= date)

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
        start_date = to_ql_date(start_date)
        date = to_ql_date(date)
        return sum(cf.amount() for cf in self.bond.cashflows() if start_date <= cf.date() <= date) / self.face_amount

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def clean_price(self, last, quote, date, settlement_days, quote_type=None, yield_curve=None, **kwargs):
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
        quote_type: str, optional
            The quote type for calculation ex: CLEAN_PRICE, DIRTY_PRICE, YIELD
            Default: None
        yield_curve: :py:func:`YieldCurveTimeSeries`
            The yield curve object to be used to get the price from yield curve.

        Returns
        -------
        scalar
            The bond's clean price.

        """
        if quote_type is None:
            quote_type = self.quote_type

        if quote_type == CLEAN_PRICE:
            return quote
        elif quote_type == DIRTY_PRICE:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return quote - self.accrued_interest(last=last, date=settlement_date, **kwargs)
        elif quote_type == DISCOUNT:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            return self.face_amount - quote * 100 * self.day_counter.yearFraction(settlement_date, self.maturity_date)
        elif quote_type == YIELD:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return self.bond.cleanPrice(quote, self.day_counter, self.yield_quote_compounding,
                                        self.yield_quote_frequency, settlement_date)
        elif quote_type == YIELD_CURVE:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            yield_curve = yield_curve.yield_curve(date=date)
            if self.is_expired(settlement_date):
                return np.nan
            return ql.BondFunctions.cleanPrice(self.bond, yield_curve, settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def dirty_price(self, last, quote, date, settlement_days, quote_type=None, yield_curve=None, **kwargs):
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
        quote_type: str, optional
            The quote type for calculation ex: CLEAN_PRICE, DIRTY_PRICE, YIELD
            Default: None
        yield_curve: :py:func:`YieldCurveTimeSeries`
            The yield curve object to be used to get the price from yield curve.

        Returns
        -------
        scalar
            The bond's dirty price.
        """

        if quote_type is None:
            quote_type = self.quote_type

        if quote_type == CLEAN_PRICE:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return quote + self.accrued_interest(last=last, date=settlement_date, **kwargs)
        elif quote_type == DIRTY_PRICE:
            return quote
        elif quote_type == DISCOUNT:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            return self.face_amount - quote * 100 * self.day_counter.yearFraction(settlement_date, self.maturity_date)
        elif quote_type == YIELD:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            if self.is_expired(settlement_date):
                return np.nan
            return self.bond.dirtyPrice(quote, self.day_counter, self.yield_quote_compounding,
                                        self.yield_quote_frequency, settlement_date)
        elif quote_type == YIELD_CURVE:
            date = to_ql_date(date)
            settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                    self.business_convention)
            yield_curve = yield_curve.yield_curve(date=date)
            if self.is_expired(settlement_date):
                return np.nan
            clean_price = ql.BondFunctions.cleanPrice(self.bond, yield_curve, settlement_date)
            return clean_price + self.accrued_interest(last=last, date=settlement_date, **kwargs)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def value(self, last, quote, date, last_available=False, clean_price=False, **kwargs):
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
        clean_price: bool, optional
            Use only the bond clean price to calculate the position value.

        Returns
        -------
        scalar
            The dirty value of the bond as of `date`.
        """
        date = to_datetime(date)
        if last_available:
            quote = self.quotes.get_values(index=date, last_available=last_available)
            if clean_price:
                return self.clean_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount
            else:
                return self.dirty_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount
        if date > self.quotes.ts_values.last_valid_index():
            return np.nan
        if date < self.quotes.ts_values.first_valid_index():
            return np.nan

        if clean_price:
            return self.clean_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount
        else:
            return self.dirty_price(last=last, quote=quote, date=date, settlement_days=0) / self.face_amount

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def risk_value(self, last, quote, date, last_available=False, clean_price=False, **kwargs):
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
        clean_price: bool, optional
            Use only the bond clean price to calculate the position value.

        Returns
        -------
        scalar
            The dirty value of the bond as of `date`.
        """
        return self.value(last=last, quote=quote, date=date, last_available=last_available, clean_price=clean_price,
                          **kwargs)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, last_available=False, **kwargs):
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
        last_available: bool, optional
            Whether to use last available data.
            Default: False.

        Returns
        -------
        scalar, None
            Performance of a unit of the bond.
        """
        first_available_date = self.quotes.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = self.quotes.get_values(index=start_date)
        if date < start_date:
            return np.nan
        start_value = self.value(quote=start_quote, date=start_date)
        value = self.value(quote=quote, date=date)
        paid_interest = self.cash_to_date(start_date=start_date, date=date)
        return (value + paid_interest) / start_value - 1

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytm(self, last, quote, date, day_counter, compounding, frequency, settlement_days, quote_type=None, **kwargs):
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
        quote_type: str, optional
            The quote type for calculation ex: CLEAN_PRICE, DIRTY_PRICE, YIELD
            Default: None

        Returns
        -------
        scalar
            The bond's yield to maturity.
        """
        bond = kwargs.get('bond', self.bond)  # Useful to pass bonds other than self as arguments.
        date = to_ql_date(date)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan
        ql.Settings.instance().evaluationDate = date
        if quote_type is None:
            quote_type = self.quote_type
        if quote_type == CLEAN_PRICE:
            return bond.bondYield(quote, day_counter, compounding, frequency, settlement_date)
        elif quote_type == DIRTY_PRICE:
            # TODO: This part needs testing.
            clean_quote = quote - self.accrued_interest(last=last, date=settlement_date, **kwargs)
            return bond.bondYield(clean_quote, day_counter, compounding, frequency, settlement_date)
        elif quote_type == YIELD:
            # TODO: This part needs testing.
            interest_rate = ql.InterestRate(quote, self.day_counter, self.yield_quote_compounding,
                                            self.yield_quote_frequency)
            return interest_rate.equivalentRate(compounding, frequency, 1)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytw_and_worst_date(self, last, quote, date, day_counter, compounding, frequency, settlement_days,
                           quote_type=None, **kwargs):
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
        quote_type: str, optional
            The quote type for calculation ex: CLEAN_PRICE, DIRTY_PRICE, YIELD
            Default: None

        Returns
        -------
        tuple (scalar, QuantLib.Date)
            The bond's yield to worst and worst date.
        """
        date = to_ql_date(date)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan, np.nan
        return min(((self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                              frequency=frequency, settlement_days=settlement_days, bond=bond, quote_type=quote_type,
                              **kwargs), key)
                    for key, bond in self.bond_components.items() if key > settlement_date), key=itemgetter(0))

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def ytw(self, last, quote, date, day_counter, compounding, frequency, settlement_days, quote_type=None, **kwargs):
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
        quote_type: str, optional
            The quote type for calculation ex: CLEAN_PRICE, DIRTY_PRICE, YIELD
            Default: None

        Returns
        -------
        scalar
            The bond's yield to worst.
        """
        date = to_ql_date(date)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            # input("Returning nan because its expired date: {0}, settlement {1}".format(date, settlement_date))
            return np.nan
        return min((self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                             frequency=frequency, settlement_days=settlement_days, bond=bond,
                             quote_type=quote_type, **kwargs)
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
        if date < list(self.bond_components.keys())[0]:
            # If date is before the first call dates, rolling calls are NOT possible.
            return self.ytw_and_worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                           compounding=compounding, frequency=frequency,
                                           settlement_days=settlement_days, **kwargs)
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
        to_date = to_ql_date(to_date)
        self.bond_components = {to_date: self._create_call_component(to_date=to_date)}
        result = self.ytw(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                          frequency=frequency, settlement_days=settlement_days, **kwargs)
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
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(date) or self.is_expired(settlement_date):
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
        settlement_date = self.calendar.advance(date, ql.Period(self.settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(date) or self.is_expired(settlement_date):
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
        """
        date = to_ql_date(date)
        ytm = self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                       frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                       **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        if self.yield_quote_compounding == ql.Simple:
            duration_type = ql.Duration.Simple
        return ql.BondFunctions_duration(self.bond, ytm, self.day_counter, self.yield_quote_compounding,
                                         self.yield_quote_frequency, duration_type, settlement_date)

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
            Bond's duration to worst.
        """
        date = to_ql_date(date)
        ytw, worst_date = self.ytw_and_worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                                  compounding=compounding, frequency=frequency,
                                                  settlement_days=settlement_days,
                                                  bypass_set_floating_rate_index=True, **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        if self.yield_quote_compounding == ql.Simple:
            duration_type = ql.Duration.Simple
        return ql.BondFunctions_duration(self.bond_components[worst_date], ytw, self.day_counter,
                                         self.yield_quote_compounding, self.yield_quote_frequency,
                                         duration_type, settlement_date)

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
            Bond's duration to worst, considering rolling call possibility.
        """
        date = to_ql_date(date)
        ytw, worst_date = self.ytw_and_worst_date_rolling_call(last=last, quote=quote, date=date,
                                                               day_counter=day_counter, compounding=compounding,
                                                               frequency=frequency, settlement_days=settlement_days,
                                                               bypass_set_floating_rate_index=True, **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        bond = self._create_call_component(to_date=worst_date)
        if self.yield_quote_compounding == ql.Simple:
            duration_type = ql.Duration.Simple
        return ql.BondFunctions_duration(bond, ytw, self.day_counter, self.yield_quote_compounding,
                                         self.yield_quote_frequency, duration_type, settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def convexity_to_mat(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
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
        date = to_ql_date(date)
        ytm = self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                       frequency=frequency, settlement_day=settlement_days, bypass_set_floating_rate_index=True,
                       **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        return ql.BondFunctions_convexity(self.bond, ytm, self.day_counter, self.yield_quote_compounding,
                                          self.yield_quote_frequency, settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def convexity_to_worst(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
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
        date = to_ql_date(date)
        ytw, worst_date = self.ytw_and_worst_date(last=last, quote=quote, date=date, day_counter=day_counter,
                                                  compounding=compounding, frequency=frequency,
                                                  settlement_days=settlement_days,
                                                  bypass_set_floating_rate_index=True, **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        return ql.BondFunctions_convexity(self.bond_components[worst_date], ytw, day_counter, compounding, frequency,
                                          settlement_date)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def convexity_to_worst_rolling_call(self, last, quote, date, day_counter, compounding, frequency,
                                        settlement_days, **kwargs):
        """
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
            Bond's duration to worst, considering rolling calls.
        """
        date = to_ql_date(date)
        ytw, worst_date = self.ytw_and_worst_date_rolling_call(last=last, quote=quote, date=date,
                                                               day_counter=day_counter, compounding=compounding,
                                                               frequency=frequency, settlement_days=settlement_days,
                                                               bypass_set_floating_rate_index=True,
                                                               **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        bond = self._create_call_component(to_date=worst_date)
        return ql.BondFunctions_convexity(bond, ytw, day_counter, compounding, frequency, settlement_date)

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
        date = to_ql_date(date)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days),
                                                self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        if yield_curve_timeseries.calendar.isHoliday(date):
            yield_date = yield_curve_timeseries.calendar.adjust(date, ql.Preceding)
            yield_curve = yield_curve_timeseries.implied_term_structure(date=yield_date, future_date=date)
        else:
            yield_curve = yield_curve_timeseries.yield_curve(date=date)

        ql.Settings.instance().evaluationDate = date
        if self.quote_type == CLEAN_PRICE:
            pass
        elif self.quote_type == DIRTY_PRICE:
            quote = quote - self.accrued_interest(last=last, date=settlement_date, **kwargs)
        elif self.quote_type == YIELD:
            quote = self.clean_price_from_ytm(last=last, quote=quote, date=date, day_counter=day_counter,
                                              compounding=compounding, frequency=frequency,
                                              bypass_set_floating_rate_index=True,
                                              settlement_days=settlement_days)
        return ql.BondFunctions_zSpread(self.bond, quote, yield_curve, day_counter, compounding, frequency,
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
        yield_curve_timeseries: :py:class:`YieldCurveTimeSeries`
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
                                   frequency=frequency, settlement_days=settlement_days,
                                   bypass_set_floating_rate_index=True,  **kwargs)
