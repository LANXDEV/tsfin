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
A class for modelling interest rate swaps.
"""
import numpy as np
import QuantLib as ql
from functools import wraps
from tsfin.instruments.interest_rates.base_interest_rate import BaseInterestRate
from tsfin.base import to_ql_date, to_ql_frequency, to_ql_date_generation, conditional_vectorize, \
    to_ql_business_convention, to_ql_calendar, to_ql_day_counter, to_ql_protection_side, to_ql_cds_engine
from tsfin.constants import FREQUENCY, DATE_GENERATION, RECOVERY_RATE, COUPONS, BASE_SPREAD_TAG, CALENDAR, \
    TENOR_PERIOD, BUSINESS_CONVENTION, DAY_COUNTER, FIXING_DAYS, FIRST_ACCRUAL_DATE


def cds_default_values(f):

    @wraps(f)
    def new_f(self, **kwargs):

        try:
            kwargs['date'] = to_ql_date(kwargs['date'])
        except TypeError:
            kwargs['date'] = to_ql_date(kwargs['date'][0])

        date = kwargs['date']

        if kwargs.get('recovery_rate', None) is None:
            kwargs['recovery_rate'] = getattr(self, 'recovery_rate')
        if kwargs.get('maturity', None) is None:
            kwargs['maturity'] = self.maturity(date=date)
        if kwargs.get('day_counter', None) is None:
            kwargs['day_counter'] = getattr(self, 'day_counter')
        if kwargs.get('date_generation', None) is None:
            kwargs['date_generation'] = getattr(self, 'date_generation')
        if kwargs.get('frequency', None) is None:
            kwargs['frequency'] = getattr(self, 'coupon_frequency')
        if kwargs.get('calendar', None) is None:
            kwargs['calendar'] = getattr(self, 'calendar')
        if kwargs.get('business_convention', None) is None:
            kwargs['business_convention'] = getattr(self, 'business_convention')
        if kwargs.get('first_accrual_date', None) is None:
            kwargs['first_accrual_date'] = getattr(self, 'first_accrual_date')
        if kwargs.get('start_date', None) is None:
            kwargs['start_date'] = kwargs['first_accrual_date'] if kwargs['first_accrual_date'] is not None else date
        last_available = kwargs.get('last_available', True)
        if kwargs.get('spread_rate', None) is None:
            kwargs['spread_rate'] = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        kwargs['side'] = to_ql_protection_side(side=kwargs['side']) if kwargs.get('side', None) is not None else\
            ql.Protection.Buyer
        return f(self, **kwargs)

    return new_f


class CDSRate(BaseInterestRate):
    """ Model for rolling interest cds rates (fixed tenor, like the ones quoted in Bloomberg).

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.fixing_days = int(self.ts_attributes[FIXING_DAYS])
        self.date_generation = to_ql_date_generation(self.ts_attributes[DATE_GENERATION])
        self.coupon_frequency = ql.Period(to_ql_frequency(self.ts_attributes[FREQUENCY]))
        self.recovery_rate = float(self.ts_attributes[RECOVERY_RATE])
        self.coupon = float(self.ts_attributes[COUPONS])
        self.base_yield = self.ts_attributes[BASE_SPREAD_TAG]
        try:
            self.first_accrual_date = to_ql_date(self.ts_attributes[FIRST_ACCRUAL_DATE])
        except KeyError:
            self.first_accrual_date = None
        self.month_end = False
        # Rate Helper
        self.helper_rate = ql.SimpleQuote(0)

    @cds_default_values
    def security(self, date, cds_curve_time_series, spread_rate, recovery_rate, first_accrual_date, side, notional,
                 maturity, frequency, calendar, business_convention, date_generation, day_counter, upfront_price=None,
                 engine_name='MID_POINT', last_available=True, *args, **kwargs):

        cds = self.credit_default_swap(first_accrual_date=first_accrual_date, side=side, notional=notional,
                                       spread_rate=spread_rate, maturity=maturity, upfront_price=upfront_price,
                                       frequency=frequency, calendar=calendar, business_convention=business_convention,
                                       date_generation=date_generation, day_counter=day_counter, *args, **kwargs)
        probability_handle = cds_curve_time_series.probability_curve_handle(date=date)
        base_yield_curve_handle = cds_curve_time_series.base_yield_curve_handle(date=date)
        engine = to_ql_cds_engine(engine_name=engine_name)(probability_handle, recovery_rate, base_yield_curve_handle)
        cds.setPricingEngine(engine)
        return cds

    def is_expired(self, date, *args, **kwargs):
        """ Returns False.

        Parameters
        ----------
        date: QuantLib.Date
            The date.

        Returns
        -------
        bool
            Always False.
        """
        return False

    def cds_rate_helper(self, date, base_yield_curve, last_available=True, *args, **kwargs):
        """ Helper for yield curve construction.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        base_yield_curve: YieldCurveTimeSeries.yield_curve_handle
            Yield curve used as base for discounting cash flows
        last_available: bool, optional
            Whether to use last available quotes if missing data.

        Returns
        -------
        QuantLib.RateHelper
            Rate helper for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        self.link_to_term_structure(date=date, yield_curve=base_yield_curve)
        self.helper_rate.setValue(float(rate))
        return ql.SpreadCdsHelper(ql.QuoteHandle(self.helper_rate),
                                  self._tenor,
                                  0,
                                  self.calendar,
                                  self.frequency,
                                  self.business_convention,
                                  self.date_generation,
                                  self.day_counter,
                                  self.recovery_rate,
                                  self.term_structure)

    def credit_default_swap(self, first_accrual_date, side, notional, spread_rate, maturity, frequency, calendar,
                            business_convention, date_generation, day_counter, upfront_price=None, *args, **kwargs):

        """
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The accrual start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param spread_rate: float
            The cds spread rate
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :return: QuantLib.CreditDefaultSwap
        """

        # maybe the user passed a quote multiplied by 100, if so we divide by 100 to be correctly used by the CDS.
        schedule = ql.Schedule(first_accrual_date, maturity, frequency, calendar, business_convention, ql.Unadjusted,
                               date_generation, self.month_end)
        if upfront_price is None:
            return ql.CreditDefaultSwap(side, notional, spread_rate, schedule, business_convention, day_counter)
        else:
            if upfront_price > 1:
                upfront_price = upfront_price/100
            upfront = 1 - upfront_price
            return ql.CreditDefaultSwap(side, notional, upfront, spread_rate, schedule, business_convention,
                                        day_counter)

    @conditional_vectorize('date', 'spread_rate')
    @cds_default_values
    def net_present_value(self, date, spread_rate, cds_curve_time_series, recovery_rate, first_accrual_date, side,
                          notional, maturity, upfront_price, frequency, calendar, business_convention, date_generation,
                          day_counter, engine_name='MID_POINT', *args, **kwargs):
        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param cds_curve_time_series: :py:obj:CDSCurveTimeSeries
            the curve used for the calculation
        :param spread_rate: float
            The cds spread rate
        :param recovery_rate: float, optional
            The CDS recovery rate
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The CDS start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :param engine_name: str
            The name representing a QuantLib cds engine, 'MID_POINT', 'ISDA', 'INTEGRAL'
        :return float
            CDS Net Present Value.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.security(date=date, spread_rate=spread_rate, cds_curve_time_series=cds_curve_time_series,
                            recovery_rate=recovery_rate, first_accrual_date=first_accrual_date, side=side,
                            notional=notional, maturity=maturity, upfront_price=upfront_price, frequency=frequency,
                            calendar=calendar, business_convention=business_convention, date_generation=date_generation,
                            day_counter=day_counter, engine_name=engine_name, *args, **kwargs)
        return cds.NPV()

    @conditional_vectorize('date', 'spread_rate')
    @cds_default_values
    def cds_spread(self, date, spread_rate, cds_curve_time_series, recovery_rate, first_accrual_date, side, notional,
                   maturity, upfront_price, frequency, calendar, business_convention, date_generation, day_counter,
                   engine_name='MID_POINT', *args, **kwargs):
        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param cds_curve_time_series: :py:obj:CDSCurveTimeSeries
            the curve used for the calculation
        :param spread_rate: float
            The cds spread rate
        :param recovery_rate: float, optional
            The CDS recovery rate
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The CDS start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :param engine_name: str
            The name representing a QuantLib cds engine, 'MID_POINT', 'ISDA', 'INTEGRAL'
        :return float
            CDS fair spread
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.security(date=date, spread_rate=spread_rate, cds_curve_time_series=cds_curve_time_series,
                            recovery_rate=recovery_rate, first_accrual_date=first_accrual_date, side=side,
                            notional=notional, maturity=maturity, upfront_price=upfront_price, frequency=frequency,
                            calendar=calendar, business_convention=business_convention, date_generation=date_generation,
                            day_counter=day_counter, engine_name=engine_name, *args, **kwargs)
        return cds.fairSpread()

    @conditional_vectorize('date', 'spread_rate')
    @cds_default_values
    def default_leg_npv(self, date, spread_rate, cds_curve_time_series, recovery_rate, first_accrual_date,
                        side, notional, maturity, upfront_price, frequency, calendar, business_convention,
                        date_generation, day_counter, engine_name='MID_POINT', *args, **kwargs):
        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param cds_curve_time_series: :py:obj:CDSCurveTimeSeries
            the curve used for the calculation
        :param spread_rate: float
            The cds spread rate
        :param recovery_rate: float, optional
            The CDS recovery rate
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The CDS start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :param engine_name: str
            The name representing a QuantLib cds engine, 'MID_POINT', 'ISDA', 'INTEGRAL'
        :return float
            CDS Net Present Value of the default leg
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.security(date=date, spread_rate=spread_rate, cds_curve_time_series=cds_curve_time_series,
                            recovery_rate=recovery_rate, first_accrual_date=first_accrual_date, side=side,
                            notional=notional, maturity=maturity, upfront_price=upfront_price, frequency=frequency,
                            calendar=calendar, business_convention=business_convention, date_generation=date_generation,
                            day_counter=day_counter, engine_name=engine_name, *args, **kwargs)
        return cds.defaultLegNPV()

    @conditional_vectorize('date', 'spread_rate')
    @cds_default_values
    def coupon_leg_npv(self, date, spread_rate, cds_curve_time_series, recovery_rate, first_accrual_date,
                       side, notional, maturity, upfront_price, frequency, calendar, business_convention,
                       date_generation, day_counter, engine_name='MID_POINT', *args, **kwargs):
        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param cds_curve_time_series: :py:obj:CDSCurveTimeSeries
            the curve used for the calculation
        :param spread_rate: float
            The cds spread rate
        :param recovery_rate: float, optional
            The CDS recovery rate
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The CDS start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :param engine_name: str
            The name representing a QuantLib cds engine, 'MID_POINT', 'ISDA', 'INTEGRAL'
        :return float
            CDS Net Present Value of the coupon leg
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.security(date=date, spread_rate=spread_rate, cds_curve_time_series=cds_curve_time_series,
                            recovery_rate=recovery_rate, first_accrual_date=first_accrual_date, side=side,
                            notional=notional, maturity=maturity, upfront_price=upfront_price, frequency=frequency,
                            calendar=calendar, business_convention=business_convention, date_generation=date_generation,
                            day_counter=day_counter, engine_name=engine_name, *args, **kwargs)
        return cds.couponLegNPV()

    @conditional_vectorize('date', 'spread_rate')
    @cds_default_values
    def upfront_npv(self, date, spread_rate, cds_curve_time_series, recovery_rate, first_accrual_date,
                    side, notional, maturity, upfront_price, frequency, calendar, business_convention,
                    date_generation, day_counter, engine_name='MID_POINT', *args, **kwargs):
        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param cds_curve_time_series: :py:obj:CDSCurveTimeSeries
            the curve used for the calculation
        :param spread_rate: float
            The cds spread rate
        :param recovery_rate: float, optional
            The CDS recovery rate
        :param first_accrual_date: pd.Datetime or QuantLib.Date
            The CDS start date
        :param side: QuantLib.Protection
            The Credit Default deal side
        :param notional: float
            Size of the contract
        :param maturity: QuantLib.Date
            The CDS maturity date
        :param upfront_price: float
            The par value of the upfront payment.
        :param frequency: QuantLib.Frequency / QuantLib.Period
            The CDS coupon frequency
        :param calendar: QuantLib.Calendar
            The CDS calendar
        :param business_convention: QuantLib.BusinessConvention
            The CDS Business day convention
        :param date_generation: QuantLib.DateGeneration
            The CDS schedule date generation rule
        :param day_counter: QuantLib.DayCounter
            The CDS day counting rule
        :param engine_name: str
            The name representing a QuantLib cds engine, 'MID_POINT', 'ISDA', 'INTEGRAL'
        :return float
            CDS Net Present Value of the upfront price
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.security(date=date, spread_rate=spread_rate, cds_curve_time_series=cds_curve_time_series,
                            recovery_rate=recovery_rate, first_accrual_date=first_accrual_date, side=side,
                            notional=notional, maturity=maturity, upfront_price=upfront_price, frequency=frequency,
                            calendar=calendar, business_convention=business_convention, date_generation=date_generation,
                            day_counter=day_counter, engine_name=engine_name, *args, **kwargs)
        return cds.upfrontNPV()
