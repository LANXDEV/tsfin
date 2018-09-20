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
FraDDI class used to model DDI FRA contracts traded at the B3 Exchange (Brazil).

TODO: Propose implementation of this class in QuantLib.
"""
from functools import wraps
import numpy as np
import QuantLib as ql
from tsfin.base import Instrument, conditional_vectorize, to_datetime, to_ql_date, to_ql_frequency, to_ql_calendar, \
    to_ql_compounding, to_ql_day_counter, to_ql_business_convention
from tsfin.constants import CALENDAR, TENOR_PERIOD, MATURITY_DATE, DAY_COUNTER, COMPOUNDING, FREQUENCY, \
    BUSINESS_CONVENTION, FIXING_DAYS


def default_arguments(f):
    """Decorator to set default arguments for :py:class:`FraDDI`.

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

    """
    @wraps(f)
    def new_f(self, **kwargs):
        if kwargs.get('last', None) is True:
            quotes = self.quotes
            kwargs['date'] = quotes.ts_values.last_valid_index()
            kwargs['quote'] = quotes.ts_values[kwargs['dates']]
            return f(self, **kwargs)
        if 'date' not in kwargs.keys():
            kwargs['date'] = getattr(self, 'quotes').ts_values.index
            if 'quote' not in kwargs.keys():
                kwargs['quote'] = getattr(self, 'quotes').ts_values.values
        elif 'quote' not in kwargs.keys():
            kwargs['quote'] = getattr(self, 'quotes').get_values(index=kwargs['date'])
        return f(self, **kwargs)
    new_f._decorated_by_default_arguments_ = True
    return new_f


class FraDDI(Instrument):
    """ Class to model FRA DDI instruments (`cupom cambial` FRAs)

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        The TimeSeries representing the FraDDI.
    first_cc: :py:obj:`CupomCambial`
        Object representing the (rolling) first cupom cambial.
        TODO: Remove necessity of this parameter by adding a rate helper capable of handling these types of fras.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, first_cc):
        super().__init__(timeseries)
        self.first_cc = first_cc
        self.calendar = to_ql_calendar(self.attributes[CALENDAR])
        try:
            self._tenor = ql.PeriodParser.parse(self.attributes[TENOR_PERIOD])
        except KeyError:
            # If the deposit rate has no tenor, it must have a maturity.
            self._maturity = to_ql_date(to_datetime(self.attributes[MATURITY_DATE]))
        self.day_counter = to_ql_day_counter(self.attributes[DAY_COUNTER])
        self.compounding = to_ql_compounding(self.attributes[COMPOUNDING])
        self.frequency = to_ql_frequency(self.attributes[FREQUENCY])
        self.business_convention = to_ql_business_convention(self.attributes[BUSINESS_CONVENTION])
        self.fixing_days = int(self.attributes[FIXING_DAYS])

    def reference_date(self, date):
        """ Check maturity of the shortest leg of the contract.

        Parameters
        ----------
        date: QuantLib.Date
            Calculation date.

        Returns
        -------
        QuantLib.Date
            Maturity date of the shortest leg of the contract.
        """
        date = to_ql_date(date)
        calendar = self.calendar
        return calendar.advance(calendar.endOfMonth(calendar.advance(date, 2, ql.Days)), 1, ql.Days)

    def is_expired(self, date, *args, **kwargs):
        """ Check if instrument is expired.

        Parameters
        ----------
        date: QuantLib.Date
            Calculation date.

        Returns
        -------
        bool
            True if the instrument is past maturity date, False otherwise.
        """
        calendar = self.calendar
        maturity_date = self.maturity
        reference_date = self.reference_date(date)
        last_possible_reference_date = calendar.advance(calendar.endOfMonth(calendar.advance(maturity_date, -2,
                                                                                             ql.Months)), 1, ql.Days)
        if reference_date > last_possible_reference_date:
            return True
        return False

    @conditional_vectorize('date')
    def value(*args, **kwargs):
        raise NotImplementedError("This method is not yet implemented.")

    @conditional_vectorize('date')
    def tenor(self, date, *args, **kwargs):
        """ Tenor period to maturity.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        QuantLib.Period
            Period to maturity.

        """
        date = to_ql_date(date)
        if self.is_expired(date):
            raise ValueError("The requested date is equal or higher than the instrument's maturity: {}".format(
                self.name))
        reference_date = self.reference_date(date)
        maturity = self.maturity
        days = self.calendar.businessDaysBetween(reference_date, maturity)
        return ql.Period(days, ql.Days)

    @default_arguments
    @conditional_vectorize('date')
    def performance(self, start_date=None, date=None, **kwargs):
        raise NotImplementedError("performance method is not implemented for DDI instruments yet.")

    def rate_helper(self, date, last_available=True, *args, **kwargs):
        """ Get a Rate helper object for this instrument.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool
            Whether to use last available quotes if missing data.

        Returns
        -------
        QuantLib.RateHelper
            Object used to build yield curves.
        """
        # Returns None if impossible to obtain a rate helper from this time series.
        if self.is_expired(date):
            return None
        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        first_cc_rate = self.first_cc.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        if np.isnan(first_cc_rate):
            return None
        date = to_ql_date(date)
        reference_date = self.reference_date(date)
        tenor = self.tenor(date)
        maturity_date = self.calendar.advance(reference_date, tenor)
        day_counter = self.day_counter
        implied_rate = (((1 + first_cc_rate * day_counter.yearFraction(date, reference_date)) *
                        (1 + rate * day_counter.yearFraction(reference_date, maturity_date)) - 1)
                        / day_counter.yearFraction(date, maturity_date))
        tenor = ql.Period(self.calendar.businessDaysBetween(date, maturity_date), ql.Days)
        return ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(implied_rate)), tenor, 0, self.calendar,
                                    ql.Unadjusted, False, self.day_counter)
