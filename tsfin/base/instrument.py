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
Base class for all other instrument classes.
"""
from functools import wraps
import numpy as np
import QuantLib as ql
from tsio.tools import to_datetime, at_index
from tsfin.constants import QUOTES, TENOR_PERIOD, MATURITY_DATE
from tsfin.base.basetools import conditional_vectorize
from tsfin.base.qlconverters import to_ql_date


def default_arguments(f):
    """ Decorator to set default arguments for :py:class:`TimeSeries` methods.

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
    | quote                      | corresponding quotes at passed date      |
    +----------------------------+------------------------------------------+
    """
    @wraps(f)
    def new_f(self, **kwargs):
        quotes = self.quotes
        if 'last' not in kwargs.keys():
            kwargs['last'] = False
        # If last True, use last available date and value for calculation.
        if kwargs.get('last', None) is True:
            kwargs['date'] = quotes.ts_values.last_valid_index()
            kwargs['quote'] = at_index(quotes.ts_values, index=kwargs['date'])
            return f(self, **kwargs)
        # If not, use all the available dates and values.
        if 'date' not in kwargs.keys():
            kwargs['date'] = quotes.ts_values.index
            if 'quote' not in kwargs.keys():
                kwargs['quote'] = at_index(quotes.ts_values, index=kwargs['date'])
        elif 'quote' not in kwargs.keys():
            kwargs['quote'] = at_index(quotes.ts_values, index=kwargs['date'])
        return f(self, **kwargs)
    new_f._decorated_by_default_arguments_ = True
    return new_f


class Instrument:
    """ Base for classes representing financial instruments.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`

    """

    def __init__(self, timeseries):
        self.timeseries = timeseries

    def __getattr__(self, attr):
        if attr == 'quotes':
            # If one asks for the quotes, try to get the QUOTES component of self.timeseries.
            # Override the value in the variable QUOTES if you want another value written in the database.
            attr = QUOTES
            try:
                return getattr(self.timeseries, attr)
            except AttributeError:
                # If there is no QUOTES component, assume that the quotes are in the ts_values of the self.timeseries.
                return self.timeseries
        try:
            return getattr(self.timeseries, attr)
        except AttributeError:
            raise AttributeError("The {0} {1} has no attribute '{2}'.".format(type(self).__name__,
                                                                              self.timeseries.ts_name, attr))

    def is_expired(self, date, *args, **kwargs):
        """
        Parameters
        ----------
        date: datetime-like

        Returns
        -------
        bool
            True if the instrument is expired or matured, False otherwise.
        """
        return False

    def maturity(self, date, *args, **kwargs):
        """
        Parameters
        ----------
        date: datetime-like

        Returns
        -------
        QuantLib.Date, None
             Date representing the maturity or expiry of the instrument. Returns None if there is no maturity.
        """
        return None

    def tenor(self, date, *args, **kwargs):
        """
        Parameters
        ----------
        date: datetime-like

        Returns
        -------
        QuantLib.Period, None
            Period representing the time to maturity or expiry of the instrument. Returns None if there is no tenor.
        """
        try:
            return ql.PeriodParser.parse(self.get_attribute(TENOR_PERIOD))
        except:
            try:
                try:
                    tenor = float(self.get_attribute(TENOR_PERIOD))
                    if tenor < 1:
                        return ql.Period(int(self.get_attribute(TENOR_PERIOD)*252), ql.Days)
                        # TODO: Amend these approx. tenors
                    return ql.Period(int(tenor), ql.Years)
                except:
                    date = to_ql_date(date)
                    maturity = to_ql_date(to_datetime(self.get_attribute(MATURITY_DATE)))
                    if date >= maturity:
                        return None
                    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
                    # TODO: This will have to be parametrized afterwards.
                    return ql.Period(calendar.businessDaysBetween(date, maturity), ql.Days)
            except:
                # Impossible to find a tenor or maturity for this TimeSeries.
                return None

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
           List of Tuples with the amount paid between the period.


        """
        return 0

    @conditional_vectorize('date')
    def cash_to_date(self, start_date, date, *args, **kwargs):
        """
        Parameters
        ----------
        start_date: datetime-like
        date: datetime-like, optional
            Defaults to last available date in ``self.price.ts_values``.

        Returns
        -------
        scalar
            Cash amount paid by a unit of the instrument between `start_date` and `date`.
        """
        return 0

    @default_arguments
    @conditional_vectorize('date', 'quote')
    def value(self, last, date, quote, last_available=False, *args, **kwargs):
        """Try to deduce dirty value for a unit of the time series (as a financial instrument).

        Warnings
        --------
        The result may be wrong. Use this method only for emergency/temporary cases. It is safer to instantiate a
        'real' financial instrument class and call its ``value`` method instead of using this.

        Parameters
        ----------
        last: bool, optional
            Whether to use last available date and quote.
        date: date-like, optional
            The date.
        quote: scalar, optional
            The quote.
        last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.

        Returns
        -------
        scalar, None
            The unit dirty value of the instrument.
        """
        quotes = self.quotes
        if last_available:
            return quotes.get_value(date=date, last_available=last_available)
        if date > quotes.ts_values.last_valid_index():
            return np.nan
        if date < quotes.ts_values.first_valid_index():
            return np.nan
        return quote

    @default_arguments
    @conditional_vectorize('quote', 'date')
    def performance(self, start_date=None, start_quote=None, date=None, quote=None, *args, **kwargs):
        """Try to deduce performance for a unit of the TimeSeries (as a financial instrument).

        Warnings
        --------
        The result may be wrong. Use this method only for emergency/temporary cases. It is safer to instantiate a
        'real' financial instrument class and call its ``performance`` method instead of using this.

        Parameters
        ----------
        start_date: datetime-like, optional
            The starting date of the period.
        date: datetime-like, optional
            The ending date of the period.
        start_quote: float, optional
            The quote of the instrument in `start_date`. Defaults to the quote in `start_date`.
        quote
            The quote of the instrument in `date`. Defaults to the quote in `date`.

        Returns
        -------
        scalar, None
            Performance of a unit of the instrument.
        """
        quotes = self.quotes
        first_available_date = quotes.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if start_quote is None:
            start_quote = quotes.get_values(index=start_date)
        if date < start_date:
            return np.nan
        start_value = self.value(quote=start_quote, date=start_date)
        value = self.value(quote=quote, date=date)

        return (value / start_value) - 1
