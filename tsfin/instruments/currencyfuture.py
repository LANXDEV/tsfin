"""
Currency Future class to model currency futures. Also contains the BaseCurrencyFuture and CurrencyFutureHelper classes,
which work as analogies to QuantLib's instrument and helper classes, respectively.
TODO: Propose implementation of these classes in QuantLib.
"""
from functools import wraps
import numpy as np
import QuantLib as ql
from tsfin.base import Instrument, conditional_vectorize, to_ql_date, to_ql_frequency, to_ql_calendar, \
    to_ql_compounding, to_ql_day_counter, to_ql_currency
from tsfin.constants import BASE_CURRENCY, COUNTER_CURRENCY, BASE_RATE_COMPOUNDING, BASE_RATE_DAY_COUNTER, \
    BASE_RATE_FREQUENCY, COUNTER_RATE_COMPOUNDING, COUNTER_RATE_DAY_COUNTER, COUNTER_RATE_FREQUENCY, CALENDAR, \
    MATURITY_DATE, MULTIPLY_QUOTES_BY, SETTLEMENT_DAYS


def usdbrl_next_maturity(date):
    """ Next maturity of BM&F (Brazil) USDBRL future contracts.

    Parameters
    ----------
    date: QuantLib.Date
        Reference date.

    Returns
    -------
    QuantLib.Date
        Maturity date of closest-to-maturity contract.
    """
    date = to_ql_date(date)
    calendar = ql.Brazil()
    return calendar.advance(calendar.endOfMonth(calendar.advance(date, 1, ql.Days)), 1, ql.Days)


def default_arguments_currency_future(f):
    """ Decorator to set default arguments for :py:class:`CurrencyFuture`.

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
    | base_rate_day_counter      | self.base_rate_day_counter               |
    +----------------------------+------------------------------------------+
    | base_rate_compounding      | self.base_rate_compounding               |
    +----------------------------+------------------------------------------+
    | base_rate_frequency        | self.base_rate_frequency                 |
    +----------------------------+------------------------------------------+
    | counter_rate_day_counter   | self.counter_rate_day_counter            |
    +----------------------------+------------------------------------------+
    | counter_rate_compounding   | self.counter_rate_compounding            |
    +----------------------------+------------------------------------------+
    | counter_rate_frequency     | self.counter_rate_frequency              |
    | maturity                   | self.maturity_date                       |
    +----------------------------+------------------------------------------+

    """
    @wraps(f)
    def new_f(self, *args, **kwargs):
        if 'base_rate_day_counter' not in kwargs.keys():
            kwargs['base_rate_day_counter'] = getattr(self, 'base_rate_day_counter')
        if 'counter_rate_day_counter' not in kwargs.keys():
            kwargs['counter_rate_day_counter'] = getattr(self, 'counter_rate_day_counter')
        if 'base_rate_compounding' not in kwargs.keys():
            kwargs['base_rate_compounding'] = getattr(self, 'base_rate_compounding')
        if 'counter_rate_compounding' not in kwargs.keys():
            kwargs['counter_rate_compounding'] = getattr(self, 'counter_rate_compounding')
        if 'base_rate_frequency' not in kwargs.keys():
            kwargs['base_rate_frequency'] = getattr(self, 'base_rate_frequency')
        if 'counter_rate_frequency' not in kwargs.keys():
            kwargs['counter_rate_frequency'] = getattr(self, 'counter_rate_frequency')
        if 'maturity' not in kwargs.keys():
            kwargs['maturity'] = getattr(self, 'maturity_date')
        if 'settlement_days' not in kwargs.keys():
            kwargs['settlement_days'] = getattr(self, 'settlement_days')
        if 'last' not in kwargs.keys():
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
            kwargs['quote'] = getattr(self, 'quotes').get_value(date=kwargs['date'])
        return f(self, *args, **kwargs)
    new_f._decorated_by_default_arguments_ = True
    return new_f


def default_arguments_base_currency_future(f):
    """ Decorator to set default arguments for :py:class:`BaseCurrencyFuture`.

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
    | base_rate_day_counter      | self.base_rate_day_counter               |
    +----------------------------+------------------------------------------+
    | base_rate_compounding      | self.base_rate_compounding               |
    +----------------------------+------------------------------------------+
    | base_rate_frequency        | self.base_rate_frequency                 |
    +----------------------------+------------------------------------------+
    | counter_rate_day_counter   | self.counter_rate_day_counter            |
    +----------------------------+------------------------------------------+
    | counter_rate_compounding   | self.counter_rate_compounding            |
    +----------------------------+------------------------------------------+
    | counter_rate_frequency     | self.counter_rate_frequency              |
    +----------------------------+------------------------------------------+
    | maturity                   | self.maturity_date                       |
    +----------------------------+------------------------------------------+

    """
    @wraps(f)
    def new_f(self, *args, **kwargs):
        if 'base_rate_day_counter' not in kwargs.keys():
            kwargs['base_rate_day_counter'] = getattr(self, 'base_rate_day_counter')
        if 'counter_rate_day_counter' not in kwargs.keys():
            kwargs['counter_rate_day_counter'] = getattr(self, 'counter_rate_day_counter')
        if 'base_rate_compounding' not in kwargs.keys():
            kwargs['base_rate_compounding'] = getattr(self, 'base_rate_compounding')
        if 'counter_rate_compounding' not in kwargs.keys():
            kwargs['counter_rate_compounding'] = getattr(self, 'counter_rate_compounding')
        if 'base_rate_frequency' not in kwargs.keys():
            kwargs['base_rate_frequency'] = getattr(self, 'base_rate_frequency')
        if 'counter_rate_frequency' not in kwargs.keys():
            kwargs['counter_rate_frequency'] = getattr(self, 'counter_rate_frequency')
        if 'maturity' not in kwargs.keys():
            kwargs['maturity'] = getattr(self, 'maturity_date')
        if 'settlement_days' not in kwargs.keys():
            kwargs['settlement_days'] = getattr(self, 'settlement_days')
        return f(self, **kwargs)
    new_f._decorated_by_default_arguments_ = True
    return new_f


class CurrencyFuture(Instrument):
    """ Currency future time series object.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
    """
    def __init__(self, timeseries, *args, **kwargs):
        super().__init__(timeseries)
        self.base_currency = to_ql_currency(self.ts_attributes[BASE_CURRENCY])
        self.counter_currency = to_ql_currency(self.ts_attributes[COUNTER_CURRENCY])
        self.maturity_date = to_ql_date(self.ts_attributes[MATURITY_DATE])
        self.timeseries.quotes.ts_values *= self.ts_attributes.get(MULTIPLY_QUOTES_BY, 1)
        self.quotes = self.timeseries.price.ts_values
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.base_rate_day_counter = to_ql_day_counter(self.ts_attributes[BASE_RATE_DAY_COUNTER])
        self.counter_rate_day_counter = to_ql_day_counter(self.ts_attributes[COUNTER_RATE_DAY_COUNTER])
        self.base_rate_compounding = to_ql_compounding(self.ts_attributes[BASE_RATE_COMPOUNDING])
        self.counter_rate_compounding = to_ql_compounding(self.ts_attributes[COUNTER_RATE_COMPOUNDING])
        self.base_rate_frequency = to_ql_frequency(self.ts_attributes[BASE_RATE_FREQUENCY])
        self.counter_rate_frequency = to_ql_frequency(self.ts_attributes[COUNTER_RATE_FREQUENCY])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])

        self.base_currency_future = BaseCurrencyFuture(self.base_currency, self.counter_currency, self.maturity_date,
                                                       self.calendar, self.base_rate_day_counter,
                                                       self.base_rate_compounding, self.base_rate_frequency,
                                                       self.counter_rate_day_counter,
                                                       self.counter_rate_compounding,
                                                       self.counter_rate_frequency, self.settlement_days)

    def is_expired(self, date, *args, **kwargs):
        """Check if the contract is expired.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        bool
            Whether the contract is expired.
        """
        date = to_ql_date(date)
        if date >= self.calendar.advance(self.maturity_date, -1, ql.Days):
            return True
        return False

    def maturity(self, date, *args, **kwargs):
        """Return maturity date of the contract.

        Parameters
        ----------
        date: QuantLib.Date
            The reference date.

        Returns
        -------
        QuantLib.Date
            The maturity date of the contract.
        """
        return self.maturity_date

    def tenor(self, date, *args, **kwargs):
        """Tenor to maturity of the contract.

        Parameters
        ----------
        date: QuantLib.Date

        Returns
        -------
        QuantLib.Period
            Period representing the time to maturity of the contract.
        """
        return ql.Period(self.calendar.businessDaysBetween(date, self.maturity_date), ql.Days)

    def cash_to_date(self, start_date, date, *args, **kwargs):
        """Cashflow received/paid by the holder of a contract.

        Parameters
        ----------
        start_date: QuantLib.Date
        date: QuantLib.Date

        Returns
        -------
        scalar
            Cash amount received/paid by the holder of a contract between `start_date` and `date`.
        """
        # TODO: Implement this.
        raise NotImplementedError("This method is not yet implemented.")

    def value(self, last, date, quote, last_available_data=False, *args, **kwargs):
        """Value of a contract.

        Parameters
        ----------
        last: bool
            Use last data in ``self.price.ts_values``.
        date: QuantLib.Date
            Valuation date.
        quote: scalar
            Quote of the contract at `date`.
        last_available_data: bool
            Whether to use last available data, if unavailable.

        Returns
        -------
        scalar
            Valuation of the contract at `date`.
        """
        raise NotImplementedError("This method is not yet implemented.")

    @default_arguments_currency_future
    @conditional_vectorize('quote', 'spot', 'base_rate', 'date')
    def counter_rate(self, spot, base_rate, last, quote, date, counter_rate_day_counter, counter_rate_compounding,
                     counter_rate_frequency, base_rate_day_counter, base_rate_compounding, base_rate_frequency,
                     maturity, *args, **kwargs):
        """Obtain the interest rate of the counter currency.

        Parameters
        ----------
        spot: scalar
            Spot exchange rate.
        base_rate: scalar
            Interest rate of the base currency.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments_currency_future`.
        quote: scalar, optional
            Quote of the contract.
            Default: see :py:func:`default_arguments_currency_future`.
        date: QuantLib.Date, optional
            Reference date.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_day_counter: QuantLib.DayCounter, optional
            Day counter of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_day_counter: QuantLib.DayCounter, optional
            Day base of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        maturity: QuantLib.Date, optional
            Maturity date of the contract.
            Default: see :py:func:`default_arguments_currency_future`.

        Returns
        -------
        scalar
            Implied interest rate to `maturity` in the counter currency.
        """
        return self.base_currency_future.counter_rate(quote=quote, spot=spot, base_rate=base_rate, date=date,
                                                      counter_rate_day_counter=counter_rate_day_counter,
                                                      counter_rate_compounding=counter_rate_compounding,
                                                      counter_rate_frequency=counter_rate_frequency,
                                                      base_rate_day_counter=base_rate_day_counter,
                                                      base_rate_compounding=base_rate_compounding,
                                                      base_rate_frequency=base_rate_frequency, maturity=maturity).rate()

    @default_arguments_currency_future
    @conditional_vectorize('quote', 'spot', 'counter_rate', 'date')
    def base_rate(self, spot, counter_rate, last, quote, date, counter_rate_day_counter, counter_rate_compounding,
                  counter_rate_frequency, base_rate_day_counter, base_rate_compounding, base_rate_frequency,
                  maturity, *args, **kwargs):
        """Obtain the interest rate of the base currency.

        Parameters
        ----------
        spot: scalar
            Spot exchange rate.
        counter_rate: scalar
            Interest rate of the base currency.
        last: bool, optional
            Whether to use last data.
            Default: see :py:func:`default_arguments_currency_future`.
        quote: scalar, optional
            Quote of the contract.
            Default: see :py:func:`default_arguments_currency_future`.
        date: QuantLib.Date, optional
            Reference date.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_day_counter: QuantLib.DayCounter, optional
            Day counter of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_day_counter: QuantLib.DayCounter, optional
            Day base of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        maturity: QuantLib.Date, optional
            Maturity date of the contract.
            Default: see :py:func:`default_arguments_currency_future`.

        Returns
        -------
        scalar
            Implied interest rate to `maturity` in the base currency.
        """
        return self.base_currency_future.base_rate(quote=quote, spot=spot, counter_rate=counter_rate, date=date,
                                                   counter_rate_day_counter=counter_rate_day_counter,
                                                   counter_rate_compounding=counter_rate_compounding,
                                                   counter_rate_frequency=counter_rate_frequency,
                                                   base_rate_day_counter=base_rate_day_counter,
                                                   base_rate_compounding=base_rate_compounding,
                                                   base_rate_frequency=base_rate_frequency, maturity=maturity).rate()

    def helper(self, date, last_available=True, *args, **kwargs):
        """Obtain a currency helper for time series of currency curves building.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool, optional
            Whether to use last available data.
            Default: True.

        Returns
        -------
        :py:obj:`CurrencyFutureHelper`
            Helper for use of currency curve time series objects.
        """
        if self.is_expired(date):
            return None
        quote = self.price.get_value(date=date, last_available=last_available, default=np.nan)
        if np.isnan(quote):
            return None
        date = to_ql_date(date)
        return CurrencyFutureHelper(base_currency=self.base_currency, counter_currency=self.counter_currency,
                                    maturity=self.maturity_date, calendar=self.calendar,
                                    base_rate_day_counter=self.base_rate_day_counter,
                                    base_rate_compounding=self.base_rate_compounding,
                                    base_rate_frequency=self.base_rate_frequency,
                                    counter_rate_day_counter=self.counter_rate_day_counter,
                                    counter_rate_compounding=self.counter_rate_compounding,
                                    counter_rate_frequency=self.counter_rate_frequency, date=date,
                                    quote=quote)


class BaseCurrencyFuture:
    """ A currency future object.

    Analogous to QuantLib instrument class (but simplified).

    Parameters
    ----------
    base_currency: QuantLib.Currency
        Representation of the base currency.
    counter_currency: QuantLib.Currency
        Representation of the counter currency.
    maturity: QuantLib.Date
        Maturity of the contract.
    calendar: QuantLib.Calendar
        Calendar followed by the contract.
    base_rate_day_counter: QuatLib.DayCounter
        Day count convention of the base currency interest rate.
    base_rate_compounding: QuantLib.Compounding
        Compounding convention of the base currency interest rate.
    base_rate_frequency: QuantLib.Frequency
        Compounding frequency of the base currency interest rate.
    counter_rate_day_counter: QuatLib.DayCounter
        Day count convention of the counter currency interest rate.
    counter_rate_compounding: QuantLib.Compounding
        Compounding convention of the counter currency interest rate.
    counter_rate_frequency: QuantLib.Frequency
        Compounding frequency of the counter currency interest rate.
    settlement_days: int
        Settlement days for the contract.
    """
    def __init__(self, base_currency, counter_currency, maturity, calendar, base_rate_day_counter,
                 base_rate_compounding, base_rate_frequency, counter_rate_day_counter, counter_rate_compounding,
                 counter_rate_frequency, settlement_days):
        self.base_currency = base_currency
        self.counter_currency = counter_currency
        self.maturity = maturity
        self.calendar = calendar
        self.base_rate_day_counter = base_rate_day_counter
        self.base_rate_compounding = base_rate_compounding
        self.base_rate_frequency = base_rate_frequency
        self.counter_rate_day_counter = counter_rate_day_counter
        self.counter_rate_compounding = counter_rate_compounding
        self.counter_rate_frequency = counter_rate_frequency
        self.settlement_days = settlement_days

    @default_arguments_base_currency_future
    def counter_rate(self, spot, base_rate, quote, date, counter_rate_day_counter, counter_rate_compounding,
                     counter_rate_frequency, base_rate_day_counter, base_rate_compounding, base_rate_frequency,
                     maturity, *args, **kwargs):
        """Obtain the interest rate of the counter currency.

        Parameters
        ----------
        spot: scalar
            Spot exchange rate.
        base_rate: scalar
            Interest rate of the base currency.
        quote: scalar, optional
            Quote of the contract.
            Default: see :py:func:`default_arguments_base_currency_future`.
        date: QuantLib.Date, optional
            Reference date.
            Default: see :py:func:`default_arguments_base_currency_future`.
        counter_rate_day_counter: QuantLib.DayCounter, optional
            Day counter of the counter currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        counter_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the counter currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        counter_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the counter currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        base_rate_day_counter: QuantLib.DayCounter, optional
            Day base of the base currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        base_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the base currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        base_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the base currency interest rate.
            Default: see :py:func:`default_arguments_base_currency_future`.
        maturity: QuantLib.Date, optional
            Maturity date of the contract.
            Default: see :py:func:`default_arguments_base_currency_future`.

        Returns
        -------
        scalar
            Implied interest rate to `maturity` in the counter currency.
        """
        date = to_ql_date(date)
        maturity = to_ql_date(maturity)
        base_rate = ql.InterestRate(base_rate, base_rate_day_counter, base_rate_compounding, base_rate_frequency)
        compound = quote / spot * base_rate.discountFactor(date, maturity)
        return ql.InterestRate_impliedRate(compound, counter_rate_day_counter, counter_rate_compounding,
                                           counter_rate_frequency, date, maturity)

    @default_arguments_base_currency_future
    def base_rate(self, spot, counter_rate, quote, date, counter_rate_day_counter, counter_rate_compounding,
                  counter_rate_frequency, base_rate_day_counter, base_rate_compounding, base_rate_frequency,
                  maturity, *args, **kwargs):
        """Obtain the interest rate of the base currency.

        Parameters
        ----------
        spot: scalar
            Spot exchange rate.
        counter_rate: scalar
            Interest rate of the base currency.
        quote: scalar, optional
            Quote of the contract.
            Default: see :py:func:`default_arguments_currency_future`.
        date: QuantLib.Date, optional
            Reference date.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_day_counter: QuantLib.DayCounter, optional
            Day counter of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        counter_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the counter currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_day_counter: QuantLib.DayCounter, optional
            Day base of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_compounding: QuantLib.Compounding, optional
            Compounding convention of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        base_rate_frequency: QuantLib.Frequency, optional
            Compounding frequency of the base currency interest rate.
            Default: see :py:func:`default_arguments_currency_future`.
        maturity: QuantLib.Date, optional
            Maturity date of the contract.
            Default: see :py:func:`default_arguments_currency_future`.

        Returns
        -------
        scalar
            Implied interest rate to `maturity` in the base currency.
        """
        date = to_ql_date(date)
        maturity = to_ql_date(maturity)
        counter_rate = ql.InterestRate(counter_rate, counter_rate_day_counter, counter_rate_compounding,
                                       counter_rate_frequency)
        compound = spot * counter_rate.compoundFactor(date, maturity) / quote
        return ql.InterestRate_impliedRate(compound, base_rate_day_counter, base_rate_compounding, base_rate_frequency,
                                           date, maturity)


class CurrencyFutureHelper(BaseCurrencyFuture):
    """Helper for currency future curve construction.

    Parameters
    ----------
    base_currency: QuantLib.Currency
        Representation of the base currency.
    counter_currency: QuantLib.Currency
        Representation of the counter currency.
    maturity: QuantLib.Date
        Maturity of the contract.
    calendar: QuantLib.Calendar
        Calendar of the contract.
    base_rate_day_counter: QuantLib.DayCounter
        Day counter of base currency interest rate.
    base_rate_compounding: QuantLib.Compounding
        Compounding convention of base currency interest rate.
    base_rate_frequency: QuantLib.Frequency
        Compounding frequency of base currency interest rate.
    counter_rate_day_counter: QuantLib.DayCounter
        Day counter of counter currency interest rate.
    counter_rate_compounding: QuantLib.Compounding
        Compounding convention of counter currency interest rate.
    counter_rate_frequency: QuantLib.Frequency
        Compounding frequency of counter currency interest rate.
    date: QuantLib.Date
        Reference date.
    quote: scalar
        Quote of the contract.
    """
    def __init__(self, base_currency, counter_currency, maturity, calendar, base_rate_day_counter,
                 base_rate_compounding, base_rate_frequency, counter_rate_day_counter, counter_rate_compounding,
                 counter_rate_frequency, date, quote):
        # Just a BaseCurrencyFuture with a date and a quote.
        super().__init__(base_currency, counter_currency, maturity, calendar, base_rate_day_counter,
                         base_rate_compounding, base_rate_frequency, counter_rate_day_counter,
                         counter_rate_compounding, counter_rate_frequency)
        self.date = date
        self.quote = quote
