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
A class for modelling Equity Options
"""
import QuantLib as ql
import numpy as np
from functools import wraps
from tsfin.constants import CALENDAR, MATURITY_DATE, \
    DAY_COUNTER, EXERCISE_TYPE, OPTION_TYPE, STRIKE_PRICE, UNDERLYING_INSTRUMENT, OPTION_CONTRACT_SIZE
from tsfin.base import Instrument, to_ql_date, conditional_vectorize, to_ql_calendar, to_ql_day_counter, to_datetime, \
    to_list


def option_default_arguments(f):
    """ Decorator to set default arguments for :py:class:`EquityOption` methods.
        QuantLib option is a fairly complex instrument to assemble, you need a series of default values
        that may be specific to the option timeseries or are specific to the method used for calculation.
        All values may be overridden through keyword arguments.

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

    +----------------------------+------------------------------------------------+
    | Missing Attributes or Overrides   | Default Value(s)                        |
    +==================================+==========================================+
    | last_available                   | True, retrieve last available data       |
    +----------------------------------+------------------------------------------+
    | dividend_tax                     | 0 (No adjust)                            |
    +----------------------------------+------------------------------------------+
    | spot_price                       | self.underlying_instrument.spot_price    |
    +----------------------------------+------------------------------------------+
    | option_price                     | self.ts_mid_price                        |
    +----------------------------------+------------------------------------------+
    | dividend_yield                   | self.underlying_instrument.dividend_yield|
    +----------------------------------+------------------------------------------+
    | volatility                       | None                                     |
    |                                  |(It will self calculate the implied vol)  |
    +----------------------------------+------------------------------------------+
    | process                          | self.ql_process                          |
    +----------------------------------+------------------------------------------+
    | calendar                         | self.calendar                            |
    +----------------------------------+------------------------------------------+
    | day_counter                      | self.day_counter                         |
    +----------------------------------+------------------------------------------+
    | maturity                         | self.option_maturity                     |
    +----------------------------------+------------------------------------------+
    | engine_name                      | 'BINOMIAL_VANILLA', see ql_option_engine |
    |                                  | for other possible values                |
    +----------------------------------+------------------------------------------+
    | model_name                       | 'LR', see ql_option_engine for other     |
    |                                  | possible values                          |
    +----------------------------------+------------------------------------------+
    | time_steps                       | '801', see ql_option_engine for other    |
    |                                  | possible values                          |
    +----------------------------------+------------------------------------------+
    | exercise_type                    | self.exercise_type                       |
    +----------------------------------+------------------------------------------+
    | option_type                      | self.option_type                         |
    +----------------------------------+------------------------------------------+
    | strike                           | self.strike                              |
    +----------------------------------+------------------------------------------+
    """

    @wraps(f)
    def new_f(self, **kwargs):

        # Setup
        kwargs['date'] = to_ql_date(kwargs['date'])
        kwargs['base_date'] = to_ql_date(kwargs['base_date'])
        search_date = kwargs['base_date'] if kwargs['date'] > kwargs['base_date'] else kwargs['date']
        ql.Settings.instance().evaluationDate = kwargs['date']
        kwargs['last_available'] = True if kwargs.get('last_available', None) is None else kwargs['last_available']
        last = kwargs['last_available']

        # Underlying Timeseries Values
        kwargs['spot_price'] = float(self.underlying_instrument.spot_price(date=search_date, last_available=last)) \
            if kwargs.get('spot_price', None) is None else kwargs['spot_price']
        kwargs['dividend_yield'] = float(self.underlying_instrument.dividend_yield(date=search_date,
                                                                                   last_available=last)) \
            if kwargs.get('dividend_yield', None) is None else kwargs['dividend_yield']
        kwargs['dividend_tax'] = 0 if kwargs.get('dividend_tax', None) is None else kwargs['dividend_tax']

        # Option Timeseries Values
        kwargs['option_price'] = float(self.ts_mid_price(date=search_date, last_available=last)) \
            if kwargs.get('option_price', None) is None else kwargs['option_price']
        kwargs['quote'] = kwargs['option_price'] if kwargs.get('quote', None) is None else kwargs['quote']
        kwargs['volatility'] = kwargs.get('volatility', None)

        # Timeseries Attributes
        kwargs['calendar'] = getattr(self, 'calendar') if kwargs.get('calendar', None) is None else kwargs['calendar']
        kwargs['day_counter'] = getattr(self, 'day_counter') if kwargs.get('day_counter', None) is None else \
            kwargs['day_counter']
        kwargs['maturity'] = getattr(self, 'option_maturity') if kwargs.get('maturity', None) is None else \
            to_ql_date(kwargs['maturity'])
        kwargs['exercise_type'] = getattr(self, 'exercise_type') if kwargs.get('exercise_type', None) is None else \
            kwargs['exercise_type']
        kwargs['option_type'] = getattr(self, 'option_type') if kwargs.get('option_type', None) is None else \
            kwargs['option_type']
        kwargs['strike'] = getattr(self, 'strike') if kwargs.get('strike', None) is None else kwargs['strike']

        # QuantLib Process
        kwargs['process'] = getattr(self, 'ql_process') if kwargs.get('process', None) is None else kwargs['process']

        # QuantLib Option Engine Arguments
        kwargs['engine_name'] = 'BINOMIAL_VANILLA' if kwargs.get('engine_name', None) is None else kwargs['engine_name']
        kwargs['model_name'] = 'LR' if kwargs.get('model_name', None) is None else kwargs['model_name']
        kwargs['time_steps'] = 801 if kwargs.get('time_steps', None) is None else kwargs['time_steps']

        # Option build-up
        ql_exercise = option_exercise_type(exercise_type=kwargs['exercise_type'], date=kwargs['date'],
                                           maturity=kwargs['maturity'])
        ql_option_type = to_ql_option_type(kwargs['option_type'])
        payoff = ql_option_payoff(ql_option_type, kwargs['strike'])
        self.option = ql_option_instrument(payoff, ql_exercise)
        return f(self, **kwargs)
    return new_f


def to_ql_option_type(arg):
    """Converts a string with the option type to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Option

    """
    if arg.upper() == 'CALL':
        return ql.Option.Call
    elif arg.upper() == 'PUT':
        return ql.Option.Put


def option_exercise_type(exercise_type, date, maturity):
    if exercise_type.upper() == 'AMERICAN':
        if date > maturity:
            date = maturity
        return ql.AmericanExercise(date, maturity)
    elif exercise_type.upper() == 'EUROPEAN':
        return ql.EuropeanExercise(maturity)
    else:
        raise ValueError('Exercise type not supported')


def ql_option_engine(engine_name=None, process=None, model_name=None, time_steps=None, **kwargs):

    if engine_name.upper() == 'BINOMIAL_VANILLA':
        return ql.BinomialVanillaEngine(process, model_name, time_steps)
    elif engine_name.upper() == 'ANALYTIC_HESTON':
        return ql.AnalyticHestonEngine(ql.HestonModel(process))
    elif engine_name.upper() == 'ANALYTIC_EUROPEAN':
        return ql.AnalyticEuropeanEngine(process)
    elif engine_name.upper() == 'ANALYTIC_EUROPEAN_DIVIDEND':
        return ql.AnalyticDividendEuropeanEngine(process)
    else:
        return None


def ql_option_instrument(*args):

    return ql.VanillaOption(*args)


def ql_option_payoff(*args):

    return ql.PlainVanillaPayoff(*args)


class EquityOption(Instrument):
    """ Model for Equity Options using the Black Scholes Merton model.

    :param timeseries: :py:class:`TimeSeries`
        The TimeSeries representing the option.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.option_type = self.ts_attributes[OPTION_TYPE]
        self.strike = float(self.ts_attributes[STRIKE_PRICE])
        self.contract_size = float(self.ts_attributes[OPTION_CONTRACT_SIZE])
        self.option_maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.exercise_type = self.ts_attributes[EXERCISE_TYPE]
        self.underlying_name = self.ts_attributes[UNDERLYING_INSTRUMENT]
        self.underlying_instrument = None
        self.ql_process = None
        self.option = None
        self.implied_volatility = dict()

    def set_underlying_instrument(self, underlying_instrument):
        """
        :param underlying_instrument: :py:class:'Equity'
            A class for modeling Equity, ETF.
        :return:
        """
        self.underlying_instrument = underlying_instrument

    def set_ql_process(self, ql_process):
        """
        :param ql_process: :py:class:'BlackScholesMerton'
            A class used to handle the Black Scholes Merton model from QuantLib.
        :return:
        """
        self.ql_process = ql_process

    def set_pricing_engine(self, ql_engine):
        """

        :param ql_engine: QuantLib.PricingEngine
            The engine used to calculate the option.
        :return:
        """

        self.option.setPricingEngine(ql_engine)

    def is_expired(self, date, *args, **kwargs):
        """
        :param date: date-like
            The date.
        :return bool
            True if the instrument is expired or matured, False otherwise.
        """
        date = to_ql_date(date)
        if date >= self.option_maturity:
            return True
        return False

    def maturity(self, date, *args, **kwargs):
        """
        :param date: date-like
            The date.
        :return QuantLib.Date, None
            Date representing the maturity or expiry of the instrument. Returns None if there is no maturity.
        """
        return self.option_maturity

    @conditional_vectorize('date', 'quote', 'volatility')
    @option_default_arguments
    def value(self, date, base_date, quote, volatility, dividend_tax, last_available, exercise_type, *args, **kwargs):
        """Try to deduce dirty value for a unit of the time series (as a financial instrument).

        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param quote: scalar, optional
            The quote.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param dividend_tax: float, default=1
            The dividend % tax applied.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_type: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :return scalar, None
            The unit dirty value of the instrument.
        """
        size = self.contract_size
        if quote is not None:
            return float(quote)*size
        else:
            price = self.price(date=date, base_date=base_date, dividend_tax=dividend_tax, last_available=last_available,
                               exercise_type=exercise_type, volatility=volatility, **kwargs)
            return price * size

    @conditional_vectorize('date', 'quote')
    def performance(self, date=None, quote=None, start_date=None, start_quote=None, *args, **kwargs):
        """
        Parameters
        ----------
        :param start_date: datetime-like, optional
            The starting date of the period.
            Default: The first date in ``self.quotes``.
        :param date: datetime-like, optional, (c-vectorized)
            The ending date of the period.
            Default: see :py:func:`default_arguments`.
        :param start_quote: scalar, optional, (c-vectorized)
            The quote of the instrument in `start_date`.
            Default: the quote in `start_date`.
        :param quote: scalar, optional, (c-vectorized)
            The quote of the instrument in `date`.
            Default: see :py:func:`default_arguments`.
        :return scalar, None
            Performance of a unit of the option.
        """
        first_available_date = self.px_mid.ts_values.first_valid_index()
        if start_date is None:
            start_date = first_available_date
        if start_date < first_available_date:
            start_date = first_available_date
        if date < start_date:
            return np.nan
        start_value = self.value(date=start_date, quote=quote, *args, **kwargs)
        value = self.value(date=date, quote=quote, *args, **kwargs)

        return (value / start_value) - 1

    def notional(self):
        """
        :return: float
            The notional of the contract based on the option contract size and strike.
        """
        return self.contract_size * self.strike

    def intrinsic(self, date, spot_price, strike):
        """
        :param date: date-like
            The date
        :param spot_price: float
            The underlying spot price.
        :param strike: float
            The options strike price
        :return: float
            The intrinsic value o the option at date.
        """
        if date > self.option_maturity:
            return 0
        else:
            intrinsic = 0
            if self.option_type == 'CALL':
                intrinsic = spot_price - strike
            if self.option_type == 'PUT':
                intrinsic = strike - spot_price
            if intrinsic < 0:
                return 0
            else:
                return intrinsic
    
    def ts_mid_price(self, date, last_available=True, fill_value=np.nan):

        date = to_datetime(to_list(date))
        return self.px_mid.get_values(index=date, last_available=last_available, fill_value=fill_value)

    def ts_implied_volatility(self, date, last_available=False, fill_value=np.nan):

        date = to_datetime(to_list(date))
        return self.ivol_mid.get_values(index=date, last_available=last_available, fill_value=fill_value)

    def option_engine(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, volatility,
                      process, calendar, day_counter, maturity, engine_name, **kwargs):

        """
        :param date: QuantLib.Date
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            The underlying spot price.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            An override of the dividend yield in case you don't wan't to use the timeseries one.
        :param dividend_tax: float
            The dividend % tax applied.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: QuantLib.VanillaOption
            This method returns the VanillaOption with a QuantLib engine. Used for calculating the option values
            and greeks.
        """
        process.update_process(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               calendar=calendar, day_counter=day_counter, volatility=0.2, maturity=maturity,
                               dividend_tax=dividend_tax)

        self.set_pricing_engine(ql_option_engine(engine_name=engine_name, process=process.process, **kwargs))

        if date > base_date:
            date = base_date
        if volatility is not None:
            self.implied_volatility[date] = ql.SimpleQuote(volatility)
        try:
            process.volatility_update(volatility=self.implied_volatility[date], calendar=calendar,
                                      day_counter=day_counter)
        except KeyError:
            try:
                implied_vol = self.option.impliedVolatility(targetValue=option_price, process=process.process,
                                                            accuracy=1.0e-4, maxEvaluations=100)

            except RuntimeError:
                # need a better fix for this, some days it's impossible to find the implied vol so we use the prior
                # day volatility as a proxy.
                prior_date = calendar.advance(date, -1, ql.Days)
                mid_price = float(self.ts_mid_price(date=prior_date, last_available=True))
                implied_vol = self.option.impliedVolatility(targetValue=mid_price, process=process.process,
                                                            accuracy=1.0e-4, maxEvaluations=100)
            self.implied_volatility[date] = ql.SimpleQuote(implied_vol)
            process.volatility_update(volatility=self.implied_volatility[date], calendar=calendar,
                                      day_counter=day_counter)

    @conditional_vectorize('date', 'volatility', 'spot_price')
    @option_default_arguments
    def price(self, date, base_date, spot_price, volatility, dividend_yield, dividend_tax, last_available, process,
              calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option price at date.
        """
        if self.is_expired(date=date):
            return self.intrinsic(date=maturity, spot_price=spot_price, **kwargs)
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dividend_tax=dividend_tax, last_available=last_available, volatility=volatility,
                               process=process, calendar=calendar, day_counter=day_counter, maturity=maturity,
                               engine_name=engine_name, **kwargs)
        return self.option.NPV()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def price_underlying(self, date, base_date, spot_price, dividend_yield, dividend_tax, last_available, volatility,
                         process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option price based on the date and underlying spot price.
        """
        price = self.price(date=date, base_date=base_date, spot_price=None, dividend_yield=dividend_yield,
                           dividend_tax=dividend_tax, last_available=last_available, volatility=volatility,
                           process=process, calendar=calendar, day_counter=day_counter, maturity=maturity,
                           engine_name=engine_name, **kwargs)
        self.ql_process.spot_price_update(spot_price=spot_price)
        return self.option.NPV()

    @conditional_vectorize('date', 'spot_price', 'volatility', 'option_price')
    @option_default_arguments
    def delta(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available, volatility,
              process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option delta at date.
        """
        if self.is_expired(date=date):
            if self.intrinsic(date=self.option_maturity, spot_price=spot_price, **kwargs) > 0:
                return 1
            else:
                return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            return self.option.delta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def delta_underlying(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available,
                         volatility, process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option delta based on the date and underlying spot price.
        """
        delta = self.delta(date=date, base_date=base_date, spot_price=None, option_price=option_price,
                           dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                           volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                           maturity=maturity, engine_name=engine_name, **kwargs)
        self.ql_process.spot_price_update(spot_price=spot_price)
        return self.option.delta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def gamma(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available, volatility,
              process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option gamma at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            return self.option.gamma()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def theta(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available, volatility,
              process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option theta at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            return self.option.theta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def vega(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available, volatility,
             process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option vega at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.vega()
                except:
                    return 0

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def rho(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available, volatility,
            process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option rho at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.rho()
                except:
                    return 0

    @conditional_vectorize('date', 'spot_price', 'option_price')
    @option_default_arguments
    def implied_vol(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available,
                    process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option volatility based on the option price and date.
        """
        self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                           dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                           process=process, calendar=calendar, day_counter=day_counter, maturity=maturity,
                           engine_name=engine_name, **kwargs)

        return self.implied_volatility[date].value()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def optionality(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available,
                    volatility, process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option optionality at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                               dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                               volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                               maturity=maturity, engine_name=engine_name, **kwargs)
            price = self.option.NPV()
            if date > base_date:
                intrinsic = self.intrinsic(date=base_date, spot_price=spot_price, **kwargs)
            else:
                intrinsic = self.intrinsic(date=date, spot_price=spot_price, **kwargs)
            return price - intrinsic

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def underlying_price(self, date, **kwargs):
        """
        :param date: date-like
            The date.
        :return: float
            The option underlying spot price.
        """
        date = to_ql_date(date)
        spot_price = self.underlying_instrument.spot_price(date=date)
        return spot_price

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def delta_value(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, last_available,
                    volatility, process, calendar, day_counter, maturity, engine_name, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param option_price: float
            The option price used to calculate the implied volatility.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dividend_tax: float
            The dividend % tax applied.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float
            Volatility override value to calculate the option.
        :param process: QuantLib.StochasticProcess1D
            The process used for option evaluation.
        :param calendar: QuantLib.Calendar
            The option calendar used to evaluate the model
        :param day_counter: QuantLib.DayCounter
            The option day count used to evaluate the model
        :param maturity: ql.Date
            The option maturity date.
        :param engine_name: str
            The engine name representing a QuantLib Option Pricing Engine. Other necessary arguments are passed through
            kwargs. (see ql_option_engine for details).
        :return: float
            The option delta notional value.
        """
        delta = self.delta(date=date, base_date=base_date, spot_price=spot_price, option_price=option_price,
                           dividend_yield=dividend_yield, dividend_tax=dividend_tax, last_available=last_available,
                           volatility=volatility, process=process, calendar=calendar, day_counter=day_counter,
                           maturity=maturity, engine_name=engine_name, **kwargs)
        spot = self.underlying_instrument.spot_price(date=date)
        size = float(self.contract_size)
        return delta*spot*size
