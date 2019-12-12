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
from tsfin.constants import CALENDAR, MATURITY_DATE, DAY_COUNTER, EXERCISE_TYPE, OPTION_TYPE, STRIKE_PRICE, \
    UNDERLYING_INSTRUMENT, OPTION_CONTRACT_SIZE, EARLIEST_DATE, PAYOFF_TYPE, BLACK_SCHOLES_MERTON, BLACK_SCHOLES, \
    HESTON, GJR_GARCH
from tsfin.base import Instrument, to_ql_date, conditional_vectorize, to_ql_calendar, to_ql_day_counter, to_datetime, \
    to_list, to_ql_option_type, to_ql_one_asset_option, to_ql_option_payoff, to_ql_option_engine, \
    to_ql_option_exercise_type


MID_PRICE = 'PX_MID'
IMPLIED_VOL = 'IVOL_MID'


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
    | process                          | self.ql_process                          |
    +----------------------------------+------------------------------------------+
    | engine_name                      | 'FINITE_DIFFERENCES_DIVIDEND',           |
    |                                  | see to_ql_option_engine for other        |
    |                                  |  possible values                         |
    +----------------------------------+------------------------------------------+
    | model_name                       | 'LR', see ql_option_engine for other     |
    |                                  | possible values                          |
    +----------------------------------+------------------------------------------+
    | time_steps                       | '801', see ql_option_engine for other    |
    |                                  | possible values                          |
    +----------------------------------+------------------------------------------+
    """

    @wraps(f)
    def new_f(self, **kwargs):

        # QuantLib Process
        if kwargs.get('base_equity_process', None) is None:
            kwargs['base_equity_process'] = self.ql_process

        # Yield Curve
        if kwargs.get('yield_curve', None) is not None:
            self.yield_curve = kwargs['yield_curve']

        # QuantLib Option Engine Arguments
        if kwargs.get('engine_name', None) is not None:
            self.engine_name = kwargs['engine_name']
        # Option build-up
        return f(self, **kwargs)

    return new_f


def option_default_values(f):
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
    """

    @wraps(f)
    def new_f(self, **kwargs):

        # Setup
        try:
            kwargs['date'] = to_ql_date(kwargs['date'])
        except TypeError:
            kwargs['date'] = to_ql_date(kwargs['date'][0])
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
        return f(self, **kwargs)
    return new_f


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
        self._maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.exercise_type = self.ts_attributes[EXERCISE_TYPE]
        self.underlying_name = self.ts_attributes[UNDERLYING_INSTRUMENT]
        try:
            self.earliest_date = to_ql_date(self.ts_attributes[EARLIEST_DATE])
        except KeyError:
            self.earliest_date = ql.Date.minDate()
        self.exercise = to_ql_option_exercise_type(self.exercise_type, self.earliest_date, self._maturity)
        self.payoff = to_ql_option_payoff(self.ts_attributes[PAYOFF_TYPE], to_ql_option_type(self.option_type),
                                          self.strike)
        self.option = to_ql_one_asset_option(self.payoff, self.exercise)
        self.yield_curve = None
        self.underlying_instrument = None
        # engine setup
        self.engine_name = 'FINITE_DIFFERENCES'
        self.ql_process = None
        self._implied_volatility = dict()

    def change_exercise_type(self, exercise_type):

        exercise_type = str(exercise_type).upper()
        self.exercise = to_ql_option_exercise_type(exercise_type, self.earliest_date, self._maturity)
        self.payoff = to_ql_option_payoff(self.ts_attributes[PAYOFF_TYPE], to_ql_option_type(self.option_type),
                                          self.strike)
        self.option = to_ql_one_asset_option(self.payoff, self.exercise)

    def set_yield_curve(self, yield_curve):

        self.yield_curve = yield_curve

    def set_underlying_instrument(self, underlying_instrument):
        """
        :param underlying_instrument: :py:class:'Equity'
            A class for modeling Equity, ETF.
        :return:
        """
        self.underlying_instrument = underlying_instrument

    def set_ql_process(self, ql_process, **kwargs):
        """
        :param ql_process: :py:class:'BaseEquityProcess'
            A class used to handle the Stochastic Models for Equities from QuantLib.
        :return:
        """
        self.ql_process = ql_process(calendar=self.calendar, day_counter=self.day_counter, **kwargs)

    def set_pricing_engine(self, ql_engine=None, engine_name=None, process=None):
        """

        :param ql_engine: QuantLib.PricingEngine
            The engine used to calculate the option.
        :param engine_name: str
            The engine name
        :param process: QuantLib.StochasticProcess
            The QuantLib object with the option Stochastic Process.
        :return:
        """
        if ql_engine is not None:
            self.option.setPricingEngine(ql_engine)
        elif engine_name is not None and process is not None:
            option_engine = to_ql_option_engine(engine_name=engine_name, process=process)
            self.option.setPricingEngine(option_engine)
        else:
            option_engine = to_ql_option_engine(engine_name=self.engine_name, process=process)
            self.option.setPricingEngine(option_engine)

    def is_expired(self, date, *args, **kwargs):
        """
        :param date: date-like
            The date.
        :return bool
            True if the instrument is expired or matured, False otherwise.
        """
        date = to_ql_date(date)
        if date >= self._maturity:
            return True
        return False

    def maturity(self, date, *args, **kwargs):
        """
        :param date: date-like
            The date.
        :return QuantLib.Date, None
            Date representing the _maturity or expiry of the instrument. Returns None if there is no _maturity.
        """
        return self._maturity

    @conditional_vectorize('date')
    @option_default_values
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
        start_date = to_ql_date(start_date)
        date = to_ql_date(date)
        if start_date <= self._maturity <= date:
            spot_price = float(self.underlying_instrument.spot_price(date=self._maturity, last_available=True))
            intrinsic = self.intrinsic(self._maturity, spot_price)
        else:
            intrinsic = 0
        return [(self._maturity, intrinsic*self.contract_size)]

    @option_default_arguments
    @conditional_vectorize('date', 'quote', 'volatility')
    @option_default_values
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
        first_available_date = getattr(self.timeseries, MID_PRICE).ts_values.first_valid_index()
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

    def intrinsic(self, date, spot_price):
        """
        :param date: date-like
            The date
        :param spot_price: float
            The underlying spot price.
        :return: float
            The intrinsic value o the option at date.
        """
        if date > self._maturity:
            return 0
        else:
            intrinsic = 0
            if self.option_type == 'CALL':
                intrinsic = spot_price - self.strike
            elif self.option_type == 'PUT':
                intrinsic = self.strike - spot_price
            if intrinsic < 0:
                return 0
            else:
                return intrinsic
    
    def ts_mid_price(self, date, last_available=True, fill_value=np.nan):

        date = to_datetime(to_list(date))
        return getattr(self.timeseries, MID_PRICE).get_values(index=date, last_available=last_available,
                                                              fill_value=fill_value)

    def ts_implied_volatility(self, date, last_available=False, fill_value=np.nan):

        date = to_datetime(to_list(date))
        return getattr(self.timeseries, IMPLIED_VOL).get_values(index=date, last_available=last_available,
                                                                fill_value=fill_value)

    def _process_values_update(self, base_equity_process, date, base_date, spot_price, dividend_yield, dividend_tax):
        """
        This values are all common to the py:class:'BaseEquityProcess"
        :param date: QuantLib.Date
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            The underlying spot price.
        :param dividend_yield: float
            An override of the dividend yield in case you don't wan't to use the timeseries one.
        :param dividend_tax: float
            The dividend % tax applied.
        :return:
        """
        taxed_dividend_yield = dividend_yield * (1-float(dividend_tax))
        dividend_yield = ql.InterestRate(taxed_dividend_yield, self.day_counter, ql.Simple,
                                         ql.Annual).equivalentRate(ql.Continuous, ql.NoFrequency, 1).rate()
        if date > base_date:
            date = base_date
        base_equity_process.risk_free_handle.linkTo(self.yield_curve.yield_curve(date=date))
        base_equity_process.dividend_yield.setValue(dividend_yield)
        base_equity_process.spot_price.setValue(spot_price)

    def _black_implied_vol(self, date, option_price, spot_price, base_equity_process):
        """
        :param date: QuantLib.Date
            The date.
        :param spot_price: float
            The underlying spot price.
        :param option_price: float
            The option price used to calculate the implied volatility.
        """
        self._implied_volatility[date] = ql.SimpleQuote(0.2)
        process = base_equity_process.process(volatility=self._implied_volatility[date])
        self.set_pricing_engine(engine_name=self.engine_name, process=process)

        try:
            implied_vol = self.option.impliedVolatility(targetValue=option_price, process=process)
        except RuntimeError:
            # almost all errors are due to the option price being lower than the intrinsic value.
            discount_dvd = base_equity_process.dividend_handle.discount(self._maturity)
            discount_risk_free = base_equity_process.risk_free_handle.discount(self._maturity)
            fwd_spot_price = spot_price * discount_dvd / discount_risk_free
            option_price = self.intrinsic(date=date, spot_price=fwd_spot_price) + 0.01
            implied_vol = self.option.impliedVolatility(targetValue=option_price, process=process)

        self._implied_volatility[date].setValue(implied_vol)

    def volatility_update(self, date, base_date, spot_price, option_price, dividend_yield, dividend_tax, volatility,
                          base_equity_process, get_constants_from_ts=False, **kwargs):
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
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :param get_constants_from_ts: bool
            For Stochastic process that depend of constants use this bool to specify if you are passing the values
            explicitly or the values are to be retrieved from timeseries.
        :return:
        """
        self._process_values_update(base_equity_process=base_equity_process, date=date, base_date=base_date,
                                    spot_price=spot_price, dividend_yield=dividend_yield, dividend_tax=dividend_tax)
        if date > base_date:
            date = base_date
        if volatility is not None:
            self._implied_volatility[date] = ql.SimpleQuote(volatility)
        else:
            process_name = base_equity_process.process_name
            if process_name in [BLACK_SCHOLES, BLACK_SCHOLES_MERTON]:
                self._black_implied_vol(date=date, option_price=option_price, spot_price=spot_price,
                                        base_equity_process=base_equity_process)
                process = base_equity_process.process(volatility=self._implied_volatility[date])
            elif process_name in [HESTON, GJR_GARCH]:
                if get_constants_from_ts:
                    if process_name == HESTON:
                        variance, kappa, theta, sigma, rho = base_equity_process.get_constant_values(date=date)
                        process = base_equity_process.process(variance=variance, kappa=kappa, theta=theta, sigma=sigma,
                                                              rho=rho)
                    elif process_name == GJR_GARCH:
                        omega, alpha, beta, gamma, lambda_value = base_equity_process.get_constant_values(date=date)
                        v_zero = base_equity_process.v_zero(omega=omega, alpha=alpha, beta=beta, gamma=gamma,
                                                            lambda_value=lambda_value)
                        process = base_equity_process.process(omega=omega, alpha=alpha, beta=beta, gamma=gamma,
                                                              lambda_value=lambda_value, v_zero=v_zero)
                else:
                    process = base_equity_process.process(**kwargs)
        # making sure it is using the updated model.
        self.set_pricing_engine(engine_name=self.engine_name, process=process)

    @option_default_arguments
    @conditional_vectorize('date', 'volatility', 'spot_price')
    @option_default_values
    def price(self, date, base_date, spot_price, volatility, dividend_yield, dividend_tax, base_equity_process,
              **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option price at date.
        """
        if self.is_expired(date=date):
            return self.intrinsic(date=self._maturity, spot_price=spot_price)
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
        return self.option.NPV()

    @option_default_arguments
    @conditional_vectorize('spot_price')
    @option_default_values
    def price_underlying(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility,
                         base_equity_process, **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option price based on the date and underlying spot price.
        """
        if self.is_expired(date=date):
            return self.intrinsic(date=self._maturity, spot_price=spot_price)
        else:
            if date not in self._implied_volatility.keys():
                base_spot_price = float(self.underlying_instrument.spot_price(date=date, last_available=True))
                self.volatility_update(date=date, base_date=base_date, spot_price=base_spot_price,
                                       dividend_yield=dividend_yield, dividend_tax=dividend_tax,
                                       volatility=volatility, base_equity_process=base_equity_process, **kwargs)
            self.ql_process.spot_price.setValue(spot_price)
            return self.option.NPV()

    @option_default_arguments
    @conditional_vectorize('date', 'volatility', 'spot_price')
    @option_default_values
    def delta(self, date, base_date, spot_price, volatility, dividend_yield, dividend_tax, base_equity_process,
              **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option delta at date.
        """
        if self.is_expired(date=date):
            if self.intrinsic(date=date, spot_price=spot_price) > 0:
                return 1
            else:
                return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            return self.option.delta()

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def delta_underlying(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility,
                         base_equity_process, **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option delta based on the date and underlying spot price.
        """
        if self.is_expired(date=date):
            if self.intrinsic(date=date, spot_price=spot_price) > 0:
                return 1
            else:
                return 0
        else:
            base_spot_price = float(self.underlying_instrument.spot_price(date=date, last_available=True))
            self.volatility_update(date=date, base_date=base_date, spot_price=base_spot_price,
                                   dividend_yield=dividend_yield, dividend_tax=dividend_tax,
                                   volatility=volatility, base_equity_process=base_equity_process, **kwargs)
        self.ql_process.spot_price.setValue(spot_price)
        return self.option.delta()

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def gamma(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process,
              **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option gamma at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            return self.option.gamma()

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def theta(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process,
              **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option theta at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            return self.option.theta()

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def vega(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process,
             **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option vega at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.vega()
                except RuntimeError:
                    return 0

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def rho(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process, **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option rho at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.rho()
                except RuntimeError:
                    return 0

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'option_price')
    @option_default_values
    def implied_volatility(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility,
                           base_equity_process, option_price, **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :param option_price: float
            The option price to be used for implied vol calculation
        :return: float
            The option volatility based on the option price and date.
        """
        self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dividend_tax=dividend_tax, volatility=volatility,
                               base_equity_process=base_equity_process, option_price=option_price, **kwargs)
        return self._implied_volatility[date].value()

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def optionality(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process,
                    **kwargs):
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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option optionality at date.
        """
        if self.is_expired(date=date):
            return 0
        else:
            self.volatility_update(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                                   dividend_tax=dividend_tax, volatility=volatility,
                                   base_equity_process=base_equity_process, **kwargs)
            price = self.option.NPV()
            if date > base_date:
                date = base_date
            intrinsic = self.intrinsic(date=date, spot_price=spot_price)
            return price - intrinsic

    @option_default_arguments
    @conditional_vectorize('date')
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

    @option_default_arguments
    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_values
    def delta_value(self, date, base_date, spot_price, dividend_yield, dividend_tax, volatility, base_equity_process,
                    **kwargs):
        """ Return the option delta notional value

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
        :param volatility: float
            Volatility override value to calculate the option.
        :param base_equity_process: py:class:'BaseEquityProcess"
            The Stochastic process used for calculations.
        :return: float
            The option delta notional value.
        """
        delta = self.delta(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                           dividend_tax=dividend_tax, volatility=volatility,
                           base_equity_process=base_equity_process, **kwargs)
        return delta*spot_price*self.contract_size

    def heston_helper(self, date, volatility, base_equity_process, error_type=None):
        """ Heston Calibration Helper

        :param date: date-like
            The date.
        :param volatility: float
            The option volatility
        :param base_equity_process: py:class:'BaseEquityProcess'
            The Stochastic Process used for calibration.
        :param error_type: str
            The Calibration Error used in the estimations
        :return: QuantLib.HestonModelHelper
        """
        date = to_ql_date(date)
        days = self.calendar.businessDaysBetween(date, self.maturity(date), False, True)
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(volatility))
        spot_price = base_equity_process.spot_price_handle.value()

        error_type = str(error_type).upper()
        if error_type == 'IMPLIED_VOL':
            error = ql.BlackCalibrationHelper.ImpliedVolError
        elif error_type == 'PRICE_ERROR':
            error = ql.BlackCalibrationHelper.PriceError
        elif error_type == 'RELATIVE_PRICE_ERROR':
            error = ql.BlackCalibrationHelper.RelativePriceError
        else:
            error = ql.BlackCalibrationHelper.RelativePriceError

        return ql.HestonModelHelper(ql.Period(days, ql.Days), self.calendar, spot_price,
                                    self.strike, vol_handle, base_equity_process.risk_free_handle,
                                    base_equity_process.dividend_handle, error)
