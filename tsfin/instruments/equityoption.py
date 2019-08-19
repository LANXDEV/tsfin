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
from tsfin.constants import CALENDAR, MATURITY_DATE, \
    DAY_COUNTER, EXERCISE_TYPE, OPTION_TYPE, STRIKE_PRICE, UNDERLYING_INSTRUMENT, OPTION_CONTRACT_SIZE
from tsfin.base import Instrument, to_ql_option_type, to_ql_date, conditional_vectorize, to_ql_calendar, \
    to_ql_day_counter, to_datetime, to_list


def option_default_arguments(f):
    """ Decorator to set default arguments for :py:class:`BaseEquityOption` methods.

    Parameters
    ----------
    f: method
        A method to be increased with default arguments.

    Returns
    -------
    function
        `f`, increased with default arguments.
    """
    def new_f(self, **kwargs):
        try:
            date = to_ql_date(kwargs['date'][0])
        except:
            date = to_ql_date(kwargs['date'])
        if 'last_available' not in kwargs.keys():
            kwargs['last_available'] = True
        if kwargs.get('exercise_type', None) is None:
            kwargs['exercise_type'] = self.exercise_type.upper()
        if 'dvd_tax_adjust' not in kwargs.keys():
            kwargs['dvd_tax_adjust'] = 1
        if 'spot_price' not in kwargs.keys():
            kwargs['spot_price'] = None
        if kwargs.get('spot_price', None) is None:
            spot_price = self.underlying_instrument.spot_price(date=date, last_available=kwargs['last_available'])
            kwargs['spot_price'] = spot_price
        if kwargs.get('dividend_yield', None) is None:
            dvd_yield = self.underlying_instrument.dividend_yield(date=date, last_available=kwargs['last_available'])
            kwargs['dividend_yield'] = dvd_yield
        if 'volatility' not in kwargs.keys():
            kwargs['volatility'] = None
        exercise = option_exercise_type(kwargs['exercise_type'], date=date, maturity=self.option_maturity)
        self.option = ql_option_type(self.payoff, exercise)
        return f(self, **kwargs)
    return new_f


def option_exercise_type(exercise_type, date, maturity):
    if exercise_type.upper() == 'AMERICAN':
        if date > maturity:
            date = maturity
        return ql.AmericanExercise(date, maturity)
    elif exercise_type.upper() == 'EUROPEAN':
        return ql.EuropeanExercise(maturity)
    else:
        raise ValueError('Exercise type not supported')


def ql_option_type(*args):

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
        self.opt_type = self.ts_attributes[OPTION_TYPE]
        self.strike = self.ts_attributes[STRIKE_PRICE]
        self.contract_size = self.ts_attributes[OPTION_CONTRACT_SIZE]
        self.option_maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.payoff = ql_option_payoff(to_ql_option_type(self.opt_type), self.strike)
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

    def is_expired(self, date, *args, **kwargs):
        """
        :param date: date-like
            The date.
        :return bool
            True if the instrument is expired or matured, False otherwise.
        """
        ql_date = to_ql_date(date)
        if ql_date >= self.option_maturity:
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
    def value(self, date, base_date, quote=None, volatility=None, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, *args, **kwargs):
        """Try to deduce dirty value for a unit of the time series (as a financial instrument).

        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param quote: scalar, optional
            The quote.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :return scalar, None
            The unit dirty value of the instrument.
        """
        ql_date = to_ql_date(date)
        ql_base_date = to_ql_date(base_date)
        size = float(self.contract_size)
        if quote is not None:
            return float(quote)*size
        else:
            return self.price(date=ql_date, base_date=ql_base_date, dvd_tax_adjust=dvd_tax_adjust,
                              last_available=last_available, exercise_ovrd=exercise_ovrd, volatility=volatility)*size

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
        return float(self.contract_size)*float(self.strike)

    def intrinsic(self, date):
        """
        :param date: date-like
            The date
        :return: float
            The intrinsic value o the option at date.
        """
        ql_date = to_ql_date(date)
        strike = self.strike
        if ql_date > self.option_maturity:
            return 0
        else:
            spot = float(self.underlying_instrument.spot_price(date=date, last_available=True))
            intrinsic = 0

            if self.opt_type == 'CALL':
                intrinsic = spot - strike
            if self.opt_type == 'PUT':
                intrinsic = strike - spot

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

    def option_engine(self, date, base_date, spot_price, dividend_yield, dvd_tax_adjust, volatility, **kwargs):

        """
        :param date: QuantLib.Date
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            The underlying spot price.
        :param dividend_yield: float
            An override of the dividend yield in case you don't wan't to use the timeseries one.
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: QuantLib.VanillaOption
            This method returns the VanillaOption with a QuantLib engine. Used for calculating the option values
            and greeks.
        """
        self.ql_process.update_process(date=date,
                                       base_date=base_date,
                                       spot_price=spot_price,
                                       dividend_yield=dividend_yield,
                                       calendar=self.calendar,
                                       day_counter=self.day_counter,
                                       vol_value=0.2,
                                       maturity=self.option_maturity,
                                       dvd_tax_adjust=dvd_tax_adjust)
        self.option.setPricingEngine(self.ql_process.ql_engine(engine_name='BINOMIAL_VANILLA',
                                                               process=self.ql_process.process,
                                                               model_name="LR", time_steps=801))

        if date > base_date:
            date = base_date
        if volatility is not None:
            self.implied_volatility[date] = ql.SimpleQuote(volatility)
        try:
            self.ql_process.volatility_update(vol_value=self.implied_volatility[date], calendar=self.calendar,
                                              day_counter=self.day_counter)
        except KeyError:
            mid_price = float(self.ts_mid_price(date=date, last_available=True))
            try:
                implied_vol = self.option.impliedVolatility(targetValue=mid_price,
                                                            process=self.ql_process.process,
                                                            accuracy=1.0e-4, maxEvaluations=100)
            except RuntimeError:
                prior_date = self.calendar.advance(date, -1, ql.Days)
                mid_price = float(self.ts_mid_price(date=prior_date, last_available=True))
                implied_vol = self.option.impliedVolatility(targetValue=mid_price,
                                                            process=self.ql_process.process,
                                                            accuracy=1.0e-4, maxEvaluations=100)
            self.implied_volatility[date] = ql.SimpleQuote(implied_vol)
            self.ql_process.volatility_update(vol_value=self.implied_volatility[date], calendar=self.calendar,
                                              day_counter=self.day_counter)

    @conditional_vectorize('date', 'volatility', 'spot_price')
    @option_default_arguments
    def price(self, date, base_date, spot_price, dividend_yield, dvd_tax_adjust, last_available, volatility,
              exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option price at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            if self.option_maturity >= base_date:
                return self.intrinsic(date=self.option_maturity)
            else:
                return self.intrinsic(date=base_date)
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
        return self.option.NPV()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def price_underlying(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
                         exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option price based on the date and underlying spot price.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                           dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                           **kwargs)

        self.ql_process.spot_price_update(spot_price=spot_price)
        return self.option.NPV()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def delta(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
              exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option delta at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            if self.intrinsic(date=self.option_maturity) > 0:
                return 1
            else:
                return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            return self.option.delta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def delta_underlying(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
                         exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option delta based on the date and underlying spot price.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                           dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                           **kwargs)
        self.ql_process.spot_price_update(spot_price=spot_price)
        return self.option.delta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def gamma(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
              exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option gamma at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            return self.option.gamma()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def theta(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
              exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option theta at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            return self.option.theta()

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def vega(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
             exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option vega at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.vega()
                except:
                    return 0

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def rho(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
              exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option rho at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        ql.Settings.instance().evaluationDate = date
        if date >= self.option_maturity:
            return 0
        else:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return self.option.rho()
                except:
                    return 0

    @conditional_vectorize('date', 'spot_price', 'target')
    @option_default_arguments
    def implied_vol(self, date, base_date, target, spot_price, dividend_yield,  dvd_tax_adjust, last_available,
                    volatility, exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param target: float
            The option price target.
        :param spot_price: float, optional
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.

        :return: float
            The option volatility based on the option price and date.
        """
        date = to_ql_date(date)
        ql.Settings.instance().evaluationDate = date
        if self.option is None:
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
        if spot_price is not None:
            self.ql_process.spot_price_update(spot_price=spot_price)

        return self.option.impliedVolatility(target, self.ql_process.process)

    @conditional_vectorize('date', 'spot_price', 'volatility')
    @option_default_arguments
    def optionality(self, date, base_date, spot_price, dividend_yield, dvd_tax_adjust, last_available, volatility,
                    exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float
            Volatility override value to calculate the option.
        :return: float
            The option optionality at date.
        """
        date = to_ql_date(date)
        base_date = to_ql_date(base_date)
        if date >= self.option_maturity:
            return 0
        else:
            ql.Settings.instance().evaluationDate = date
            self.option_engine(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                               dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, volatility=volatility,
                               **kwargs)
            price = self.option.NPV()
            if date > base_date:
                intrinsic = self.intrinsic(date=base_date)
            else:
                intrinsic = self.intrinsic(date=date)
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
    def delta_value(self, date, base_date, spot_price, dividend_yield,  dvd_tax_adjust, last_available, volatility,
                    exercise_ovrd=None, **kwargs):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param spot_price: float, optional
            Underlying price override value to calculate the option.
        :param dividend_yield: float
            The dividend yield of the underlying instrument
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :return: float
            The option delta notional value.
        """
        delta = self.delta(date=date, base_date=base_date, spot_price=spot_price, dividend_yield=dividend_yield,
                           dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, exercise_ovrd=exercise_ovrd,
                           volatility=volatility, **kwargs)
        spot = self.underlying_instrument.spot_price(date=date)
        size = float(self.contract_size)
        return delta*spot*size
