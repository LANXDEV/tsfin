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
    to_ql_day_counter, to_datetime


def option_exercise_type(exercise_type, date, maturity):
    if exercise_type.upper() == 'AMERICAN':
        return ql.AmericanExercise(to_ql_date(date), to_ql_date(maturity))
    elif exercise_type.upper() == 'EUROPEAN':
        return ql.EuropeanExercise(to_ql_date(maturity))
    else:
        raise ValueError('Exercise type not supported')


def ql_option_type(*args):

    return ql.VanillaOption(*args)


def ql_option_payoff(*args):

    return ql.PlainVanillaPayoff(*args)


def ql_option_engine(process):

    model = str("LR")
    time_steps = 801
    return ql.BinomialVanillaEngine(process, model, time_steps)


class BaseEquityOption(Instrument):
    """ Model for Equity Options using the Black Scholes Merton model.

    :param timeseries: :py:class:`TimeSeries`
        The TimeSeries representing the option.
    :param ql_process: :py:class:'BlackScholesMerton'
        A class used to handle the Black Scholes Merton model from QuantLib.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """

    def __init__(self, timeseries, ql_process):
        super().__init__(timeseries)
        self.opt_type = self.ts_attributes[OPTION_TYPE]
        self.strike = self.ts_attributes[STRIKE_PRICE]
        self.contract_size = self.ts_attributes[OPTION_CONTRACT_SIZE]
        self.option_maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.payoff = ql_option_payoff(to_ql_option_type(self.opt_type), self.strike)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.exercise_type = self.ts_attributes[EXERCISE_TYPE]
        self.underlying_instrument = self.ts_attributes[UNDERLYING_INSTRUMENT]
        self.ql_process = ql_process
        self.option = None

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

    @conditional_vectorize('date', 'quote')
    def value(self, date, base_date, quote=None, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, *args, **kwargs):
        """Try to deduce dirty value for a unit of the time series (as a financial instrument).

        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param quote: scalar, optional
            The quote.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
        size = float(self.contract_size)
        if quote is not None:
            return float(quote)*size
        else:
            return self.price(date=date, base_date=base_date, vol_last_available=vol_last_available,
                              dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                              exercise_ovrd=exercise_ovrd)*size

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
        first_available_date = self.ivol_mid.ts_values.first_valid_index()
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

    @conditional_vectorize('date')
    def intrinsic(self, date):
        """
        :param date: date-like
            The date
        :return: float
            The intrinsic value o the option at date.
        """
        strike = self.strike
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) > dt_maturity:
            return 0
        else:
            spot = self.ql_process.spot_price_handle.value()
            intrinsic = 0

            if self.opt_type == 'CALL':
                intrinsic = spot - strike
            if self.opt_type == 'PUT':
                intrinsic = strike - spot

            if intrinsic < 0:
                return 0
            else:
                return intrinsic

    @conditional_vectorize('date')
    def ql_option(self, date, exercise_ovrd=None):

        """
        :param date: date-like
            The date
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :return: QuantLib.VanillaOption
        """

        if exercise_ovrd is not None:
            self.exercise_type = exercise_ovrd.upper()

        exercise = option_exercise_type(self.exercise_type, date=date, maturity=self.option_maturity)

        return ql_option_type(self.payoff, exercise)

    def option_engine(self, date, vol_last_available=False, dvd_tax_adjust=1, last_available=True, exercise_ovrd=None,
                      volatility=None, underlying_price=None):

        """
        :param date: date-like
            The date.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: QuantLib.VanillaOption
            This method returns the VanillaOption with a QuantLib engine. Used for calculating the option values
            and greeks.
        """
        dt_date = to_datetime(date)
        self.option = self.ql_option(date=dt_date, exercise_ovrd=exercise_ovrd)
        vol_updated = self.ql_process.update_process(date=date, calendar=self.calendar,
                                                     day_counter=self.day_counter,
                                                     ts_option=self.timeseries,
                                                     maturity=self.option_maturity,
                                                     underlying_name=self.underlying_instrument,
                                                     vol_last_available=vol_last_available,
                                                     dvd_tax_adjust=dvd_tax_adjust,
                                                     last_available=last_available,
                                                     spot_price=underlying_price)

        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))

        if volatility is not None:
            self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                              ts_option=self.timeseries, underlying_name=self.underlying_instrument,
                                              vol_value=volatility)
            self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
            return self.option
        elif vol_updated:
            return self.option
        else:
            self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                              ts_option=self.timeseries, underlying_name=self.underlying_instrument,
                                              vol_value=0.2)
            mid_price = self.px_mid.get_values(index=dt_date, last_available=True)
            implied_vol = self.option.impliedVolatility(targetValue=mid_price, process=self.ql_process.bsm_process)
            self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                              ts_option=self.timeseries, underlying_name=self.underlying_instrument,
                                              vol_value=implied_vol)
            self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
            return self.option

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def price(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option price at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            if dt_maturity > to_datetime(base_date):
                return self.intrinsic(date=base_date)
            else:
                return self.intrinsic(date=dt_maturity)
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust,  last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
        return option.NPV()

    @conditional_vectorize('date', 'spot_price')
    def price_underlying(self, date, spot_price, base_date, vol_last_available=False, dvd_tax_adjust=1,
                         last_available=True, exercise_ovrd=None, volatility=None):
        """
        :param date: date-like
            The date.
        :param spot_price: float
            The underlying spot prices used for evaluation.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
            The option price based on the date and underlying spot price.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        if to_datetime(date) > to_datetime(base_date):
            option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd, volatility=volatility)
        else:
            option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd, volatility=volatility)

        self.ql_process.spot_price_update(date=date, underlying_name=self.underlying_instrument, spot_price=spot_price)
        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
        return option.NPV()

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def delta(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option delta at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            if self.intrinsic(date=dt_maturity) > 0:
                return 1
            else:
                return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            return option.delta()

    @conditional_vectorize('date', 'spot_price')
    def delta_underlying(self, date, spot_price, base_date, vol_last_available=False, dvd_tax_adjust=1,
                         last_available=True, exercise_ovrd=None, volatility=None):
        """
        :param date: date-like
            The date.
        :param spot_price: float
            The underlying spot prices used for evaluation.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
            The option delta based on the date and underlying spot price.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        if to_datetime(date) > to_datetime(base_date):
            option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd, volatility=volatility)
        else:
            option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd, volatility=volatility)

        self.ql_process.spot_price_update(date=date, underlying_name=self.underlying_instrument, spot_price=spot_price)
        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
        return option.delta()

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def gamma(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option gamma at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            return option.gamma()

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def theta(self, date,  base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option theta at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            return option.theta()

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def vega(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
             exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option vega at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return option.vega()
                except:
                    return 0

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def rho(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True, exercise_ovrd=None,
            volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option rho at date.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return option.rho()
                except:
                    return 0

    @conditional_vectorize('date', 'target', 'spot_price')
    def implied_vol(self, date, target, spot_price=None, vol_last_available=False, dvd_tax_adjust=1,
                    last_available=True, exercise_ovrd=None, volatility=None):
        """
        :param date: date-like
            The date.
        :param target: float
           The option price used to calculate the implied volatility.
        :param spot_price: float, optional
            The underlying spot prices used for evaluation.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
            The option volatility based on the option price and date.
        """
        if self.option is None:
            self.option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                             dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                             exercise_ovrd=exercise_ovrd, volatility=volatility)

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        if spot_price is not None:
            self.ql_process.spot_price_update(date=date, underlying_name=self.underlying_instrument,
                                              spot_price=spot_price)

        self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                          ts_option=self.timeseries, underlying_name=self.underlying_instrument,
                                          vol_value=0.2)
        implied_vol = self.option.impliedVolatility(target, self.ql_process.bsm_process)

        return implied_vol

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def optionality(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                    exercise_ovrd=None, volatility=None, underlying_price=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
        :param dvd_tax_adjust: float, default=1
            The multiplier used to adjust for dividend tax. For example, US dividend taxes are 30% so you pass 0.7.
        :param last_available: bool, optional
            Whether to use last available data in case dates are missing in ``quotes``.
        :param exercise_ovrd: str, optional
            Used to force the option model to use a specific type of option. Only working for American and European
            option types.
        :param volatility: float, optional
            Volatility override value to calculate the option.
        :param underlying_price: float, optional
            Underlying price override value to calculate the option.
        :return: float
            The option optionality at date.
        """
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            ql.Settings.instance().evaluationDate = to_ql_date(date)
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd, volatility=volatility,
                                            underlying_price=underlying_price)
            price = option.NPV()
            if to_datetime(date) > to_datetime(base_date):
                intrinsic = self.intrinsic(date=base_date)
            else:
                intrinsic = self.intrinsic(date=date)
            return price - intrinsic

    @conditional_vectorize('date')
    def underlying_price(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                         exercise_ovrd=None, volatility=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
            The option underlying spot price.
        """
        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            ql.Settings.instance().evaluationDate = to_ql_date(date)
            if to_datetime(date) > to_datetime(base_date):
                self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                   dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                   exercise_ovrd=exercise_ovrd, volatility=volatility)
            else:
                self.option_engine(date=date, vol_last_available=vol_last_available,
                                   dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                   exercise_ovrd=exercise_ovrd, volatility=volatility)

            return self.ql_process.spot_price_handle.value()

    @conditional_vectorize('date', 'volatility', 'underlying_price')
    def delta_value(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                    exercise_ovrd=None, volatility=None):
        """
        :param date: date-like
            The date.
        :param base_date: date-like
            When date is a future date base_date is the last date on the "present" used to estimate future values.
        :param vol_last_available: bool, optional
            Whether to use last available data in case dates are missing in volatility values.
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
        delta = self.delta(date=date, base_date=base_date, vol_last_available=vol_last_available,
                           dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, exercise_ovrd=exercise_ovrd,
                           volatility=volatility)
        spot = self.ql_process.spot_price_handle.value()
        size = float(self.contract_size)

        return delta*spot*size
