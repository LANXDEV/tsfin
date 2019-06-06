
import QuantLib as ql
import numpy as np
from tsfin.constants import CALENDAR, MATURITY_DATE, \
    DAY_COUNTER, EXERCISE_TYPE, OPTION_TYPE, STRIKE_PRICE, UNDERLYING_INSTRUMENT, OPTION_CONTRACT_SIZE
from tsfin.base.instrument import default_arguments
from tsfin.base import Instrument, to_ql_option_type, to_ql_date, conditional_vectorize, \
    to_ql_calendar, to_ql_day_counter, to_datetime


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

        ql_date = to_ql_date(date)
        if ql_date >= self.option_maturity:
            return True
        return False

    def maturity(self, date, *args, **kwargs):

        return self.option_maturity

    @conditional_vectorize('date', 'quote')
    def value(self, date, base_date, quote=None, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None, *args, **kwargs):

        size = float(self.contract_size)
        if quote is not None:
            return float(quote)*size
        else:
            return self.price(date=date, base_date=base_date, vol_last_available=vol_last_available,
                              dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                              exercise_ovrd=exercise_ovrd)*size

    @conditional_vectorize('date', 'quote')
    def performance(self, date=None, quote=None, start_date=None, start_quote=None, *args, **kwargs):

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

    @conditional_vectorize('date')
    def cash_to_date(self, start_date, date, *args, **kwargs):

        return 0

    def notional(self):

        return float(self.contract_size)*float(self.strike)

    def intrinsic(self, date):

        strike = self.strike
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
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

    def ql_option(self, date, exercise_ovrd=None):

        if exercise_ovrd is not None:
            self.exercise_type = exercise_ovrd.upper()

        exercise = option_exercise_type(self.exercise_type, date=date, maturity=self.option_maturity)

        return ql_option_type(self.payoff, exercise)

    @conditional_vectorize('date')
    def option_engine(self, date, vol_last_available=False, dvd_tax_adjust=1, last_available=True, exercise_ovrd=None):

        dt_date = to_datetime(date)
        self.option = self.ql_option(date=dt_date, exercise_ovrd=exercise_ovrd)
        vol_updated = self.ql_process.update_process(date=date, calendar=self.calendar,
                                                     day_counter=self.day_counter,
                                                     ts_option_name=self.ts_name,
                                                     maturity=self.option_maturity,
                                                     underlying_name=self.underlying_instrument,
                                                     vol_last_available=vol_last_available,
                                                     dvd_tax_adjust=dvd_tax_adjust,
                                                     last_available=last_available)

        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))

        if vol_updated:
            return self.option
        else:
            self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                              ts_option_name=self.ts_name, underlying_name=self.underlying_instrument,
                                              vol_value=0.2)
            mid_price = self.px_mid.get_values(index=dt_date, last_available=True)
            implied_vol = self.option.impliedVolatility(targetValue=mid_price, process=self.ql_process.bsm_process)
            self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                              ts_option_name=self.ts_name, underlying_name=self.underlying_instrument,
                                              vol_value=implied_vol)
            self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
            return self.option

    @conditional_vectorize('date')
    def price(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None):

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
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust,  last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
        return option.NPV()

    @conditional_vectorize('date', 'spot_price')
    def price_underlying(self, date, spot_price, base_date, vol_last_available=False, dvd_tax_adjust=1,
                         last_available=True, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        if to_datetime(date) > to_datetime(base_date):
            option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd)
        else:
            option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd)

        self.ql_process.spot_price_update(date=date, underlying_name=self.underlying_instrument, spot_price=spot_price)
        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
        return option.NPV()

    @conditional_vectorize('date')
    def delta(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None):

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
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            return option.delta()

    @conditional_vectorize('date', 'spot_price')
    def delta_underlying(self, date, spot_price, base_date, vol_last_available=False, dvd_tax_adjust=1,
                         last_available=True, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        if to_datetime(date) > to_datetime(base_date):
            option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd)
        else:
            option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                        dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                        exercise_ovrd=exercise_ovrd)

        self.ql_process.spot_price_update(date=date, underlying_name=self.underlying_instrument, spot_price=spot_price)
        self.option.setPricingEngine(ql_option_engine(self.ql_process.bsm_process))
        return option.delta()

    @conditional_vectorize('date')
    def gamma(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            return option.gamma()

    @conditional_vectorize('date')
    def theta(self, date,  base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
              exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            return option.theta()

    @conditional_vectorize('date')
    def vega(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1,last_available=True,
             exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return option.vega()
                except:
                    return 0

    @conditional_vectorize('date')
    def rho(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            if self.exercise_type == 'AMERICAN':
                return None
            else:
                try:
                    return option.rho()
                except:
                    return 0

    @conditional_vectorize('date', 'target')
    def implied_vol(self, date, target, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                    exercise_ovrd=None):

        if self.option is None:
            self.option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                             dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                             exercise_ovrd=exercise_ovrd)

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        self.ql_process.volatility_update(date=date, calendar=self.calendar, day_counter=self.day_counter,
                                          ts_option_name=self.ts_name, underlying_name=self.underlying_instrument,
                                          vol_value=0.2)
        implied_vol = self.option.impliedVolatility(target, self.ql_process.bsm_process)

        return implied_vol

    @conditional_vectorize('date')
    def optionality(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                    exercise_ovrd=None):

        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            ql.Settings.instance().evaluationDate = to_ql_date(date)
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            price = option.NPV()
            if to_datetime(date) > to_datetime(base_date):
                intrinsic = self.intrinsic(date=base_date)
            else:
                intrinsic = self.intrinsic(date=date)
            return price - intrinsic

    @conditional_vectorize('date')
    def underlying_price(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                         exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        dt_maturity = to_datetime(self.option_maturity)
        if to_datetime(date) >= dt_maturity:
            return 0
        else:
            ql.Settings.instance().evaluationDate = to_ql_date(date)
            if to_datetime(date) > to_datetime(base_date):
                option = self.option_engine(date=base_date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)
            else:
                option = self.option_engine(date=date, vol_last_available=vol_last_available,
                                            dvd_tax_adjust=dvd_tax_adjust, last_available=last_available,
                                            exercise_ovrd=exercise_ovrd)

            return self.ql_process.spot_price_handle.value()

    @conditional_vectorize('date')
    def delta_value(self, date, base_date, vol_last_available=False, dvd_tax_adjust=1, last_available=True,
                    exercise_ovrd=None):

        delta = self.delta(date=date, base_date=base_date, vol_last_available=vol_last_available,
                           dvd_tax_adjust=dvd_tax_adjust, last_available=last_available, exercise_ovrd=exercise_ovrd)
        spot = self.ql_process.spot_price_handle.value()
        size = float(self.contract_size)

        return delta*spot*size
