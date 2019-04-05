
import QuantLib as ql
from tsfin.constants import CALENDAR, MATURITY_DATE, \
    DAY_COUNTER, EXERCISE_TYPE, OPTION_TYPE, STRIKE_PRICE, UNDERLYING_INSTRUMENT
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


class BaseEquityOption(Instrument):

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.opt_type = to_ql_option_type(self.ts_attributes[OPTION_TYPE])
        self.strike = self.ts_attributes[STRIKE_PRICE]
        self.option_maturity = to_ql_date(to_datetime(self.ts_attributes[MATURITY_DATE]))
        self.payoff = ql.PlainVanillaPayoff(self.opt_type, self.strike)
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        self.exercise_type = self.ts_attributes[EXERCISE_TYPE]
        self.underlying_instrument = self.ts_attributes[UNDERLYING_INSTRUMENT]
        self.option = None

    def is_expired(self, date, *args, **kwargs):

        ql_date = to_ql_date(date)
        if ql_date >= self.option_maturity:
            return True
        return False

    def maturity(self, date, *args, **kwargs):

        return self.option_maturity

    @conditional_vectorize('date')
    def option_engine(self, date, ql_process, exercise_ovrd=None):

        dt_date = to_datetime(date)
        if exercise_ovrd is not None:
            self.exercise_type = exercise_ovrd.upper()

        if self.exercise_type.upper() == 'AMERICAN':

            exercise = ql.AmericanExercise(to_ql_date(dt_date), self.option_maturity)
        elif self.exercise_type.upper() == 'EUROPEAN':

            exercise = ql.EuropeanExercise(self.option_maturity)
        else:
            raise ValueError('Type not supported')

        option = ql.VanillaOption(self.payoff, exercise)
        bsm_process = ql_process.process
        try:
            bsm_process_at_date = bsm_process["({})(IVOL_MID)".format(self.ts_name)][dt_date]
            option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process_at_date, "crr", 200))
        except KeyError:
            mid_price = self.px_mid.ts_values.loc[to_datetime(dt_date)]
            ivol_last = self.ivol_mid.ts_values.last('1D').values
            bsm_process = ql_process.update_missing_vol(date=dt_date, vol_value=ivol_last,
                                                        ts_name="({})(IVOL_MID)".format(self.ts_name)).process
            bsm_process_at_date = bsm_process["({})(IVOL_MID)".format(self.ts_name)][dt_date]
            option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process_at_date, "crr", 200))
            implied_vol = option.impliedVolatility(targetValue=mid_price, process=bsm_process_at_date)
            bsm_process = ql_process.update_only_vol(date=dt_date, vol_value=implied_vol*100,
                                                     ts_name="({})(IVOL_MID)".format(self.ts_name)).process
            bsm_process_at_date = bsm_process["({})(IVOL_MID)".format(self.ts_name)][dt_date]
            option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process_at_date, "crr", 200))

        return option

    @conditional_vectorize('date')
    def price(self, date, ql_process, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        option = self.option_engine(date=date, ql_process=ql_process, exercise_ovrd=exercise_ovrd)
        return option.NPV()

    @conditional_vectorize('date')
    def delta(self, date, ql_process, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        option = self.option_engine(date=date, ql_process=ql_process, exercise_ovrd=exercise_ovrd)
        return option.delta()

    @conditional_vectorize('date')
    def gamma(self, date, ql_process, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        option = self.option_engine(date=date, ql_process=ql_process, exercise_ovrd=exercise_ovrd)
        return option.gamma()

    @conditional_vectorize('date')
    def vega(self, date, ql_process, exercise_ovrd=None):

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        option = self.option_engine(date=date, ql_process=ql_process, exercise_ovrd=exercise_ovrd)
        if isinstance(self.exercise, ql.AmericanExercise):
            return None
        else:
            return option.vega()
