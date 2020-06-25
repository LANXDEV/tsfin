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
Functions for converting strings to QuantLib objects. Used to map attributes stored in the database to objects.
"""
import pandas as pd
import QuantLib as ql


def to_ql_date(arg):
    """Converts a string, datetime.datetime or numpy.datetime64 instance to ql.Date instance.

    :param arg: date-like
        The date  to be converted to a QuantLib date
    :return QuantLib.Date
    """
    if isinstance(arg, ql.Date):
        return arg
    else:
        arg = pd.to_datetime(arg)
        return ql.Date(arg.day, arg.month, arg.year)


def to_ql_frequency(arg):
    """Converts string with a period representing a tenor to a QuantLib period.

    :param arg: str
        The frequency name
    :return QuantLib.Frequency, QuantLib.Period

    """

    if arg.upper() == "ANNUAL":
        return ql.Annual
    elif arg.upper() == "SEMIANNUAL":
        return ql.Semiannual
    elif arg.upper() == "QUARTERLY":
        return ql.Quarterly
    elif arg.upper() == "EVERY_FOUR_MONTH":
        return ql.EveryFourthMonth
    elif arg.upper() == "BIMONTHLY":
        return ql.Bimonthly
    elif arg.upper() == "MONTHLY":
        return ql.Monthly
    elif arg.upper() == "AT_MATURITY":
        return ql.Once
    elif arg.upper() == "BIWEEKLY":
        return ql.Biweekly
    elif arg.upper() == "WEEKLY":
        return ql.Weekly
    elif arg.upper() == "DAILY":
        return ql.Daily
    elif arg.upper() == 'NOFREQUENCY':
        return ql.NoFrequency
    else:
        raise ValueError("Unable to convert {} to a QuantLib frequency".format(arg))


def to_ql_weekday(arg):
    """Converts string with a period representing a tenor to a QuantLib Weekday.

    :param arg: str
        The weekday name
    :return QuantLib.Weekday

    """
    arg = str(arg).upper()
    if arg == 'SUNDAY':
        return ql.Sunday
    elif arg == 'MONDAY':
        return ql.Monday
    elif arg == 'TUESDAY':
        return ql.Tuesday
    elif arg == 'WEDNESDAY':
        return ql.Wednesday
    elif arg == 'THURSDAY':
        return ql.Thursday
    elif arg == 'FRIDAY':
        return ql.Friday
    elif arg == 'SATURDAY':
        return ql.Saturday
    else:
        raise ValueError("Unable to convert {} to a QuantLib weekday".format(arg))


def to_ql_calendar(arg):
    """Converts string with a calendar name to a calendar instance of QuantLib.

    :param arg: str
        The Calendar 2 letter code, exceptions being TARGET, NYSE and NULL
    :return QuantLib.Calendar
    """

    if arg.upper() == "US":
        return ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    elif arg.upper() in ["NYSE", "CE"]:
        return ql.UnitedStates(ql.UnitedStates.NYSE)
    elif arg.upper() == "FD":
        return ql.UnitedStates(ql.UnitedStates.FederalReserve)
    elif arg.upper() == "EX":
        return ql.JointCalendar(ql.UnitedStates(ql.UnitedStates.NYSE),
                                ql.UnitedStates(ql.UnitedStates.Settlement))
    elif arg.upper() in ["GB", "UK"]:
        return ql.UnitedKingdom()
    elif arg.upper() == 'LS':
        return ql.UnitedKingdom(ql.UnitedKingdom.Exchange)
    elif arg.upper() == "BZ":
        return ql.Brazil(ql.Brazil.Settlement)
    elif arg.upper() == "B2":
        return ql.Brazil(ql.Brazil.Exchange)
    elif arg.upper() in ["TE", 'TARGET']:
        return ql.TARGET()
    elif arg.upper() in ['C%', 'C+']:
        return ql.China()
    elif arg.upper() == 'JN':
        return ql.Japan()
    elif arg.upper() == 'SZ':
        return ql.Switzerland()
    elif arg.upper() == 'AU':
        return ql.Australia()
    elif arg.upper() == 'SA':
        return ql.SouthAfrica()
    elif arg.upper() == 'TU':
        return ql.Turkey()
    elif arg.upper() == 'NULL':
        return ql.NullCalendar()
    else:
        raise ValueError("Unable to convert {} to a QuantLib calendar".format(arg))


def to_ql_currency(arg):
    """Converts string with a calendar name to a calendar instance of QuantLib.

    :param arg: str
        The currency 3 letter identifier
    :return QuantLib.Currency
    """

    if arg.upper() == "USD":
        return ql.USDCurrency()
    elif arg.upper() == "BRL":
        return ql.BRLCurrency()
    elif arg.upper() == "EUR":
        return ql.EURCurrency()
    elif arg.upper() == "GBP":
        return ql.GBPCurrency()
    elif arg.upper() == "AUD":
        return ql.AUDCurrency()
    elif arg.upper() == "JPY":
        return ql.JPYCurrency()
    elif arg.upper() == "TRY":
        return ql.TRYCurrency()
    elif arg.upper() == "ZAR":
        return ql.ZARCurrency()
    elif arg.upper() == "CHF":
        return ql.CHFCurrency()
    elif arg.upper() in ["CNY", "CNH"]:
        return ql.CNYCurrency()
    else:
        raise ValueError("Unable to convert {} to a QuantLib currency".format(arg))


def to_ql_business_convention(arg):
    """Converts a string with business convention name to the corresponding QuantLib object.

    :param arg: str
        The business convention name
    :return QuantLib.BusinessConvention

    """

    if arg.upper() == "FOLLOWING":
        return ql.Following
    elif arg.upper() == "MODIFIEDFOLLOWING":
        return ql.ModifiedFollowing
    elif arg.upper() == "PRECEDING":
        return ql.Preceding
    elif arg.upper() == "MODIFIEDPRECEDING":
        return ql.ModifiedPreceding
    elif arg.upper() == "UNADJUSTED":
        return ql.Unadjusted
    else:
        raise ValueError("Unable to convert {} to a QuantLib business convention".format(arg))


def to_ql_day_counter(arg):
    """Converts a string with day_counter name to the corresponding QuantLib object.

    :param arg: str
        The day count name
    :return QuantLib.DayCounter

    """
    if arg.upper() == "THIRTY360E":
        return ql.Thirty360(ql.Thirty360.European)
    elif arg.upper() == "THIRTY360":
        return ql.Thirty360()
    elif arg.upper() == "ACTUAL360":
        return ql.Actual360()
    elif arg.upper() == "ACTUAL365":
        return ql.Actual365Fixed()
    elif arg.upper() == "ACTUALACTUAL":
        return ql.ActualActual(ql.ActualActual.ISMA)
    elif arg.upper() == "ACTUALACTUALISMA":
        return ql.ActualActual(ql.ActualActual.ISMA)
    elif arg.upper() == "ACTUALACTUALISDA":
        return ql.ActualActual(ql.ActualActual.ISDA)
    elif arg.upper() == "BUSINESS252":
        return ql.Business252()
    else:
        raise ValueError("Unable to convert {} to a QuantLib day counter".format(arg))


def to_ql_date_generation(arg):
    """Converts a string with date_generation name to the corresponding QuantLib object.

    :param arg: str
        The Date generation rule name
    :return QuantLib.DateGeneration

    """
    if arg.upper() == "FORWARD":
        return ql.DateGeneration.Forward
    elif arg.upper() == "BACKWARD":
        return ql.DateGeneration.Backward
    elif arg.upper() == "CDS20IMM":
        return ql.DateGeneration.TwentiethIMM
    elif arg.upper() == "CDS2015":
        return ql.DateGeneration.CDS2015
    elif arg.upper() == "CDS":
        return ql.DateGeneration.CDS
    else:
        raise ValueError("Unable to convert {} to a QuantLib date generation specification".format(arg))


def to_ql_compounding(arg):
    """Converts a string with compounding convention name to the corresponding QuantLib object.

    :param arg: str
        The compounding type
    :return QuantLib.Compounding

    """
    if arg.upper() == "COMPOUNDED":
        return ql.Compounded
    elif arg.upper() == "SIMPLE":
        return ql.Simple
    elif arg.upper() == "CONTINUOUS":
        return ql.Continuous
    else:
        raise ValueError("Unable to convert {} to a QuantLib compounding specification".format(arg))


def to_ql_duration(arg):
    """Converts a string with duration name to the corresponding QuantLib object.

    :param arg: str
        The duration name
    :return QuantLib.Duration
        The QuantLib object representing a duration calculation
    """

    if arg.upper() == 'MODIFIED':
        return ql.Duration.Modified
    if arg.upper() == 'SIMPLE':
        return ql.Duration.Simple
    else:
        return ql.Duration.Macaulay


def to_ql_rate_index(index, tenor=None, yield_curve_handle=ql.YieldTermStructureHandle()):
    """Return the QuantLib.Index with the specified tenor and yield_curve_handle.

    :param index: str
        Index name
    :param tenor: QuantLib.Period
        The QuantLib object representing the tenor of the index.
    :param yield_curve_handle: QuantLib.YieldTermStructureHandle
        The QuantLib Yield Term Structure to be used in the projections.
    :return QuantLib.IborIndex, QuantLib.OvernightIndex
    """

    if index.upper() == "USDLIBOR":
        if tenor == ql.Period(1, ql.Days):
            return ql.USDLiborON(yield_curve_handle)
        else:
            return ql.USDLibor(tenor, yield_curve_handle)
    elif index.upper() == "FEDFUNDS":
        return ql.FedFunds(yield_curve_handle)
    elif index.upper() == "EURLIBOR":
        return ql.EURLibor(tenor, yield_curve_handle)
    elif index.upper() == "EONIA":
        return ql.Eonia(tenor, yield_curve_handle)


def to_ql_ibor_index(index, tenor=None, fixing_days=None, currency=None, calendar=None, business_convention=None,
                     end_of_month=None, day_counter=None, yield_curve_handle=None, overnight_index=False):
    """Generic constructor of a QuantLib.Index. Mostly useful when you have to create custom calendars.

    :param index: str
        Index name
    :param tenor: QuantLib.Period
        The QuantLib object representing the tenor of the index.
    :param fixing_days: float
        The number of days used in the fixing.
    :param currency: QuantLib.Currency
        The QuantLib object representing the target currency.
    :param end_of_month: bool
        End of month parameter of the index schedule.
    :param calendar: QuantLib.Calendar
        Calendar of the parent bond.
    :param business_convention: QuantLib.BusinessConvention
        The business convention rule.
    :param day_counter: QuantLib.DayCounter
        DayCounter of the index.
    :param yield_curve_handle: QuantLib.YieldTermStructureHandle
        The QuantLib Yield Term Structure to be used in the projections.
    :return QuantLib.IborIndex

    """
    if yield_curve_handle is None:
        yield_curve_handle = ql.YieldTermStructureHandle()
    if overnight_index:
        return ql.OvernightIndex(index, fixing_days, currency, calendar, day_counter, yield_curve_handle)
    return ql.IborIndex(index, tenor, fixing_days, currency, calendar, business_convention, end_of_month, day_counter,
                        yield_curve_handle)


def to_ql_short_rate_model(model_name):
    """ Return a QuantLib object representing a Short Rate Model

    :param model_name: str
        The model name ('BLACK_KARASINSKI', 'HULL_WHITE', 'G2')
    :return: QuantLib.ShortRateModel
    """

    model_name = str(model_name).upper()
    if model_name == 'HULL_WHITE':
        return ql.HullWhite
    elif model_name == 'BLACK_KARASINSKI':
        return ql.BlackKarasinski
    elif model_name == 'G2':
        return ql.G2
    else:
        return None


def ql_tenor_to_maturity_date(base_date, tenor):
    """Return the maturity date base on a initial date and a specified date.

    :param base_date: QuantLib.Date
        The base date for calculation.
    :param tenor: str
        The string representing the tenor
    :return QuantLib.Date
    """
    maturity_date = to_ql_date(base_date) + ql.PeriodParser.parse(tenor)
    return maturity_date


def to_ql_time_unit(arg):
    """Converts a string with a time unit name to the corresponding QuantLib object.

    :param arg: str
        A one letter string representing the time unit.
    :return QuantLib.TimeUnit
        The QuantLib object presenting the time unit

    """
    if arg.upper() == 'Y':
        return ql.Years
    elif arg.upper() == 'M':
        return ql.Months
    elif arg.upper() == 'W':
        return ql.Weeks
    elif arg.upper() == 'D':
        return ql.Days
    else:
        raise ValueError("Unable to convert {} to a QuantLib Time Unit".format(arg))


def to_ql_swaption_engine(model_name, model):
    """ Return the QuantLib swaption engine

    :param model_name: str
        The Swaption model name
    :param model: QuantLib.CalibratedModel
        The QuantLib.CalibratedModel already setup and dully linked to a curve to be passed to the engime
    :return: QuantLib.PricingEngine
        An object with the Swaption QuantLib Engine
    """

    model_name = str(model_name).upper()
    if model_name == 'BLACK_KARASINSKI':
        return ql.TreeSwaptionEngine(model, 100)
    elif model_name == 'HULL_WHITE':
        return ql.JamshidianSwaptionEngine(model)
    elif model_name == 'G2':
        return ql.G2SwaptionEngine(model, 10, 400)


def to_ql_option_type(arg):
    """Converts a string with the option type to the corresponding QuantLib object.

    :param arg: str
        The option type name, 'CALL'or 'PUT'
    :return QuantLib.Option.Call or QuantLib.Option.Put
        The QuantLib object that representing a Call or a Put
    """
    if arg.upper() == 'CALL':
        return ql.Option.Call
    elif arg.upper() == 'PUT':
        return ql.Option.Put


def to_ql_option_exercise_type(exercise_type, earliest_date, maturity):
    """ Returns the QuantLib object representing the exercise type.

    :param exercise_type: str
        The exercise name
    :param earliest_date: QuantLib.Date
        The earliest date of exercise
    :param maturity: QuantLib.Date
        The maturity / exercise date
    :return: QuantLib.Exercise
    """
    if exercise_type.upper() == 'AMERICAN':
        return ql.AmericanExercise(earliest_date, maturity)
    elif exercise_type.upper() == 'EUROPEAN':
        return ql.EuropeanExercise(maturity)
    else:
        raise ValueError('Exercise type not supported')


def to_ql_option_engine(engine_name=None, process=None, model=None):
    """ Returns a QuantLib.PricingEngine for Options

    :param engine_name: str
        The engine name
    :param process: QuantLib.StochasticProcess
        The QuantLib object with the option Stochastic Process.
    :param model: QuantLib.CalibratedModel
    :return: QuantLib.PricingEngine
    """
    if engine_name.upper() == 'BINOMIAL_VANILLA':
        return ql.BinomialVanillaEngine(process, 'LR', 801)
    elif engine_name.upper() == 'ANALYTIC_HESTON':
        if model is None:
            model = ql.HestonModel(process)
        elif model is not None and process is not None:
            model = model(process)
        return ql.AnalyticHestonEngine(model)
    elif engine_name.upper() == 'ANALYTIC_EUROPEAN':
        return ql.AnalyticEuropeanEngine(process)
    elif engine_name.upper() == 'ANALYTIC_EUROPEAN_DIVIDEND':
        return ql.AnalyticDividendEuropeanEngine(process)
    elif engine_name.upper() == "FINITE_DIFFERENCES":
        return ql.FdBlackScholesVanillaEngine(process)
    elif engine_name.upper() == 'HESTON_FINITE_DIFFERENCES':
        if model is None:
            model = ql.HestonModel(process)
        elif model is not None and process is not None:
            model = model(process)
        return ql.FdHestonVanillaEngine(model)
    elif engine_name.upper() == "BARONE_ADESI_WHALEY":
        return ql.BaroneAdesiWhaleyEngine(process)
    elif engine_name.upper() == "BJERKSUND_STENSLAND":
        return ql.BjerksundStenslandEngine(process)
    elif engine_name.upper() == "ANALYTIC_GJR_GARCH":
        if model is None:
            model = ql.GJRGARCHModel(process)
        elif model is not None and process is not None:
            model = model(process)
        return ql.AnalyticGJRGARCHEngine(model)
    elif engine_name.upper() == 'MC_GJR_GARCH':
        return ql.MCEuropeanGJRGARCHEngine(process=process, traits='pseudorandom', timeStepsPerYear=20,
                                           requiredTolerance=0.02)
    else:
        return None


def to_ql_equity_model(model_name):
    """ Return the QuantLib object representing an equity model

    :param model_name: str
        The model name. ('HESTON', 'GJR_GARCH', 'BATES')
    :return: QuantLib.CalibratedModel
    """
    model_name = str(model_name).upper()
    if model_name == "HESTON":
        return ql.HestonModel
    elif model_name == "GJR_GARCH":
        return ql.GJRGARCHModel
    elif model_name == "BATES":
        return ql.BatesModel


def to_ql_one_asset_option(payoff, exercise):
    """ Returns the QuantLib object representing an option.

    :param payoff: QuantLib.StrikedTypePayoff
        The QuantLib object representing the payoff
    :param exercise: QuantLib.Exercise
        The QuantLib Object representing the exercise type
    :return: QuantLib.OneAssetOption
    """
    return ql.VanillaOption(payoff, exercise)


def to_ql_option_payoff(payoff_type, ql_option_type, strike):
    """ Returns the QuantLib object representing an option payoff.

    :param payoff_type: str:
        The option payoff type name
    :param ql_option_type: QuantLib.Option.Call, QuantLib.Option.Put
        The option type (Call or Put)
    :param strike: float
        The strike value
    :return: QuantLib.StrikedTypePayoff
    """
    if str(payoff_type).upper() == 'PLAIN_VANILLA':
        return ql.PlainVanillaPayoff(ql_option_type, strike)


def to_ql_protection_side(side):
    """ Return the QuantLib object representing the protection side of a CDS deal

    :param side: str
        The Credit Default deal side, 'BUY' or 'SELL'
    :return: QuantLib.Protection
    """
    side = str(side).upper()
    if side == "BUY":
        return ql.Protection.Buyer
    elif side == "SELL":
        return ql.Protection.Seller
    else:
        return None


def to_ql_cds_engine(engine_name):
    """ Return the QuantLib CDS engine

    :param engine_name: str
        The engine name 'MID_POINT', 'ISDA', 'INTEGRAL'
    :return: QuantLib.PricingEngine
    """
    engine_name = str(engine_name).upper()
    if engine_name == "MID_POINT":
        return ql.MidPointCdsEngine
    elif engine_name == "ISDA":
        return ql.IsdaCdsEngine
    elif engine_name == "INTEGRAL":
        return ql.IntegralCdsEngine
    else:
        return None


def to_ql_fitting_method(fitting_method, constraint_at_zero=True):
    """
    Parameters
    ----------
    fitting_method: str
       the fitting method
    constraint_at_zero: bool
    Returns
    -------
        QuantLib.FittingMethod
    """
    fitting_method = str(fitting_method).lower()
    if fitting_method == "exponential_splines":
        return ql.ExponentialSplinesFitting(constraint_at_zero)
    elif fitting_method == "nelson_siegel":
        return ql.NelsonSiegelFitting()
    elif fitting_method == "svensson":
        return ql.SvenssonFitting()
    elif fitting_method == 'second_degree_polynomial':
        return ql.SimplePolynomialFitting(2, constraint_at_zero)
    elif fitting_method == 'third_degree_polynomial':
        return ql.SimplePolynomialFitting(3, constraint_at_zero)


def to_ql_piecewise_curve(helpers, calendar, day_counter, curve_type, constraint_at_zero=True):
    """
    Parameters
    ----------
    helpers: list of QuantLib.RateHelper
        The Rate Helpers of the instruments used in the piecewise interpolation.
    calendar: QuantLib.Calendar
        Calendar for the yield curves.
    day_counter: QuantLib.DayCounter
        The curve day count
    curve_type: str
        The curve interpolation type.
    constraint_at_zero: bool
    Returns
    -------
        QuantLib.PiecewiseCurve
    """
    curve_type = str(curve_type).lower()
    if curve_type == "linear_zero":
        return ql.PiecewiseLinearZero(0, calendar, helpers, day_counter)
    elif curve_type == "cubic_zero":
        return ql.PiecewiseCubicZero(0, calendar, helpers, day_counter)
    elif curve_type == "log_cubic_discount":
        return ql.PiecewiseLogCubicDiscount(0, calendar, helpers, day_counter)
    elif curve_type == "log_linear_discount":
        return ql.PiecewiseLogLinearDiscount(0, calendar, helpers, day_counter)
    elif curve_type == "spline_cubic_discount":
        return ql.PiecewiseSplineCubicDiscount(0, calendar, helpers, day_counter)
    elif curve_type == 'kruger_zero':
        return ql.PiecewiseKrugerZero(0, calendar, helpers, day_counter)
    elif curve_type == 'convex_monotone_zero':
        return ql.PiecewiseConvexMonotoneZero(0, calendar, helpers, day_counter)
    elif curve_type in ["exponential_splines", "nelson_siegel", "svensson", "second_degree_polynomial",
                        "third_degree_polynomial"]:
        fitting_method = to_ql_fitting_method(fitting_method=curve_type, constraint_at_zero=constraint_at_zero)
        return ql.FittedBondDiscountCurve(0, calendar, helpers, day_counter, fitting_method)
    else:
        return ql.PiecewiseLinearZero(0, calendar, helpers, day_counter)


def to_ql_interpolated_curve(node_dates, node_rates, day_counter, calendar, interpolation_type):

    """
    Parameters
    ----------
    node_dates: list
        The list of dates
    node_rates: list
        The list of rates
    day_counter: QuantLib.DayCounter
        The curve day count
    calendar: QuantLib.Calendar
        The curve calendar.
    interpolation_type: str
        The curve interpolation method
    Returns
    -------
        QuantLib.YieldTermStructure
    """
    interpolation_type = str(interpolation_type).lower()
    if interpolation_type == "cubic_zero":
        return ql.CubicZeroCurve(node_dates, node_rates, day_counter, calendar)
    elif interpolation_type == "log_linear_zero":
        return ql.LogLinearZeroCurve(node_dates, node_rates, day_counter, calendar)
    elif interpolation_type == "log_cubic_zero":
        return ql.LogCubicZeroCurve(node_dates, node_rates, day_counter, calendar)
    elif interpolation_type == "spline_cubic_zero":
        return ql.NaturalCubicZeroCurve(node_dates, node_rates, day_counter, calendar)
    elif interpolation_type == "monotonic_cubic_zero":
        return ql.MonotonicCubicZeroCurve(node_dates, node_rates, day_counter, calendar)
    else:
        return ql.ZeroCurve(node_dates, node_rates, day_counter, calendar)


def to_ql_position_side(side):
    """Return the QuantLib object representing if a position is long or short

    :param side: str
        The position side, LONG or SHORT
    :return: QuantLib.Position
    """

    side = str(side).upper()
    if side == 'LONG':
        return ql.Position.Long
    elif side == 'SHORT':
        return ql.Position.Short

