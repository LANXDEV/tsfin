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

    Parameters
    ----------
    arg: date-like

    Returns
    -------
    QuantLib.Date

    """
    if isinstance(arg, ql.Date):
        return arg
    else:
        arg = pd.to_datetime(arg)
        return ql.Date(arg.day, arg.month, arg.year)


def to_ql_frequency(arg):
    """Converts string with a period representing a tenor to a QuantLib period.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Period

    """

    if arg.upper() == "ANNUAL":
        return ql.Annual
    elif arg.upper() == "SEMIANNUAL":
        return ql.Semiannual
    elif arg.upper() == "QUARTERLY":
        return ql.Quarterly
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
    else:
        raise ValueError("Unable to convert {} to a QuantLib frequency".format(arg))


def to_ql_calendar(arg):
    """Converts string with a calendar name to a calendar instance of QuantLib.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Calendar

    """

    if arg.upper() == "US":
        return ql.UnitedStates()
    elif arg.upper() in ["NYSE", "CE"]:
        return ql.UnitedStates(ql.UnitedStates.NYSE)
    elif arg.upper() == "FD":
        return ql.UnitedStates(ql.UnitedStates.FederalReserve)
    elif arg.upper() == "EX":
        return ql.JointCalendar(ql.UnitedStates(ql.UnitedStates.NYSE),
                                ql.UnitedStates(ql.UnitedStates.Settlement))
    elif arg.upper() == "GB":
        return ql.UnitedKingdom()
    elif arg.upper() == "BZ":
        return ql.Brazil()
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
    else:
        raise ValueError("Unable to convert {} to a QuantLib calendar".format(arg))


def to_ql_currency(arg):
    """Converts string with a calendar name to a calendar instance of QuantLib.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Currency

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

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.BusinessConvention

    """

    if arg.upper() == "FOLLOWING":
        return ql.Following
    elif arg.upper() == "MODIFIEDFOLLOWING":
        return ql.ModifiedFollowing
    elif arg.upper() == "UNADJUSTED":
        return ql.Unadjusted
    else:
        raise ValueError("Unable to convert {} to a QuantLib business convention".format(arg))


def to_ql_day_counter(arg):
    """Converts a string with day_counter name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.DayCounter

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

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.DateGeneration

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

    # Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Compounding

    """
    if arg.upper() == "COMPOUNDED":
        return ql.Compounded
    elif arg.upper() == "SIMPLE":
        return ql.Simple
    elif arg.upper() == "CONTINUOUS":
        return ql.Continuous
    else:
        raise ValueError("Unable to convert {} to a QuantLib compounding specification".format(arg))


def to_ql_overnight_index(arg):
    """Converts a string with overnight index name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.OvernightIndex

    """
    if arg.upper() == "FEDFUNDS":
        return ql.FedFunds()
    else:
        raise ValueError("Unable to convert {} to a QuantLib overnight index".format(arg))


def to_ql_rate_index(arg, tenor=None):
    """Converts a string with index name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str
    tenor: QuantLib.Period


    Returns
    -------
    QuantLib.Libor
    QuantLib.OvernightIndex

    """
    if arg.upper() == "USDLIBOR":
        if tenor is not None:
            return ql.USDLibor(tenor)
        else:
            return ql.USDLibor()
    elif arg.upper() == "FEDFUNDS":
        return ql.FedFunds()
    elif arg.upper() == "EURLIBOR":
        if tenor is not None:
            return ql.EURLibor(tenor)
        else:
            return ql.EURLibor()
    elif arg.upper() == "EONIA":
        if tenor is not None:
            return ql.Eonia(tenor)
        else:
            return ql.Eonia()
    else:
        raise ValueError("Unable to convert {} to a QuantLib index".format(arg))


def to_ql_duration(arg):
    """Converts a string with duration name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Duration
    """

    if arg.upper() == 'MODIFIED':
        return ql.Duration.Modified
    if arg.upper() == 'SIMPLE':
        return ql.Duration.Simple
    else:
        return ql.Duration.Macaulay


def to_ql_float_index(index, tenor, yield_curve_handle=None):
    """Return the QuantLib.Index with the specified tenor and yield_curve_handle.

    Parameters
    ----------
    index: str
        Index name
    tenor: QuantLib.Period
        The QuantLib object representing the tenor of the index.
    yield_curve_handle: QuantLib.YieldTermStructureHandle
        The QuantLib Yield Term Structure to be used in the projections.

    Returns
    -------
    QuantLib.Index

    """

    if index.upper() == "USDLIBOR":
        return ql.USDLibor(tenor, yield_curve_handle)
    elif index.upper() == "FEDFUNDS":
        return ql.FedFunds(tenor, yield_curve_handle)
    elif index.upper() == "EURLIBOR":
        return ql.EURLibor(tenor, yield_curve_handle)
    elif index.upper() == "EONIA":
        return ql.Eonia(tenor, yield_curve_handle)


def to_ql_ibor_index(index, tenor, fixing_days, currency, calendar, business_convention, end_of_month, day_counter,
                     yield_curve_handle):
    """Generic constructor of a QuantLib.Index. Mostly useful when you have to create custom calendars.

    Parameters
    ----------
    index: str
        Index name
    tenor: QuantLib.Period
        The QuantLib object representing the tenor of the index.
    fixing_days: float
        The number of days used in the fixing.
    currency: QuantLib.Currency
        The QuantLib object representing the target currency.
    end_of_month: bool
        End of month parameter of the index schedule.
    calendar: QuantLib.Calendar
        Calendar of the parent bond.
    business_convention: QuantLib.BusinessConvention
        The business convention rule.
    day_counter: QuantLib.DayCounter
        DayCounter of the index.
    yield_curve_handle: QuantLib.YieldTermStructureHandle
        The QuantLib Yield Term Structure to be used in the projections.

    Returns
    -------
    QuantLib.IborIndex

    """
    return ql.IborIndex(index, tenor, fixing_days, currency, calendar, business_convention, end_of_month, day_counter,
                        yield_curve_handle)


def to_ql_short_rate_model(arg):
    """Converts a string with the short rate model name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.ShortRateModel
    """

    if arg.upper() == 'HULL_WHITE':
        return ql.HullWhite
    elif arg.upper() == 'BLACK_KARASINSKI':
        return ql.BlackKarasinski
    elif arg.upper() == 'G2':
        return ql.G2


def ql_tenor_to_maturity_date(base_date, tenor):
    """Return the maturity date base on a initial date and a specified date.

    Parameters
    ----------
    base_date: QuantLib.Date
        The base date for calculation.
    tenor: str
        The string representing the tenor

    Returns
    -------
    QuantLib.Date
    """
    maturity_date = to_ql_date(base_date) + ql.PeriodParser.parse(tenor)
    return maturity_date


def to_ql_time_unit(arg):
    """Converts a string with a time unit name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.TimeUnit

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
