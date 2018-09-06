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

    if arg.upper() == "NYSE":
        return ql.UnitedStates(ql.UnitedStates.NYSE)
    if arg.upper() == "UK":
        return ql.UnitedKingdom()
    if arg.upper() == "BZ":
        return ql.Brazil()
    if arg.upper() == "TARGET":
        return ql.TARGET()
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
    if arg.upper() == "BRL":
        return ql.BRLCurrency()
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
    elif arg.upper() == "ACTUAL360":
        return ql.Actual360()
    elif arg.upper() == "ACTUAL365":
        return ql.Actual365Fixed()
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
    else:
        raise ValueError("Unable to convert {} to a QuantLib date generation specification".format(arg))


def to_ql_compounding(arg):
    """Converts a string with compounding convention name to the corresponding QuantLib object.

    Parameters
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


def to_ql_index(arg):
    """Converts a string with index name to the corresponding QuantLib object.

    Parameters
    ----------
    arg: str

    Returns
    -------
    QuantLib.Index

    """
    if arg.upper() == "USDLIBOR":
        return ql.USDLibor
    else:
        raise ValueError("Unable to convert {} to a QuantLib index".format(arg))


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
        return ql.FedFunds
    else:
        raise ValueError("Unable to convert {} to a QuantLib overnight index".format(arg))
