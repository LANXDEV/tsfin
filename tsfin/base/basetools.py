"""
Basic independent tools that can be imported by any module in the package.
"""
from functools import wraps
from bisect import bisect_right
import time
from datetime import datetime
from datetime import time as dtime
import numpy as np
import pandas as pd
import QuantLib as ql


def to_datetime(arg):
    """Converts a QuantLib.Date instance to datetime.datetime instance.

    Parameters
    ----------
    arg: date-like

    Returns
    -------
    datetime.datetime

    """
    if isvectorizable(arg):
        try:
            # Works if all elements in arg are QuantLib.Date objects.
            return [datetime(day=arg_n.dayOfMonth(), month=arg_n.month(), year=arg_n.year()) for arg_n in arg]
        except AttributeError:
            return pd.to_datetime(arg)
    else:
        # arg is not vectorizable.
        try:
            # Works if arg is a ql.Date object.
            return datetime(day=arg.dayOfMonth(), month=arg.month(), year=arg.year())
        except AttributeError:
            return pd.to_datetime(arg)


def collapse_intraday_ts_values(ts_list, initial_date=None, final_date=None):
    """Drop (inplace) intraday data from TimeSeries' ts_values.

    Parameters
    ----------
    ts_list: list-like of TimeSeries objects
    initial_date: datetime.datetime
    final_date: datetime.datetime

    """
    no_time = dtime(0, 0)
    if initial_date is None and final_date is None:
        for ts in ts_list:
            last_date = ts.ts_values.last_valid_index()
            if last_date:
                ts.ts_values = ts.ts_values[(ts.ts_values.index.time == no_time) |
                                            (ts.ts_values.index == last_date)]
    elif initial_date is not None and final_date is None:
        for ts in ts_list:
            last_date = ts.ts_values.last_valid_index()
            if last_date:
                ts.ts_values = ts.ts_values[(ts.ts_values.index.time == no_time) |
                                            (ts.ts_values.index == last_date) |
                                            (ts.ts_values.index <= initial_date)]
    elif initial_date is None and final_date is not None:
        for ts in ts_list:
            last_date = ts.ts_values.last_valid_index()
            if last_date:
                ts.ts_values = ts.ts_values[(ts.ts_values.index.time == no_time) |
                                            (ts.ts_values.index == last_date) |
                                            (ts.ts_values.index >= final_date)]
    else:
        for ts in ts_list:
            last_date = ts.ts_values.last_valid_index()
            if last_date:
                ts.ts_values = ts.ts_values[(ts.ts_values.index.time == no_time) |
                                            (ts.ts_values.index == last_date) |
                                            (ts.ts_values.index <= initial_date) |
                                            (ts.ts_values.index >= final_date)]


def rebase_ts_values(union=None, intersection=None, dates=None, last=False, initial_date=None, final_date=None,
                     collapse_intraday=False, collapse_intraday_initial_date=None, collapse_intraday_final_date=None,
                     **kwargs):
    """Reindex (inplace) TimeSeries's ts_values so that their index coincides.

    Parameters
    ----------
    union: list-like of TimeSeries objects, optional
        The union of these TimeSeries' indexes will be used to build the final index.
    intersection: list-like of TimeSeries objects, optional
        The intersection of these TimeSeries' indexes will be used to build the final index.
        If the 'union' argument is not None, the final index will be the taken from the intersection of the
        intersection of indexes in the 'intersection' argument with the union of the indexes in the 'union' argument.
    dates: list-like of datetime.datetime, optional
        Starting index.
    last: bool, optional
        If True, select only the last date after all union and intersection operations in the indexes.
    initial_date: datetime.datetime, optional
        If passed, drop dates in the index that are lower than it.
    final_date: datetime.datetime, optional
        If passed, drop dates in the index that are higher than it.
    collapse_intraday: bool, optional
        Drop intraday dates.
    collapse_intraday_initial_date: datetime.datetime, optional
        Drop intraday after this date.
    collapse_intraday_final_date: datetime.datetime, optional
        Drop intraday before this date.

    Returns
    -------
    pandas.DatetimeIndex
        The final (common) index obtained.

    """
    ts_list = list()
    if union is not None:
        ts_list += [ts for ts in union]
    if intersection is not None:
        ts_list += [ts for ts in intersection]

    if not ts_list:
        raise ValueError("rebase_ts_values should receive iterables of TimeSeries in either its 'union' or "
                         "'intersection' arguments.")

    # Remove intraday data if necessary, but not the last date's!
    if collapse_intraday:
        collapse_intraday_ts_values(ts_list, initial_date=collapse_intraday_final_date,
                                    final_date=collapse_intraday_final_date)

    # Reindex on the union of all the dates.
    if dates is None:
        dates = set()
        if union is not None:
            dates = dates.union(index_union([ts.ts_values for ts in union]))
        if intersection is not None:
            if union is None:
                dates = intersection[0].ts_values.index
            for ts in intersection:
                dates = dates.intersection(ts.ts_values.index)
        dates = pd.DatetimeIndex(sorted(dates))
    for ts in ts_list:
        ts.ts_values = ts.ts_values.reindex(index=dates, method='pad')
    if last:
        last_date = max(ts_list[0].ts_values.index)
        for ts in ts_list:
            filter_series(ts.ts_values, initial_date=last_date)
    if initial_date is not None or final_date is not None:
        for ts in ts_list:
            filter_series(ts.ts_values, initial_date=initial_date, final_date=final_date)
    if collapse_intraday:
        collapse_intraday_ts_values(ts_list, initial_date=collapse_intraday_initial_date,
                                    final_date=collapse_intraday_final_date)

    # Return the final index.
    return ts_list[0].ts_values.index


def index_union(df_list):
    """Union of TimeSeries' ts_values indexes.

    Parameters
    ----------
    df_list: list-like of TimeSeries objects.

    Returns
    -------
    pandas.DatetimeIndex

    """
    all_dates = set()
    for df in df_list:
        all_dates = all_dates.union(df.index)
    all_dates = sorted(all_dates)
    all_dates = pd.DatetimeIndex(all_dates)
    return all_dates


def filter_series(df, initial_date=None, final_date=None):
    """Filter (inplace) a Series/DataFrame DatetimeIndex to keep only dates that are between two dates.

    Parameters
    ----------
    df: pandas.Series, pandas.DataFrame
    initial_date: datetime.datetime, optional
    final_date: datetime.datetime, optional

    """

    if initial_date is None and final_date is not None:
        final_date = pd.to_datetime(final_date)
        df.drop(df[(df.index > final_date)].index, inplace=True)
    elif final_date is None and initial_date is not None:
        initial_date = pd.to_datetime(initial_date)
        df.drop(df[(df.index < initial_date)].index, inplace=True)
    elif final_date is None and initial_date is None:
        pass
    elif initial_date == final_date:
        initial_date = pd.to_datetime(initial_date)
        df.drop(df[(df.index != initial_date)].index, inplace=True)
    else:
        initial_date = pd.to_datetime(initial_date)
        final_date = pd.to_datetime(final_date)
        df.drop(df[(df.index < initial_date) | (df.index > final_date)].index, inplace=True)


def to_list(arg):
    """Make the object iterable in a 'reasonable way'.

    'reasonable' means:
        If the object is a string, tuple or dict, return [object].
        If the object has no __iter__ method, return [object].
        If the object has an __iter__ method, return object (unchanged).

    Parameters
    ----------
    arg

    Returns
    -------
    iterable object

    """
    if hasattr(arg, '__iter__') and not isinstance(arg, str) and not isinstance(arg, tuple) and not isinstance(arg,
                                                                                                               dict):
        return arg
    else:
        return [arg]


def to_upper_list(arg):
    """Convert a string or list of strings in upper-case list of strings.

    Parameters
    ----------
    arg: str or list-like of str.

    Returns
    -------
    list

    """
    if isinstance(arg, str):
        return [arg.upper()]
    else:
        return [x.upper() for x in arg]


def to_lower_list(arg):
    """Convert a string or list of strings in lower-case list of strings.

    Parameters
    ----------
    arg: str or list-like of str.

    Returns
    -------
    list

    """
    if isinstance(arg, str):
        return [arg.lower()]
    else:
        return [x.lower() for x in arg]


def isvectorizable(obj):
    """Check if object is numpy.vectorize-vectorizable.

    Parameters
    ----------
    obj

    Returns
    -------
    bool

    """
    return hasattr(obj, '__iter__') and not isinstance(obj, (str, datetime, np.datetime64, ql.Date))


def conditional_vectorize(*args):
    """Create a function generator that numpy.vectorizes a given function depending of the passed arguments.

    This function is intended to be used as a decorator/wrapper for class methods or functions that may accept single
    or iterable types of input for the same argument.

    Parameters
    ----------
    args: strings
        The argument names of the input function ('f', that will be passed to the 'function_builder') that signal
        vectorization of the input function. If one of the arguments in 'args' given to 'f' are iterable,
        'function_builder' will return a vectorized 'f'. Otherwise, 'function_builder' will return 'f' unchanged.

    Returns
    -------
    function

    Examples
    --------
    Suppose you have a method or function that carries out a complicated calculation.

    # >>> def calculation(argument1, argument2):
    ...     return argument1*argument2  # Just an example!
    ... print(calculation(argument=1, argument2=2))
    2

    Suppose you now want the calculation function to accept iterable arguments and run the calculation in every
    argument within the iterable automatically, without polluting the function's implementation.

    # >>> @conditional_vectorize('argument1', 'argument2')
    ... def calculation(argument1, argument2)
    ...     return argument1*argument2
    ...
    ... print(calculation(argument1=1, argument2=2))
    2
    ...
    ... print(calculation(argument1=[1, 2], argument2=2))
    [2, 4]
    ...
    ... print(calculation(argument1=[1, 2], argument2=[3, 4]))
    [3, 8]
    ...
    ... print(calculation(argument1=2, argument2=[4, 5]))
    [8, 10]

    Warnings
    --------
    Every function decorated by conditional_vectorize must ALWAYS be called with keyword arguments. Failure to do so
    may cause unexpected behavior.

    """
    def function_builder(f):
        @wraps(f)
        def new_f(*fargs, **kwargs):
            if any(isvectorizable(kwargs.get(arg, None)) for arg in args):
                excluded = [x for x in kwargs.keys() if x not in args] + list(fargs)
                try:
                    return ExtendedArray(np.vectorize(f, excluded=excluded)(*fargs, **kwargs), meta=kwargs)
                except ValueError as e:
                    for arg in args:
                        arg_value = kwargs.get(arg, None)
                        if arg_value is not None:
                            if len(arg_value) == 0:
                                return np.nan
                    raise e
            else:
                return f(*fargs, **kwargs)
        new_f._conditional_vectorized_ = True
        new_f._conditional_vectorize_args_ = args
        return new_f
    return function_builder


def timeit(loops=10):
    """Time function execution.

    Decorate a function/method in test-code with this to obtain an estimate of the function's run-time. The execution
    time is printed after some executions (given by 'loops' argument).

    Parameters
    ----------
    loops: int, optional

    """
    # Loops is the number of times to execute the passed function
    if loops <= 0:
        loops = 1

    def build_timed_method(method):
        def timed_method(*args, **kwargs):
            times = list()
            for i in range(loops):
                # print("Running iteration {}".format(i))
                start_time = time.time()
                result = method(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            print("timeit: {0} {1:.9}ms, avg of {2} iterations.".format(method.__name__, np.mean(times)*1000, loops))
            return result
        return timed_method
    return build_timed_method


def find_le(a, x):
    """ Find rightmost value in `a` that is less than or equal `x`.

    Parameters
    ----------
    a: list-like
        Sorted list.
    x: object
        object to compare against elements of `a`.

    Returns
    -------
    object
        Leftmost value in `a` that is less than or equal `x`.
    """
    i = bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError


def find_gt(a, x):
    """ Find leftmost value in `a` that is greater than `x`.

    Parameters
    ----------
    a: list-like
        Sorted list.
    x: object
        object to compare against elements of `a`.

    Returns
    -------
    object
        Leftmost value in a greater than x.
    """
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError


class ExtendedArray(np.ndarray):
    def __new__(cls, input_array, meta=None):
        # Input array is an already formed ndarray instance.
        # We first cast to be our class type.
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.meta = meta
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.meta = getattr(obj, 'meta', None)

