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

import pandas as pd
import QuantLib as ql
import numpy as np
from pandas.tseries.offsets import BDay, Week, BMonthEnd, BYearEnd
from scipy.optimize import root
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import to_datetime
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.base.instrument import Instrument
from tsfin.instruments import FixedRateBond, CallableFixedRateBond, FloatingRateBond, ContingentConvertibleBond, \
    DepositRate, ZeroRate, OISRate, CurrencyFuture, SwapRate, Swaption, EquityOption, CDSRate, EurodollarFuture, \
    Equity
from tsfin.constants import TYPE, BOND, BOND_TYPE, FIXEDRATE, CALLABLEFIXEDRATE, FLOATINGRATE, INDEX, DEPOSIT_RATE, \
    DEPOSIT_RATE_FUTURE, CURRENCY_FUTURE, SWAP_RATE, OIS_RATE, EQUITY_OPTION, FUND, EQUITY, CDS, \
    INDEX_TIME_SERIES, ZERO_RATE, SWAP_VOL, CDX, EURODOLLAR_FUTURE, FUND_TYPE, ETF, CONTINGENTCONVERTIBLE


def generate_instruments(ts_collection, indexes=None, index_curves=None):
    """ Given a collection of :py:obj:`TimeSeries`, instantiate instruments with each one of them.

    If an element is not an instance of :py:class:`TimeSeries`, does nothing with it.

    Parameters
    ----------
    ts_collection: :py:obj:`TimeSeriesCollection`
        Collection of time series.
    indexes: dict, optional
        Dictionary with ``{index_name: index_time_series}``, where `index_name` is the name of the 'index' (e.g.:
        libor3m, CDI), in the INDEX attribute of a time series. `index_time_series` is a :py:obj:`TimeSeries`
        representing the index. This is currently needed only for floating rate bonds. Default is None.
    index_curves: dict, optional
        Dictionary with ``{index_name: index_yield_curve_time_series}``, where `index_name` is the name of the 'index'
        (e.g.: libor3m, CDI), in the INDEX attribute of a time series. `index_yield_curve_time_series` is a
        :py:obj:`YieldCurveTimeSeries` representing the yield curve of the index. This is currently
        needed only for floating rate bonds. Default is None.

    Returns
    -------
    :py:obj:`TimeSeriesCollection`
        Time series collection with the created instruments.
    """
    instrument_list = list()

    for ts in ts_collection:
        if not isinstance(ts, TimeSeries):
            # Then ts must be an instance of its object already. Add it to instrument list and skip.
            instrument_list.append(ts)
            continue

        ts_type = ts.get_attribute(TYPE)

        if ts_type == BOND:
            bond_type = str(ts.get_attribute(BOND_TYPE)).upper()
            if bond_type in [FLOATINGRATE, CONTINGENTCONVERTIBLE]:
                # Floating rate bonds need some special treatment.
                reference_curve = index_curves[str(ts.get_attribute(INDEX)).upper()] if index_curves is not None \
                        else None
                index_timeseries = indexes[str(ts.get_attribute(INDEX_TIME_SERIES)).upper()] if indexes is not None \
                    else None
                if bond_type == FLOATINGRATE:
                    instrument = FloatingRateBond(ts, reference_curve=reference_curve,
                                                  index_timeseries=index_timeseries)
                elif bond_type == CONTINGENTCONVERTIBLE:
                    instrument = ContingentConvertibleBond(ts, reference_curve=reference_curve,
                                                           index_timeseries=index_timeseries)
            elif bond_type == FIXEDRATE:
                instrument = FixedRateBond(ts)
            elif bond_type == CALLABLEFIXEDRATE:
                instrument = CallableFixedRateBond(ts)
            else:
                instrument_list.append(ts)
                continue

        elif ts_type in (DEPOSIT_RATE, DEPOSIT_RATE_FUTURE):
            instrument = DepositRate(ts)
        elif ts_type == ZERO_RATE:
            instrument = ZeroRate(ts)
        elif ts_type == CURRENCY_FUTURE:
            instrument = CurrencyFuture(ts)
        elif ts_type == SWAP_RATE:
            instrument = SwapRate(ts)
        elif ts_type == SWAP_VOL:
            instrument = Swaption(ts)
        elif ts_type == OIS_RATE:
            instrument = OISRate(ts)
        elif ts_type == EQUITY_OPTION:
            instrument = EquityOption(ts)
        elif ts_type == EQUITY:
            instrument = Equity(ts)
        elif ts_type == FUND:
            fund_type = str(ts.get_attribute(FUND_TYPE)).upper()
            if fund_type == ETF:
                instrument = Equity(ts)
            else:
                instrument = Instrument(ts)

        elif ts_type in [CDS, CDX]:
            instrument = CDSRate(ts)
        elif ts_type == EURODOLLAR_FUTURE:
            instrument = EurodollarFuture(ts)
        else:
            instrument = TimeSeries(ts)

        instrument_list.append(instrument)

    return TimeSeriesCollection(instrument_list)


def ts_values_to_dict(*args):
    """ Produce a date-indexed dictionary of tuples from the values of multiple time series.

    Parameters
    ----------
    args: time series names (each time series represents a parameter).

    Returns
    -------
    dict
        Dictionary of parameter tuples, indexed by dates.
    """
    params_df = pd.concat([getattr(ts, 'ts_values') for ts in args], axis=1)
    params = dict()
    for data in params_df.itertuples():
        params[to_ql_date(data[0])] = tuple(data[i] for i in range(1, len(data)))
    return params


def ts_to_dict(*args):
    """ Produce a date-indexed dictionary of tuples from the values of multiple time series.

    Parameters
    ----------
    args: time series names (each time series represents a parameter).

    Returns
    -------
    dict
        Dictionary of parameter tuples, indexed by dates.
    """
    params_df = pd.concat([ts for ts in args], axis=1)
    params = dict()
    for data in params_df.itertuples():
        params[to_ql_date(data[0])] = tuple(data[i] for i in range(1, len(data)))
    return params


def filter_series(df, initial_date=None, final_date=None):
    """
    Filter inplace a pandas.Series to keep its index between an initial and a final date.

    Parameters
    ----------
    df: :py:obj:`pandas.Series`
        Series to be filtered.
    initial_date: date-like
        Initial date.
    final_date: date-like
        Final date.
    """
    if initial_date is None and final_date is not None:
        final_date = to_datetime(final_date)
        df.drop(df[(df.index > final_date)].index, inplace=True)
    elif final_date is None and initial_date is not None:
        initial_date = to_datetime(initial_date)
        df.drop(df[(df.index < initial_date)].index, inplace=True)
    elif final_date is None and initial_date is None:
        pass
    elif initial_date == final_date:
        initial_date = to_datetime(initial_date)
        df.drop(df[(df.index != initial_date)].index, inplace=True)
    else:
        initial_date = to_datetime(initial_date)
        final_date = to_datetime(final_date)
        df.drop(df[(df.index < initial_date) | (df.index > final_date)].index, inplace=True)


def returns(ts, calc_type='D', force=False):
    """ Calculate returns time series of returns for various time windows.

    Parameters
    ----------
    ts: :py:obj:`TimeSeries`, :py:obj:`pandas.Series`, :py:obj:`pandas.DataFrame`
        Time series whose returns will be calculated.
    calc_type: {'D', 'W', 'M', '6M', 'Y', '3Y', 'WTD', 'MTD', 'YTD', 'SI'}, optional
        The time window for return calculation. Default is 'D' (daily returns).
    force: bool, optional
        Backward-fill missing data. Default is False.

    Returns
    -------
    :py:obj:`pandas.Series`, :py:obj:`pandas.DataFrame`
        Series or DataFrame of returns.
    """
    if isinstance(ts, TimeSeries):
        df = ts.ts_values
    else:
        df = ts
    if df.empty:
        return df

    first_index = df.first_valid_index()
    last_index = df.last_valid_index()

    def array_return(x):
        return x[-1] / x[0] - 1

    def one_month_ago(x):
        return to_datetime(to_ql_date(x) - ql.Period(1, ql.Months))

    def six_months_ago(x):
        return to_datetime(to_ql_date(x) - ql.Period(6, ql.Months))

    calc_type = calc_type.upper()
    if calc_type == 'D':
        return df.pct_change()
    elif calc_type == 'W':
        df = df.reindex()
        return df.resample(BDay()).fillna(method='pad').rolling(6, min_periods=2).apply(array_return)
    elif calc_type == 'M':
        one_month_ago = df.index.map(one_month_ago)
        df_one_month_ago = df.reindex(one_month_ago, method='pad')
        if force:
            df_one_month_ago = df_one_month_ago.fillna(df.loc[first_index])
        return pd.Series(index=df.index, data=df.values/df_one_month_ago.values) - 1
    elif calc_type == '6M':
        six_months_ago = df.index.map(six_months_ago)
        df_six_months_ago = df.reindex(six_months_ago, method='pad')
        if force is True:
            df_six_months_ago = df_six_months_ago.fillna(df.loc[first_index])
        return pd.Series(index=df.index, data=df.values/df_six_months_ago.values) - 1
    elif calc_type == 'Y':
        one_year_ago = df.index - pd.DateOffset(years=1)
        df_one_year_ago = df.reindex(one_year_ago, method='pad')
        if force is True:
            df_one_year_ago = df_one_year_ago.fillna(df.loc[first_index])
        return pd.Series(index=df.index, data=df.values/df_one_year_ago.values) - 1
    elif calc_type == '3Y':
        three_years_ago = df.index - pd.dateOffset(years=3)
        df_three_years_ago = df.reindex(three_years_ago, method='pad')
        if force:
            df_three_years_ago = df_three_years_ago.fillna(df.loc[first_index])
        return pd.Series(index=df.index, data=df.values/df_three_years_ago.values) - 1
    elif calc_type == 'WTD':
        index = pd.date_range(first_index, last_index, freq=Week(weekday=4))
        df_week_end = df.reindex(index, method='pad').reindex(df.index, method='pad')
        return df / df_week_end - 1
    elif calc_type == 'MTD':
        index = pd.date_range(first_index, last_index, freq=BMonthEnd())
        df_month_end = df.reindex(index, method='pad').reindex(df.index, method='pad')
        return df / df_month_end - 1
    elif calc_type == 'YTD':
        index = pd.date_range(first_index, last_index, freq=BYearEnd())
        df_year_end = df.reindex(index, method='pad').reindex(df.index, method='pad')
        return df / df_year_end - 1
    elif calc_type == 'SI':
        return df / df.loc[first_index] - 1


def ql_swaption_engine(model_class, term_structure):

    if model_class == 'BLACK_KARASINSKI':
        model = ql.BlackKarasinski(term_structure)
        engine = ql.TreeSwaptionEngine(model, 100)
        return model, engine
    elif model_class == 'HULL_WHITE':
        model = ql.HullWhite(term_structure)
        engine = ql.JamshidianSwaptionEngine(model)
        return model, engine
    elif model_class == 'G2':
        model = ql.G2(term_structure)
        engine = ql.G2SwaptionEngine(model, 10, 400)
        return model, engine


def calibrate_swaption_model(date, model_class, term_structure_ts, swaption_vol_ts_collection, use_scipy=False):
    """ Calibrate a Hull-White QuantLib model.

    Parameters
    ----------
    date: QuantLib.Date
        Calibration date.
    model_class: QuantLib.Model
        Model for calibration.
    term_structure_ts: :py:obj:`YieldCurveTimeSeries`
        Yield curve time series of the curve.
    swaption_vol_ts_collection: :py:obj:`TimeSeriesCollection`
        Collection of swaption volatility (Black, log-normal) quotes.

    Returns
    -------
    QuantLib.Model
        Calibrated model.
    """
    # This has only been tested for model_class = HullWhite
    date = to_ql_date(date)
    print("Calibrating {0} short rate model for date = {1}".format(model_class, date))
    yield_curve = term_structure_ts.yield_curve(date=date)
    term_structure = ql.YieldTermStructureHandle(yield_curve)

    ql.Settings.instance().evaluationDate = date
    model, engine = ql_swaption_engine(model_class=model_class, term_structure=term_structure)

    swaption_helpers = list()
    swaption_vol = generate_instruments(swaption_vol_ts_collection)
    for swaption in swaption_vol:
        swaption.set_yield_curve(yield_curve=yield_curve)
        helper = swaption.rate_helper(date=date)
        helper.setPricingEngine(engine)
        swaption_helpers.append(helper)

    if use_scipy:
        initial_condition = np.array(list(model.params()))
        cost_function = cost_function_generator(model=model, helpers=swaption_helpers)
        solution = root(cost_function, initial_condition, method='lm')
    else:
        optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
        end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
        model.calibrate(swaption_helpers, optimization_method, end_criteria)
    return model


def to_np_array(ql_matrix):
    """

    :param ql_matrix: QuantLib.Matrix, QuantLib.Array
    :return: numpy array
    """

    if isinstance(ql_matrix, ql.Array):
        return np.array(ql_matrix, dtype=np.float64)
    elif isinstance(ql_matrix, tuple):
        return np.array(ql_matrix, dtype=np.float64)
    else:
        rows = ql_matrix.rows()
        columns = ql_matrix.columns()
        new_array = np.empty(shape=(rows, columns), dtype=np.float64)
        for n_row in range(ql_matrix.rows()):
            for n_col in range(ql_matrix.columns()):
                new_array[n_row, n_col] = ql_matrix[n_row][n_col]
        return new_array


def generate_discounts_array(paths, grid_dt):
    if paths.ndim == 1:
        paths[0] = 0
        discounts = np.exp(-paths.cumsum() * grid_dt)
        discounts[0] = 1
    else:
        paths[:, 0] = 0
        discounts = np.exp(-paths.cumsum(axis=1)*grid_dt)
        discounts[:, 0] = 1
    return discounts


def cost_function_generator(model, helpers, norm=False):
    """
    function for creating a cost function to be used in by scipy solvers.
    :param model: QuantLib.Model
    :param helpers: QuantLib.CalibrationHelperBase
    :param norm: bool
    :return: cost function
    """
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function


def str_to_bool(arg):
    """
    Function to convert String True or False to boll
    :param arg: str
    :return: bool
    """
    arg = str(arg).upper()
    if arg == 'TRUE':
        return True
    elif arg == 'FALSE':
        return False
