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
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.base import Instrument, to_datetime, to_ql_date, to_ql_frequency, to_ql_weekday, to_ql_option_engine, \
    to_ql_equity_model, to_ql_swaption_engine, to_ql_short_rate_model
from tsfin.instruments.interest_rates import DepositRate, ZeroRate, OISRate, SwapRate, Swaption, CDSRate, \
    EurodollarFuture
from tsfin.instruments.equities import Equity, EquityOption
from tsfin.instruments.bonds import FixedRateBond, CallableFixedRateBond, FloatingRateBond, ContingentConvertibleBond
from tsfin.instruments import CurrencyFuture, Currency
from tsfin.stochasticprocess.equityprocess import BlackScholesMerton, BlackScholes, Heston, GJRGARCH
from tsfin.constants import TYPE, BOND, BOND_TYPE, FIXEDRATE, CALLABLEFIXEDRATE, FLOATINGRATE, INDEX, DEPOSIT_RATE, \
    DEPOSIT_RATE_FUTURE, CURRENCY_FUTURE, SWAP_RATE, OIS_RATE, EQUITY_OPTION, FUND, EQUITY, CDS, \
    INDEX_TIME_SERIES, ZERO_RATE, SWAP_VOL, CDX, EURODOLLAR_FUTURE, CONTINGENTCONVERTIBLE, EXCHANGE_TRADED_FUND,\
    INSTRUMENT, BLACK_SCHOLES_MERTON, BLACK_SCHOLES, HESTON, GJR_GARCH, CURRENCY


def generate_instruments(ts_collection, indexes=None, index_curves=None):
    """ Given a collection of :py:obj:`TimeSeries`, instantiate instruments with each one of them.

    If an element is not an instance of :py:class:`TimeSeries`, does nothing with it.

    :param ts_collection: :py:obj:`TimeSeriesCollection`
        Collection of time series.
    :param indexes: dict, optional
        Dictionary with ``{index_name: index_time_series}``, where `index_name` is the name of the 'index' (e.g.:
        libor3m, CDI), in the INDEX attribute of a time series. `index_time_series` is a :py:obj:`TimeSeries`
        representing the index. This is currently needed only for floating rate bonds. Default is None.
    :param index_curves: dict, optional
        Dictionary with ``{index_name: index_yield_curve_time_series}``, where `index_name` is the name of the 'index'
        (e.g.: libor3m, CDI), in the INDEX attribute of a time series. `index_yield_curve_time_series` is a
        :py:obj:`YieldCurveTimeSeries` representing the yield curve of the index. This is currently
        needed only for floating rate bonds. Default is None.

    :return: :py:obj:`TimeSeriesCollection`
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
        elif ts_type in [EQUITY, EXCHANGE_TRADED_FUND]:
            instrument = Equity(ts)
        elif ts_type in [FUND, INSTRUMENT]:
            instrument = Instrument(ts)
        elif ts_type == CURRENCY:
            instrument = Currency(ts)
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

    :param args: time series names (each time series represents a parameter).

    :return: dict
        Dictionary of parameter tuples, indexed by dates.
    """
    params_df = pd.concat([getattr(ts, 'ts_values') for ts in args], axis=1)
    params = dict()
    for data in params_df.itertuples():
        params[to_ql_date(data[0])] = tuple(data[i] for i in range(1, len(data)))
    return params


def ts_to_dict(*args):
    """ Produce a date-indexed dictionary of tuples from the values of multiple time series.

    :param args: py:obj:TimeSeries.ts_values
        (each time series represents a parameter).
    :return dict
        Dictionary of parameter tuples, indexed by dates.
    """
    params_df = pd.concat([ts for ts in args], axis=1)
    params = dict()
    for data in params_df.itertuples():
        params[to_ql_date(data[0])] = tuple(data[i] for i in range(1, len(data)))
    return params


def filter_series(df, initial_date=None, final_date=None):
    """ Filter inplace a pandas.Series to keep its index between an initial and a final date.

    :param df: :py:obj:`pandas.Series`
        Series to be filtered.
    :param initial_date: date-like
        Initial date.
    :param final_date: date-like
        Final date.
    :return
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

    :param ts: :py:obj:`TimeSeries`, :py:obj:`pandas.Series`, :py:obj:`pandas.DataFrame`
        Time series whose returns will be calculated.
    :param calc_type: {'D', 'W', 'M', '6M', 'Y', '3Y', 'WTD', 'MTD', 'YTD', 'SI'}, optional
        The time window for return calculation. Default is 'D' (daily returns).
    :param force: bool, optional
        Backward-fill missing data. Default is False.
    :return :py:obj:`pandas.Series`, :py:obj:`pandas.DataFrame`
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


def calibrate_swaption_model(date, model_name, term_structure_ts, swaption_vol_ts_collection, solver_name=None,
                             use_scipy=False, fix_mean=False, mean_reversion_value=0.03, **kwargs):

    """ Calibrate a QuantLib Swaption model.

    :param date: QuantLib.Date
        Calibration date.
    :param model_name: str
        The model name for calibration
    :param term_structure_ts: :py:obj:`YieldCurveTimeSeries`
        Yield curve time series of the curve.
    :param swaption_vol_ts_collection: :py:obj:`TimeSeriesCollection`
        Collection of swaption volatility (Black, log-normal) quotes.
    :param solver_name: str
        The solver to be used, see :py:func:calibrate_ql_model for options
    :param use_scipy: bool
        Whether to use the QuantLib solvers or Scipy solvers, see :py:func:calibrate_ql_model for options
    :param fix_mean: bool
        If you want to fix or not the mean reversion of the model
    :param mean_reversion_value: float
        Mean reversion value, used when the mean reversion is fixed.
    :return: QuantLib.CalibratedModel
        Calibrated model.
    """
    date = to_ql_date(date)
    model_name = str(model_name).upper()
    term_structure = term_structure_ts.yield_curve_handle(date=date)

    ql.Settings.instance().evaluationDate = date
    if fix_mean:
        model = to_ql_short_rate_model(model_name=model_name)(term_structure, mean_reversion_value)
    else:
        model = to_ql_short_rate_model(model_name=model_name)(term_structure)
    engine = to_ql_swaption_engine(model_name=model_name, model=model)

    swaption_helpers = list()
    swaption_vol = generate_instruments(swaption_vol_ts_collection)
    for swaption in swaption_vol:
        swaption.link_to_term_structure(date=date, yield_curve=term_structure_ts)
        helper = swaption.rate_helper(date=date)
        helper.setPricingEngine(engine)
        swaption_helpers.append(helper)

    return calibrate_ql_model(date=date, model_name=model_name, model=model, helpers=swaption_helpers,
                              solver_name=solver_name, use_scipy=use_scipy, max_iteration=10000,
                              max_stationary_state_iteration=100, **kwargs)


def get_base_equity_process(process_name):
    """ Return the :py:class:BaseEquityProcess for different equity models.

    (GJR GARCH is experimental)
    :param process_name: str
        The equity process name: BLACK_SCHOLES_MERTON, BLACK_SCHOLES, HESTON, GJR_GARCH
    :return: :py:class:BaseEquityProcess
    """
    if process_name.upper() == BLACK_SCHOLES_MERTON:
        return BlackScholesMerton
    elif process_name.upper() == BLACK_SCHOLES:
        return BlackScholes
    elif process_name.upper() == HESTON:
        return Heston
    elif process_name.upper() == GJR_GARCH:
        return GJRGARCH


def get_equity_option_model_and_helpers(date, term_structure_ts, spot_price, dividend_yield, dividend_tax,
                                        option_collection, engine_name, model_name, process_name,
                                        implied_vol_process='BLACK_SCHOLES_MERTON', error_type=None,
                                        exercise_type='EUROPEAN', **kwargs):
    """ Returns the equity model and calibration helper from a given Equity Process

    :param date: Date-like
        The reference date of the calibration
    :param term_structure_ts: :py:class:YieldCurveTimeSeries
        The yield curve used in the calibration
    :param spot_price: float
        The reference spot price used for calibration
    :param dividend_yield: float
        The dividend yield of the underlying stock process
    :param dividend_tax: float
        The dividend tax
    :param option_collection: :py:object:TimeSeriesCollection
        The option TimeSeries used for calibration
    :param engine_name: str
        The engine name representing a QuantLib.PricingEngine
    :param model_name: str
        The mode name representing a QuantLib.CalibratedModel
    :param process_name: str
        The process name representing a :py:class:BaseEquityProcess
    :param implied_vol_process: str
        The process name used for calculating the initial implied vol
    :param error_type: str
        The name representing the QuantLib.BlackCalibrationHelper used for minimizing the function
    :param exercise_type: str
        The option exercise type: AMERICAN or EUROPEAN
    :param kwargs:
        User for passing the constant values to their underlying process
    :return: QuantLib.CalibratedModel, QuantLib.HestonModelHelper
    """

    date = to_ql_date(date)
    ql.Settings.instance().evaluationDate = date

    options = generate_instruments(option_collection)
    calendar = options[0].calendar
    day_counter = options[0].day_counter

    process = get_base_equity_process(process_name=process_name)
    process = process(calendar=calendar, day_counter=day_counter)
    process.risk_free_handle.linkTo(term_structure_ts.yield_curve(date=date))
    process.dividend_yield.setValue(float(dividend_yield * (1 - dividend_tax)))
    process.spot_price.setValue(spot_price)

    ql_model = to_ql_equity_model(model_name=model_name)
    model = ql_model(process.process(**kwargs))
    engine = to_ql_option_engine(engine_name=engine_name, model=model)

    heston_helpers = list()
    for option in options:
        option.set_yield_curve(risk_free_yield_curve_ts=term_structure_ts)
        option.set_ql_process(base_equity_process=get_base_equity_process(process_name=implied_vol_process))
        option.change_exercise_type(exercise_type=exercise_type)
        volatility = option.implied_volatility(date=date, base_date=date, spot_price=spot_price,
                                               dividend_yield=dividend_yield, dividend_tax=dividend_tax)
        heston_helper = option.heston_helper(date=date, volatility=volatility, base_equity_process=process,
                                             error_type=error_type)
        heston_helper.setPricingEngine(engine)
        heston_helpers.append(heston_helper)

    return model, heston_helpers


def cost_function_generator(model, helpers, norm=False):
    """ Creates a cost function to be used in by scipy solvers.

    :param model: QuantLib.CalibratedModel
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


def calibrate_ql_model(date, model_name, model, helpers, initial_conditions=None, use_scipy=False, solver_name=None,
                       bounds=None, max_iteration=1000, max_stationary_state_iteration=200, ql_constraint=None,
                       ql_weights=None, fix_parameters=None):

    """ Returns the QuantLib model calibrated.

    :param date:  Date-like
        The reference date of the calibration
    :param model_name: str
        The model name
    :param model: QuantLib.CalibratedModel
        The QuantLib model to be calibrated
    :param helpers: QuantLib.BlackCalibrationHelper
        The QuantLib helpers used in the calibration
    :param initial_conditions: list
        For some scipy optimizers it's necessary to pass the initial values for the roots.
    :param use_scipy: bool
         Whether to use the QuantLib solvers or Scipy solvers
    :param solver_name: str
        The name of the optimization function
    :param bounds: list
        For some scipy optimizers it's necessary to pass the bounds for the roots
    :param max_iteration: int
        For some optimizers it defines the max iterations
    :param max_stationary_state_iteration: int
        For some optimizers it defines the max iterations in stationary state
    :param ql_constraint: QuantLib.Constraint
        The QuantLib object used for defining constraints for the models, only works with QuantLib optimizers.
    :param ql_weights: list
        List of weights to be applied to the model parameters, only works with QuantLib optimizers.
    :param fix_parameters: list of bool
        A list of booleans indicating if the parameter should be fixed or not, has to be the same length as the number
        of parameters in the model. True for fixed parameter, False otherwise
    :return: QuantLib.CalibratedModel
        Returns the given QuantLib model calibrated.
    """

    date = to_ql_date(date)
    print('Calibrating {0} model for {1}'.format(model_name, date))
    ql.Settings.instance().evaluationDate = date
    solver_name = str(solver_name).upper()

    if use_scipy:
        if solver_name == 'LEVENBERG_MARQUARDT':
            from scipy.optimize import root
            if initial_conditions is None:
                raise print("Please specify the parameters initial values")
            initial_conditions = np.array(initial_conditions)
            cost_function = cost_function_generator(model, helpers)
            sol = root(cost_function, initial_conditions, method='lm')
        elif solver_name == 'LEAST_SQUARES':
            from scipy.optimize import least_squares
            if initial_conditions is None:
                raise print("Please specify the parameters initial values")
            initial_conditions = np.array(initial_conditions)
            cost_function = cost_function_generator(model, helpers)
            if bounds is None:
                bounds = (-np.inf, np.inf)
            sol = least_squares(cost_function, initial_conditions, bounds=bounds)
        elif solver_name == 'DIFFERENTIAL_EVOLUTION':
            from scipy.optimize import differential_evolution
            if bounds is None:
                raise print("Please specify the parameters bounds")
            cost_function = cost_function_generator(model, helpers, norm=True)
            sol = differential_evolution(cost_function, bounds, maxiter=max_iteration)
        elif solver_name == 'BASIN_HOPPING':
            from scipy.optimize import basinhopping
            if initial_conditions is None:
                raise print("Please specify the parameters initial values")
            initial_conditions = np.array(initial_conditions)
            if bounds is None:
                raise print("Please specify the parameters bounds")
            min_list, max_list = zip(*bounds)
            my_bound = MyBounds(xmin=list(min_list), xmax=list(max_list))
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            cost_function = cost_function_generator(model, helpers, norm=True)
            sol = basinhopping(cost_function, initial_conditions, niter=25, minimizer_kwargs=minimizer_kwargs,
                               stepsize=0.005, accept_test=my_bound, interval=5)
    else:
        end_criteria = ql.EndCriteria(maxIteration=max_iteration,
                                      maxStationaryStateIterations=max_stationary_state_iteration,
                                      rootEpsilon=1.0e-8,
                                      functionEpsilon=1.0e-8,
                                      gradientNormEpsilon=1.0e-8)
        if solver_name == 'LEVENBERG_MARQUARDT':
            optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
        elif solver_name == 'SIMPLEX':
            optimization_method = ql.Simplex(0.025)
        else:
            optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
        if ql_constraint is None:
            ql_constraint = ql.NoConstraint()
        if ql_weights is None:
            ql_weights = []
        if fix_parameters is None:
            n_params = len(model.params())
            fix_parameters = list()
            for i in range(n_params):
                fix_parameters.append(False)
        model.calibrate(helpers, optimization_method, end_criteria, ql_constraint, ql_weights, fix_parameters)
    return model


# noinspection PyDefaultArgument
class MyBounds(object):
    """
    Class for defining the bounds to be used in the Basin Hopping optimizer.
    """
    def __init__(self, xmin=[0., 0.01, 0.01, -1, 0], xmax=[1, 15, 1, 1, 1.0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def to_np_array(ql_matrix):
    """ Transform a QuantLib Matrix/Array into a Numpy array

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


def str_to_bool(arg):
    """Function to convert String True or False to bool

    :param arg: str
    :return: bool
    """
    arg = str(arg).upper()
    if arg == 'TRUE':
        return True
    elif arg == 'FALSE':
        return False


def cash_flows_from_df(df):
    """return the cash flows for IRR calculation

    :param df: pandas.DataFrame
        DataFrame with the cash flows
    :return: list, int, date
    """
    first_amount = df.first('1D').values
    first_date = df.first('1D').index[0]

    df = df.sort_index()
    cash_flow = list()

    if first_amount > 0:
        df *= -1
    first_amount = np.abs(first_amount)

    for date, value in df.iteritems():
        cash_flow.append(ql.SimpleCashFlow(float(value), to_ql_date(date)))

    return cash_flow, first_amount, first_date


def ql_irr(cash_flow, first_amount, first_date):
    """ Calculate the IRR from a given cash flow

    :param cash_flow: list
        List of QuantLib.SimpleCashFlow (s)
    :param first_amount: Int
        The Amount of the first cash flow
    :param first_date: Date-Like
        The date of the first cash flow
    :return: float
    """

    ql.Settings.instance().evaluationDate = to_ql_date(first_date)
    try:
        fixed_rate = ql.CashFlows.yieldRate(cash_flow, float(first_amount), ql.Actual365Fixed(), ql.Compounded,
                                            ql.Annual, False, ql.Date(), ql.Date(), 1.0e-6, 1000, 0.05)
    except RuntimeError:
        try:
            fixed_rate = ql.CashFlows.yieldRate(cash_flow, float(first_amount), ql.Actual365Fixed(), ql.Compounded,
                                                ql.Annual, False, ql.Date(), ql.Date(), 1.0e-6, 1000, -0.1)
        except RuntimeError:
            fixed_rate = ql.CashFlows.yieldRate(cash_flow, float(first_amount), ql.Actual365Fixed(), ql.Compounded,
                                                ql.Annual, False, ql.Date(), ql.Date(), 1.0e-6, 1000, -0.5)
    return fixed_rate


def ql_irr_from_df(df):
    """ Consolidated function to calculate the IRR from a Pandas.DataFrame

    :param df: Pandas.DataFrame
    :return: float
        The IRR
    """
    cash_flow, first_amount, first_date = cash_flows_from_df(df)
    rate = ql_irr(cash_flow=cash_flow, first_amount=first_amount, first_date=first_date)
    return rate


def nth_weekday_of_month(start_year, n_years, frequency, weekday, nth_day, min_date=None, max_date=None):
    """ Function to get a list of dates following a specific weekday at a specific recurrence inside a month

     example: the 3rd friday of the month, every 3 months.
    :param start_year: int
        The base year for the date calculation
    :param n_years:
        The amount of years ahead to forecast
    :param frequency: str
        The frequency of the monthly occurrence, ex: ANNUAL, SEMIANNUAL, EVERY_FOUR_MONTH, QUARTERLY,
         BIMONTHLY, MONTHLY"
    :param weekday: str
        The weekday of the recurrence, Monday, Tuesday...
    :param nth_day: int
        The nth occurrence inside of the month.
    :param min_date: Date-like, optional
        The minimum date of the list
    :param max_date: Date-like, optional
        The minimum date of the list
    :return: list of QuantLib.Dates
    """
    frequency = str(frequency).upper()
    ql_frequency = to_ql_frequency(frequency)
    if not 0 < ql_frequency <= 12:
        raise ValueError("Only supported frequencies are: ANNUAL, SEMIANNUAL, EVERY_FOUR_MONTH,"
                         " QUARTERLY, BIMONTHLY, MONTHLY")

    weekday = str(weekday).upper()
    ql_weekday = to_ql_weekday(weekday)
    nth_day = int(nth_day)

    dates = list()
    for j in range(n_years + 1):
        year = start_year + j
        for i in range(ql_frequency):
            month = int((i + 1) * (12 / ql_frequency))
            dates.append(ql.Date.nthWeekday(nth_day, ql_weekday, month, year))

    if min_date is not None and max_date is None:
        min_date = to_ql_date(min_date)
        dates = [date for date in dates if date >= min_date]
    elif min_date is None and max_date is not None:
        max_date = to_ql_date(max_date)
        dates = [date for date in dates if date <= max_date]
    elif min_date is not None and max_date is not None:
        min_date = to_ql_date(min_date)
        max_date = to_ql_date(max_date)
        dates = [date for date in dates if min_date <= date <= max_date]

    return dates
