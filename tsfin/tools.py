import pandas as pd
import QuantLib as ql
from pandas.tseries.offsets import BDay, Week, BMonthEnd, BYearEnd
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import to_datetime
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.base.instrument import Instrument
from tsfin.instruments.bonds import FixedRateBond, CallableFixedRateBond, FloatingRateBond
from tsfin.instruments.depositrate import DepositRate
from tsfin.instruments.ois import OISRate
from tsfin.instruments.currencyfuture import CurrencyFuture
from tsfin.instruments.swaprate import SwapRate
from tsfin.instruments.equityoptions import BaseEquityOption
from tsfin.constants import TYPE, BOND, BOND_TYPE, FIXEDRATE, CALLABLEFIXEDRATE, FLOATINGRATE, INDEX, DEPOSIT_RATE, \
    DEPOSIT_RATE_FUTURE, CURRENCY_FUTURE, SWAP_RATE, OIS_RATE, EQUITY_OPTION, RATE_INDEX, FUND, EQUITY


def generate_instruments(ts_collection, ql_process=None, indices=None, index_curves=None):
    """ Given a collection of :py:obj:`TimeSeries`, instantiate instruments with each one of them.

    If an element is not an instance of :py:class:`TimeSeries`, does nothing with it.

    Parameters
    ----------
    ts_collection: :py:obj:`TimeSeriesCollection`
        Collection of time series.
    ql_process: :py:obj: 'BlackScholesMerton'
        Class which creates the QuantLib process needed to calculate the Option.
    indices: dict, optional
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
            if bond_type == FLOATINGRATE:
                # Floating rate bonds need some special treatment.
                index_tag = ts.get_attribute(INDEX)
                instrument = FloatingRateBond(ts, reference_curve=index_tag)
            elif bond_type == FIXEDRATE:
                instrument = FixedRateBond(ts)
            elif bond_type == CALLABLEFIXEDRATE:
                instrument = CallableFixedRateBond(ts)
            else:
                instrument_list.append(ts)
                continue

        elif ts_type in (DEPOSIT_RATE, DEPOSIT_RATE_FUTURE, RATE_INDEX):
            instrument = DepositRate(ts)
        elif ts_type == CURRENCY_FUTURE:
            instrument = CurrencyFuture(ts)
        elif ts_type == SWAP_RATE:
            instrument = SwapRate(ts)
        elif ts_type == OIS_RATE:
            instrument = OISRate(ts)
        elif ts_type == EQUITY_OPTION:
            instrument = BaseEquityOption(ts, ql_process=ql_process)
        elif ts_type == EQUITY:
            instrument = Instrument(ts)
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


def filter_series(df, initial_date=None, final_date=None):
    """ Filter inplace a pandas.Series to keep its index between an initial and a final date.

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


def calibrate_hull_white_model(date, model_class, term_structure_ts, swaption_vol_ts_collection):
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
    print("Calibrating Hull-White 1F short rate model for date = {}".format(date))
    term_structure = term_structure_ts.yield_curve_handle(date=date)
    # TODO: Make all the below depend on parameters of the swaption_vol_ts_collection
    index = ql.USDLibor(ql.Period(3, ql.Months), term_structure)
    fixed_leg_tenor = ql.Period(3, ql.Months)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    ql.Settings.instance().evaluationDate = date
    model = model_class(term_structure)
    engine = ql.JamshidianSwaptionEngine(model)

    # Create the swaption helpers
    swaption_helpers = list()
    for swaption in swaption_vol_ts_collection:
        vol = swaption.get_values(index=to_datetime(date)) / 100
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(vol))
        helper = ql.SwaptionHelper(calendar.advance(date, ql.PeriodParser.parse(swaption.get_attribute('MATURITY'))),
                                   ql.PeriodParser.parse(swaption.get_attribute("TENOR")), vol_handle, index,
                                   fixed_leg_tenor, fixed_leg_daycounter, floating_leg_daycounter, term_structure)
        helper.setPricingEngine(engine)
        swaption_helpers.append(helper)

    optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
    end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
    model.calibrate(swaption_helpers, optimization_method, end_criteria)
    return model
