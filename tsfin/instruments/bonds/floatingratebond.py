from functools import wraps
import numpy as np
import QuantLib as ql
from tsfin.base.qlconverters import to_ql_date, to_ql_index
from tsfin.base.basetools import conditional_vectorize
from tsfin.instruments.bonds._basebond import _BaseBond, default_arguments


def set_floating_rate_index(f):
    """ Decorator to set values of the floating rate bond's index.

    This is needed for correct cash flow projection. Does nothing if the bond has no floating coupon.

    Parameters
    ----------
    f: method
        A method that needs the bond to have its past index values set.

    Returns
    -------
    function
        `f` itself. Only the bond instance is modified with this decorator.

    Note
    ----
    If the wrapped function is called with a True-like optional argument 'bypass_set_floating_rate_index',
    the effects of this wrapper is bypassed. This is useful for nested calling of methods that are wrapped by function.
    """
    @wraps(f)
    def new_f(self, **kwargs):
        if kwargs.get('bypass_set_floating_rate_index'):
            return f(self, **kwargs)
        date = to_ql_date(kwargs['date'])
        ql.Settings.instance().evaluationDate = date
        index_timeseries = getattr(self, 'index_timeseries')
        index = getattr(self, 'index')
        forecast_curve = getattr(self, 'forecast_curve')
        reference_curve = getattr(self, 'reference_curve')
        forecast_curve.linkTo(reference_curve.yield_curve(date))
        for dt in getattr(self, 'fixing_dates'):
            if dt <= date:
                index.addFixing(dt, index_timeseries.get_values(index=dt))
        result = f(self, **kwargs)
        ql.IndexManager.instance().clearHistory(index.name())
        return result
    new_f._decorated_by_floating_rate_index_ = True
    return new_f


class FloatingRateBond(_BaseBond):
    """ Floating rate bond.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        The TimeSeries representing the bond.
    index_timeseries: :py:obj:`TimeSeries`
        The TimeSeries containing the quotes of the index used to calculate the bond's coupon.
    reference_curve: :py:obj:YieldCurveTimeSeries
        The yield curve of the index rate, used to estimate future cash flows.

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries, index_timeseries, reference_curve):
        super().__init__(timeseries)
        self.reference_curve = reference_curve
        self.forecast_curve = ql.RelinkableYieldTermStructureHandle()
        self.index = to_ql_index(self.ts_attributes['INDEX'])(self._tenor, self.forecast_curve)
        self.index_timeseries = index_timeseries
        self.gearings = [1]  # TODO: Make it possible to have a list of different gearings.
        self.spreads = [float(self.ts_attributes["SPREAD"])]  # TODO: Make it possible to have different spreads.
        self.caps = []  # TODO: Make it possible to have caps.
        self.floors = []  # TODO: Make it possible to have floors.
        self.in_arrears = False  # TODO: Check if this can be made variable.
        self.bond = ql.FloatingRateBond(self.settlement_days, self.face_amount, self.schedule, self.index,
                                        self.day_counter, self.business_convention, self.index.fixingDays(),
                                        self.gearings, self.spreads, self.caps, self.floors, self.in_arrears,
                                        self.redemption, self.issue_date)
        self.coupon_dates = [cf.date() for cf in self.bond.cashflows()]

        # Store the fixing dates of the coupon. These will be useful later.
        index_calendar = self.index.fixingCalendar()
        index_bus_day_convention = self.index.businessDayConvention()
        reference_schedule = list(self.schedule)[:-1]
        self.fixing_dates = [self.index.fixingDate(dt) for dt in reference_schedule]

        # Coupon pricers
        pricer = ql.BlackIborCouponPricer()
        volatility = 0.0
        vol = ql.ConstantOptionletVolatility(self.settlement_days, index_calendar, index_bus_day_convention,
                                             volatility, self.day_counter)
        pricer.setCapletVolatility(ql.OptionletVolatilityStructureHandle(vol))
        ql.setCouponPricer(self.bond.cashflows(), pricer)
        self.bond_components[self.maturity_date] = self.bond
        self._bond_components_backup = self.bond_components.copy()

    def rate_helper(self, date, last_available=True, yield_type='ytm', curve_type='zero', max_inactive_days=3,
                    reference_date_for_worst_date=None, **kwargs):
        """
        Warnings
        --------
        This method is not implemented for this class yet. It returns None by default.

        Parameters
        ----------
        date: QuantLib.Date
            Date of the rate helper.
        last_available: bool, optional
            Whether to use last available quotes if missing data.
        yield_type: {'ytm', 'ytw'}, optional
            Which yield to use for the rate helper. Default is 'ytm'.
        curve_type: {'zero', 'par'}, optional
            Which type of yield curve will be built. Default is 'zero'.
        max_inactive_days: int, optional
            After how many days of missing data will this method return None. Default is 3.
        reference_date_for_worst_date: QuantLib.Date
            Date of the quote to be used to calculate the worst date of the bond.

        Returns
        -------
        QuantLib.RateHelper
            Object used to build yield curves.
        """
        # TODO: Implement this method.
        return None

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def duration_to_mat(self, duration_type, last, quote, date, day_counter, compounding, frequency, settlement_days,
                        **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Bond's duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """
        date = to_ql_date(date)
        ytm = self.ytm(last=last, quote=quote, date=date, day_counter=day_counter, compounding=compounding,
                       frequency=frequency, settlement_days=settlement_days, bypass_set_floating_rate_index=True,
                       **kwargs)
        settlement_date = self.calendar.advance(date, ql.Period(settlement_days, ql.Days), self.business_convention)
        if self.is_expired(settlement_date):
            return np.nan
        P = self.dirty_price(last=last, quote=quote, date=date, day_counter=day_counter,
                             compounding=compounding, frequency=frequency,
                             settlement_days=settlement_days, bypass_set_floating_rate_index=True)
        dy = 1e-5
        forecast_curve_timeseries = self.reference_curve
        node_dates = forecast_curve_timeseries.yield_curve(date=date).dates()
        node_rates = forecast_curve_timeseries.zero_rate_to_date(date=date, to_date=node_dates,
                                                                 compounding=ql.Simple,
                                                                 frequency=ql.Annual)
        lower_shifted_curve = ql.LogLinearZeroCurve(list(node_dates), [r - dy for r in node_rates],
                                                    forecast_curve_timeseries.day_counter,
                                                    forecast_curve_timeseries.calendar,
                                                    ql.LogLinear(),
                                                    ql.Simple,
                                                    )
        upper_shifted_curve = ql.LogLinearZeroCurve(list(node_dates), [r + dy for r in node_rates],
                                                    forecast_curve_timeseries.day_counter,
                                                    forecast_curve_timeseries.calendar,
                                                    ql.LogLinear(),
                                                    ql.Simple,
                                                    )
        self.forecast_curve.linkTo(lower_shifted_curve)
        P_m = ql.CashFlows.npv(self.bond.cashflows(),
                               ql.InterestRate(ytm - dy,
                                               self.day_counter,
                                               self.yield_quote_compounding,
                                               self.yield_quote_frequency),
                               True, settlement_date, settlement_date)
        self.forecast_curve.linkTo(upper_shifted_curve)
        P_p = ql.CashFlows.npv(self.bond.cashflows(),
                               ql.InterestRate(ytm + dy,
                                               self.day_counter,
                                               self.yield_quote_compounding,
                                               self.yield_quote_frequency),
                               True, settlement_date, settlement_date)
        return (1 + ytm / self.yield_quote_frequency) * -(1/P)*(P_p - P_m)/(2*dy)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def duration_to_worst(self, duration_type, last, quote, date, day_counter, compounding, frequency, settlement_days,
                          **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Same as duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """
        return self.duration_to_mat(duration_type=duration_type, last=last, quote=quote, day_counter=day_counter,
                                    compounding=compounding, frequency=frequency, settlement_days=settlement_days,
                                    bypass_set_floating_rate_index=True)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def duration_to_worst_rolling_call(self, duration_type, last, quote, date, day_counter, compounding, frequency,
                                       settlement_days, **kwargs):
        """
        Parameters
        ----------
        duration_type: QuantLib.Duration.Type
            The duration type.
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Same as duration to maturity.

        Note
        ----
        Duration methods need different implementation for floating rate bonds.
        See Balaraman G., Ballabio L. QuantLib Python Cookbook [ch. Duration of floating-rate bonds].
        """
        return self.duration_to_mat(duration_type=duration_type, last=last, quote=quote, day_counter=day_counter,
                                    compounding=compounding, frequency=frequency, settlement_days=settlement_days,
                                    bypass_set_floating_rate_index=True)

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def convexity_to_mat(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Bond's convexity to maturity.
        """
        # TODO: Implement this.
        return np.nan

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def convexity_to_worst(self, last, quote, date, day_counter, compounding, frequency, settlement_days, **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Bond's convexity to worst.
        """
        # TODO: Implement this.
        return np.nan

    @default_arguments
    @conditional_vectorize('quote', 'date')
    @set_floating_rate_index
    def convexity_to_worst_rolling_call(self, last, quote, date, day_counter, compounding, frequency, settlement_days,
                                        **kwargs):
        """
        Warnings
        --------
        This method is not yet implemented. Returns Numpy.nan by default.

        Parameters
        ----------
        last: bool, optional
            Whether to last data.
            Default: see :py:func:`default_arguments`.
        quote: scalar, optional
            Bond's quote.
            Default: see :py:func:`default_arguments`.
        date: QuantLib.Date, optional
            Date of the calculation.
            Default: see :py:func:`default_arguments`.
        day_counter: QuantLib.DayCounter, optional
            Day counter for the calculation.
            Default: see :py:func:`default_arguments`.
        compounding: QuantLib.Compounding, optional
            Compounding convention for the calculation.
            Default: see :py:func:`default_arguments`.
        frequency: QuantLib.Frequency, optional
            Compounding frequency.
            Default: see :py:func:`default_arguments`.
        settlement_days: int, optional
            Number of days for trade settlement.
            Default: see :py:func:`default_arguments`.

        Returns
        -------
        scalar
            Bond's convexity to worst.
        """
        # TODO: Implement this.
        return np.nan


'''
###########################################################################################
Methods of the FloatingRateBond class need to be decorated with set_floating_rate_index.
So we unwrap the base class methods from their wrappers and include set_floating_rate_index 
in the wrapper chain.
###########################################################################################
'''

# Methods to redecorate with set_floating_rate_index.
methods_to_redecorate = ['accrued_interest',
                         'cash_to_date',
                         'clean_price',
                         'dirty_price',
                         'value',
                         'performance',
                         'ytm',
                         'ytw_and_worst_date',
                         'ytw',
                         'ytw_and_worst_date_rolling_call',
                         'ytw_rolling_call',
                         'yield_to_date',
                         'worst_date',
                         'worst_date_rolling_call',
                         'clean_price_from_ytm',
                         'clean_price_from_yield_to_date',
                         'dirty_price_from_ytm',
                         'dirty_price_from_yield_to_date',
                         'zspread_to_mat',
                         'zspread_to_worst',
                         'zspread_to_worst_rolling_call',
                         'oas',
                         ]


def redecorate_with_set_floating_rate_index(cls, method):
    if not getattr(method, '_decorated_by_floating_rate_index_', None):
        if getattr(method, '_conditional_vectorized_', None) and \
                getattr(method, '_decorated_by_default_arguments_', None):
            vectorized_args = method._conditional_vectorize_args_
            setattr(cls, method.__name__, default_arguments(conditional_vectorize(vectorized_args)(
                set_floating_rate_index(method))))
        elif getattr(method, '_conditional_vectorized_', None):
            vectorized_args = method._conditional_vectorize_args_
            setattr(cls, method.__name__, conditional_vectorize(vectorized_args)(set_floating_rate_index(method)))
        elif getattr(method, '_decorated_by_default_arguments_', None):
            setattr(cls, method.__name__, default_arguments(set_floating_rate_index(method)))


# Redecorating methods.
for method_name in methods_to_redecorate:
    redecorate_with_set_floating_rate_index(FloatingRateBond, getattr(FloatingRateBond, method_name))



