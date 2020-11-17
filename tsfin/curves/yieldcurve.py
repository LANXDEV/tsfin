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
YieldCurveTimeSeries, a class to handle a time series of yield curves.
"""

from collections import namedtuple, Counter
from operator import attrgetter
import QuantLib as ql
from tsfin.base import to_list, conditional_vectorize, find_le, find_gt, to_ql_date, to_ql_piecewise_curve, \
    to_ql_interpolated_curve


# ExtRateHelpers are a named tuples containing QuantLib RateHelpers objects and other meta-information.
ExtRateHelper = namedtuple('ExtRateHelper', ['ts_name', 'issue_date', 'maturity_date', 'helper'])


class SimpleYieldCurve:
    """Base class with the necessary Yield Curve methods.

    Parameters
    ----------
    calendar: QuantLib.Calendar
        Calendar for the yield curves.
    day_counter: QuantLib.DayCounter
        Day counter for the yield curves.
    ignore_errors: bool, optional
        Use last available yield curve if building a yield curve fails at a given date. Defaults to False.
    """

    def __init__(self, calendar, day_counter, enable_extrapolation, ignore_errors):
        self.yield_curves = dict()
        self.calendar = calendar
        self.day_counter = day_counter
        self.enable_extrapolation = enable_extrapolation
        self.ignore_errors = ignore_errors

    def update_curves(self, dates):
        pass

    def yield_curve(self, date):
        """ Get the QuantLib yield curve object at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the yield curve.

        Returns
        -------
        QuantLib.YieldTermStructure
            The yield curve at `date`.
        """
        date = to_ql_date(date)
        try:
            # Try to return the yield curves if it is stored in self.yield_curves.
            return self.yield_curves[date]
        except KeyError:
            # If the required yield curve is not in self.yield_curves,
            # then create it and return it, updating self.yield_curves along the way.
            if self.ignore_errors:
                # If ignore_errors is set to True, return latest available curve if there is any error in calculating
                #  the curve in the desired 'date'.
                try:
                    self.update_curves(date)
                except Exception as e:
                    print("Error in creating curve in {}".format(date))
                    print(e)
                    curve_dates = list(self.yield_curves.keys())
                    try:
                        # Try to find latest date less than 'date'.
                        latest_available_date = find_le(curve_dates, date)
                    except ValueError:
                        # No latest date less than 'date', try to find earliest date higher than 'date' instead.
                        latest_available_date = find_gt(curve_dates, date)
                    return self.yield_curve(to_ql_date(latest_available_date))
            else:
                self.update_curves(date)
            return self.yield_curves[date]

    def spreaded_curve(self, date, spread, compounding, frequency, day_counter=None):
        """ Get yield curve at a date, added to a spread.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the spreaded yield curve.
        spread: scalar
            The spread to be added to the yield curve rates.
        compounding: QuantLib.Compounding
            The compounding convention of the spread.
        frequency: QuantLib.Frequency
            The frequency convention of the spread.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        QuantLib.ZeroSpreadedTermStructure
            The spreaded yield curve.
        """
        day_counter = day_counter if day_counter is not None else self.day_counter
        curve_handle = ql.YieldTermStructureHandle(self.yield_curve(date=date))
        spread_handle = ql.QuoteHandle(ql.SimpleQuote(spread))
        return ql.ZeroSpreadedTermStructure(curve_handle, spread_handle, compounding, frequency, day_counter)

    def spreaded_interpolated_curve(self, date, spread_dict, compounding, frequency, day_counter=None):
        """ Get yield curve at a date, added to a spread.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the spreaded yield curve.
        spread_dict: dict
            The dict with spreads and tenors
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
        Returns
        -------
        QuantLib.ZeroSpreadedTermStructure
            The spreaded yield curve.
        """

        date = to_ql_date(date)
        curve_handle = self.yield_curve_handle(date=date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        spreads = list()
        dates = list()
        spread_dict_at_date = spread_dict[date]
        for tenor_date in sorted(spread_dict_at_date.keys()):
            dates.append(tenor_date)
            time = self.day_counter.yearFraction(date, tenor_date)
            rate = spread_dict_at_date[tenor_date].equivalentRate(compounding, frequency, time).rate()
            spreads.append(ql.QuoteHandle(ql.SimpleQuote(rate)))

        spreaded_curve = ql.SpreadedLinearZeroInterpolatedTermStructure(curve_handle, spreads, dates, compounding,
                                                                        frequency, day_counter)
        spreaded_curve.enableExtrapolation()

        return spreaded_curve

    def yield_curve_handle(self, date):
        """ Handle for a yield curve at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            Date of the yield curve.

        Returns
        -------
        QuantLib.YieldTermStructureHandle
            A handle to the yield term structure object.
        """
        return ql.YieldTermStructureHandle(self.yield_curve(date))

    def spreaded_interpolated_curve_handle(self, date, spread_dict, compounding, frequency, day_counter=None):
        """ Handle for a spreaded interpolated yield curve at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            Date of the yield curve.
        spread_dict: dict
            The dict with spreads and tenors
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.
        Returns
        -------
        QuantLib.YieldTermStructureHandle
            A handle to the yield term structure object.
        """

        spreaded_curve = self.spreaded_interpolated_curve(date=date, spread_dict=spread_dict,
                                                          compounding=compounding, frequency=frequency,
                                                          day_counter=day_counter)

        return ql.YieldTermStructureHandle(spreaded_curve)

    def yield_curve_relinkable_handle(self, date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.

        Returns
        -------
        QuantLib.RelinkableYieldTermStructureHandle
            A relinkable handle to the yield term structure object.
        """
        return ql.RelinkableYieldTermStructureHandle(self.yield_curve(date))

    @conditional_vectorize('future_date')
    def implied_term_structure(self, date, future_date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.
        future_date: Date of the Implied Yield Curve
        Returns
        -------
        QuantLib.ImpliedTermStructure
            The implied term structure at the future date from date
        """
        future_date = to_ql_date(future_date)
        return ql.ImpliedTermStructure(self.yield_curve_handle(date), future_date)

    @conditional_vectorize('date', 'to_date')
    def discount_to_date(self, date, to_date, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            The date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            The maturity for the discount rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            The discount rate for `to_date` implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        return self.yield_curve(date).discount(to_date, extrapolate)

    @conditional_vectorize('date', 'to_time')
    def discount_to_time(self, date, to_time, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            The date of the yield curve.
        to_time: scalar, (c-vectorized)
            Tenor in years of the zero rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            The discount rate for `to_date` implied by the yield curve at `date`.
        """
        return self.yield_curve(date).discount(to_time, extrapolate)

    @conditional_vectorize('date', 'to_date')
    def zero_rate_to_date(self, date, to_date, compounding, frequency, extrapolate=True, day_counter=None):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            Maturity of the rate.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        scalar
            Zero rate for `to_date`, implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        return self.yield_curve(date).zeroRate(to_date, day_counter, compounding, frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_date')
    def spreaded_zero_rate_to_date(self, date, to_date, compounding, frequency, spread_dict, extrapolate=True,
                                   day_counter=None):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            Maturity of the rate.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        spread_dict: dict
            The dict with spreads and tenors.
        extrapolate: bool, optional
            Whether to enable extrapolation.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        scalar
            Zero rate for `to_date`, implied by the spreaded yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        spread_curve = self.spreaded_interpolated_curve(date=date, spread_dict=spread_dict, compounding=compounding,
                                                        frequency=frequency, day_counter=day_counter)

        return spread_curve.zeroRate(to_date, day_counter, compounding, frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_time')
    def zero_rate_to_time(self, date, to_time, compounding, frequency, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_time: scalar, (c-vectorized)
            Tenor in years of the zero rate.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            Zero rate for `to_time`, implied by the yield curve at `date`.
        """
        to_time = float(to_time)
        return self.yield_curve(date).zeroRate(to_time, compounding, frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_date1', 'to_date2')
    def forward_rate_date_to_date(self, date, to_date1, to_date2, compounding, frequency, extrapolate=True,
                                  day_counter=None):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_date1: QuantLib.Date, (c-vectorized)
            First maturity for the fra.
        to_date2: QuantLib.Date, (c-vectorized)
            Second maturity for the fra.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        scalar
            Forward rate between `to_date1` and `to_date2`, implied by the yield curve at `date`.
        """
        to_date1 = to_ql_date(to_date1)
        to_date2 = to_ql_date(to_date2)
        day_counter = day_counter if day_counter is not None else self.day_counter
        return self.yield_curve(date).forwardRate(to_date1, to_date2, day_counter, compounding, frequency,
                                                  extrapolate).rate()

    @conditional_vectorize('date', 'to_date', 'to_time')
    def forward_rate_date_to_time(self, date, to_date, to_time, compounding, frequency, to_period=None,
                                  extrapolate=True, day_counter=None):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            First maturity for the fra.
        to_time: scalar, (c-vectorized)
            Time in years after `to_date` for the fra.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        to_period: QuantLib.Period, optional
            The period of the to_time
        extrapolate: bool, optional
            Whether to enable extrapolation.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        scalar
            Forward rate between `to_date` and `to_time`, implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        to_period = to_period if to_period is not None else ql.Years
        to_date2 = self.calendar.advance(to_date, to_time, to_period)
        return self.yield_curve(date).forwardRate(to_date, to_date2, day_counter, compounding, frequency,
                                                  extrapolate).rate()

    @conditional_vectorize('date', 'to_time1', 'to_time2')
    def forward_rate_time_to_time(self, date, to_time1, to_time2, compounding, frequency, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_time1: scalar, (c-vectorized)
            First time in years for the fra.
        to_time2: scalar, (c-vectorized)
            Second time in years for the fra.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            Forward rate between `to_time1` and `to_time2`, implied by the yield curve at `date`.
        """
        return self.yield_curve(date).forwardRate(to_time1, to_time2, compounding, frequency,
                                                  extrapolate).rate()

    @conditional_vectorize('date', 'to_date')
    def implied_zero_rate_to_date(self, date, future_date, to_date, compounding, frequency, extrapolate=True,
                                  day_counter=None):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        future_date: Date of the Implied Yield Curve
            Date of the implied yield curve
        to_date: QuantLib.Date, (c-vectorized)
            Maturity of the rate.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.
        day_counter: QuantLib.DayCounter, optional
            The day counter for the calculation.

        Returns
        -------
        scalar
            Zero rate for `to_date`, implied by the yield curve at `date` at a 'future_date'.
        """
        to_date = to_ql_date(to_date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        return self.implied_term_structure(date, future_date).zeroRate(to_date, day_counter, compounding,
                                                                       frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_time')
    def implied_zero_rate_to_time(self, date, future_date, to_time, compounding, frequency, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        future_date: Date of the Implied Yield Curve
            Date of the implied yield curve
        to_time: scalar, (c-vectorized)
            Tenor in years of the zero rate.
        compounding: QuantLib.Compounding
            Compounding convention for the rate.
        frequency: QuantLib.Frequency
            Frequency convention for the rate.
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            Zero rate for `to_time`, implied by the yield curve at `date`.
        """
        to_time = float(to_time)
        return self.implied_term_structure(date, future_date).zeroRate(to_time, compounding, frequency,
                                                                       extrapolate).rate()

    @staticmethod
    def _date_to_month_year(dt_object):
        return str(dt_object.month()) + '-' + str(dt_object.year())


class YieldCurveTimeSeries(SimpleYieldCurve):

    def __init__(self, ts_collection=None, calendar=None, day_counter=None, keep_only_on_the_run_month=False,
                 ignore_errors=False, curve_type='linear_zero', freeze_curves=True, enable_extrapolation=True,
                 frozen_curve_interpolation_type='monotonic_cubic_zero', constraint_at_zero=True,
                 **other_rate_helper_args):
        """Time series of QuantLib YieldTermStructures objects.

        The QuantLib YieldTermStructure objects are stored in the dict self.yield_curves and are 'lazy' created and
        stored when requested.

        Parameters
        ----------
        ts_collection: :py:obj:`TimeSeriesCollection`
            Collection of instruments for building the yield curves.
        calendar: QuantLib.Calendar
            Calendar for the yield curves.
        day_counter: QuantLib.DayCounter
            Day counter for the yield curves.
        keep_only_on_the_run_month: bool, optional
            Whether to use only one instrument per month for the yield curves. Defaults to False.
        ignore_errors: bool, optional
            Use last available yield curve if building a yield curve fails at a given date. Defaults to False.
        **other_rate_helper_args: key=value pairs, optional
            Additional arguments to pass to ``rate_helper`` methods of the instruments in `ts_collection`.
        """
        super().__init__(calendar=calendar, day_counter=day_counter, ignore_errors=ignore_errors,
                         enable_extrapolation=enable_extrapolation)
        self.ts_collection = ts_collection
        self.keep_only_on_the_run_month = keep_only_on_the_run_month
        self.yield_curves = dict()
        self.curve_type = str(curve_type).lower()
        self.freeze_curves = freeze_curves
        self.frozen_curve_interpolation_type = str(frozen_curve_interpolation_type).lower()
        self.constraint_at_zero = constraint_at_zero
        self.other_rate_helper_args = other_rate_helper_args

    def _get_helpers(self, date):

        helpers = dict()
        for ts in self.ts_collection:
            ts_name = ts.ts_name
            issue_date = ts.issue_date
            helper = ts.rate_helper(date=date, **self.other_rate_helper_args)

            if helper is not None:
                maturity_date = helper.maturityDate()
                helper = ExtRateHelper(ts_name=ts_name, issue_date=issue_date, maturity_date=maturity_date,
                                       helper=helper)
                # Remove Helpers with the same maturity date (or tenor), keeping the last issued one - This is to avoid
                # error in QuantLib when trying to instantiate a yield curve with two helpers with same maturity
                # date or tenor.
                existing_helper = helpers.get(maturity_date, None)
                if existing_helper is None:
                    helpers[maturity_date] = helper
                else:
                    helpers[maturity_date] = max((helper, existing_helper), key=attrgetter('issue_date'))

        if self.keep_only_on_the_run_month:
            # If self.keep_only_on_the_run_month is True, then filter the returned helpers so that for each month there
            # is not more than one helper. This may be used to avoid distortions in curves being generated by a large
            # amount of TimeSeries.
            month_end_helpers = dict()
            for maturity_date, helper in helpers.items():
                month_year = self._date_to_month_year(maturity_date)
                existing_helper = month_end_helpers.get(month_year, None)
                if existing_helper is None:
                    month_end_helpers[month_year] = helper
                else:
                    month_end_helpers[month_year] = max((helper, existing_helper), key=attrgetter('issue_date'))
            helpers = {ndhelper.maturity_date: ndhelper for ndhelper in month_end_helpers.values()}

        return helpers

    def update_curves(self, dates):
        """ Update ``self.yield_curves`` with the yield curves of each date in `dates`.

        Parameters
        ----------
        dates: QuantLib.Date or list of QuantLib.Date
            The curve dates to be interpolated

        """
        dates = to_list(dates)

        for date in dates:
            date = to_ql_date(date)
            ql.Settings.instance().evaluationDate = date
            helpers_dict = self._get_helpers(date)
            # Instantiate the curve
            helpers = [ndhelper.helper for ndhelper in helpers_dict.values()]
            # Bootstrapping the nodes
            yield_curve = to_ql_piecewise_curve(helpers=helpers,
                                                calendar=self.calendar,
                                                day_counter=self.day_counter,
                                                curve_type=self.curve_type,
                                                constraint_at_zero=self.constraint_at_zero)
            # Here you can choose if you want to use the curve linked to the helpers or a frozen curve.
            # Freezing the curve is needed when you are changing exclusively the global evaluation date,
            # QuantLib helpers don't understand that only the evaluation date is changing and end up using
            # the stored spot rates instead of the implied forward rates.
            # Not freezing the curves is useful when you are changing the underlying prices of the helpers, this way
            # the curve will update accordingly to the changes in the helpers.
            if self.freeze_curves:
                node_dates = yield_curve.dates()
                node_rates = [yield_curve.zeroRate(node_date, self.day_counter, ql.Continuous, ql.NoFrequency).rate()
                              for node_date in node_dates]
                yield_curve = to_ql_interpolated_curve(node_dates=node_dates,
                                                       node_rates=node_rates,
                                                       day_counter=self.day_counter,
                                                       calendar=self.calendar,
                                                       interpolation_type=self.frozen_curve_interpolation_type)
            if self.enable_extrapolation:
                yield_curve.enableExtrapolation()
            self.yield_curves[date] = yield_curve

    def _update_all_curves(self):
        index = self.ts_collection[0].ts_values.index.tolist()
        for i in range(1, len(self.ts_collection)):
            index += self.ts_collection[i].ts_values.index.tolist()
        counted_dates = Counter(index)
        possible_dates = [date for date, count in counted_dates.items() if count >= 2]
        self.update_curves(possible_dates)
