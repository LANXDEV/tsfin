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
from tsfin.constants import ISSUE_DATE_ATTRIBUTES
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import to_list, conditional_vectorize, find_le, find_gt

# The default issue_date is used if it is impossible to decide the issue date of a given TimeSeries.
DEFAULT_ISSUE_DATE = ql.Date.minDate()

# ExtRateHelpers are a namedtuples containing QuantLib RateHelpers objects and other meta-information.
ExtRateHelper = namedtuple('ExtRateHelper', ['ts_name', 'issue_date', 'tenor', 'maturity_date', 'helper'])


class YieldCurveTimeSeries:

    def __init__(self, ts_collection, calendar, day_counter, keep_only_on_the_run_month=False, ignore_errors=False,
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
        self.ts_collection = ts_collection
        self.calendar = calendar
        self.day_counter = day_counter
        self.keep_only_on_the_run_month = keep_only_on_the_run_month
        self.yield_curves = dict()
        self.ignore_errors = ignore_errors
        self.other_rate_helper_args = other_rate_helper_args

        self.issue_dates = dict()
        # TODO: Remove issue_dates inspection from this class and add an issue_date attribute to all instrument.
        # classes.
        # Saving the issue dates.
        for ts in ts_collection:
            issue_date = None
            for issue_attribute in ISSUE_DATE_ATTRIBUTES:
                try:
                    issue_date = to_ql_date(ts.get_attribute(issue_attribute))
                    break
                except AttributeError:
                    continue
            if issue_date is None:
                # If impossible to decide the issue_date, it remains equal to "None" as we initialized above.
                # Then set it to 2000-01-01.
                issue_date = DEFAULT_ISSUE_DATE
            self.issue_dates[ts.ts_name] = issue_date

    def _get_helpers(self, date):
        helpers = dict()
        ql_date = to_ql_date(date)

        for ts in self.ts_collection:
            ts_name = ts.ts_name
            issue_date = self.issue_dates[ts_name]
            helper = ts.rate_helper(date=date, **self.other_rate_helper_args)
            if helper is not None:
                tenor = ts.tenor(date)
                maturity_date = self.calendar.advance(ql_date, tenor)
                '''
                TODO: Wrap this ``ExtRateHelper`` inside a (properly named) class or namedtuple and always
                return these objects from the instrument classes. This prevents this method from calling ``ts.tenor``
                and recalculating the ``maturity_date``, because these were already calculated inside each instrument's
                ``rate_helper`` method.
                '''
                helper = ExtRateHelper(ts_name=ts_name, issue_date=issue_date, tenor=tenor,
                                       maturity_date=maturity_date, helper=helper)
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
                    month_end_helpers[month_year] = max((helper, existing_helper),
                                                        key=attrgetter('issue_date'))
            helpers = {ndhelper.maturity_date: ndhelper for ndhelper in month_end_helpers.values()}

        return helpers

    def update_curves(self, dates):
        """ Update ``self.yield_curves`` with the yield curves of each date in `dates`.

        Parameters
        ----------
        dates: list of QuantLib.Dates
        """
        dates = to_list(dates)

        for date in dates:
            print("Calculating for date...")
            ql_date = to_ql_date(date)
            ql.Settings.instance().evaluationDate = ql_date

            helpers_dict = self._get_helpers(date)

            # Instantiate the curve
            helpers = [ndhelper.helper for ndhelper in helpers_dict.values()]
            yield_curve = ql.PiecewiseLinearZero(ql_date, helpers, self.day_counter)  # Just bootstraping the nodes

            # Get dates and discounts
            node_dates = yield_curve.dates()
            # node_discounts = [yield_curve.discount(date) for date in node_dates]
            node_rates = [yield_curve.zeroRate(date, self.day_counter, ql.Continuous).rate() for date in node_dates]

            # Freezing the curve so that nothing is bothered by changing the singleton (global variable) evaluationDate.
            # yield_curve = ql.DiscountCurve(node_dates, node_discounts, yield_curve.dayCounter())
            yield_curve = ql.MonotonicCubicZeroCurve(node_dates, node_rates,
                                                     self.day_counter,
                                                     self.calendar,
                                                     ql.MonotonicCubic(),
                                                     ql.Continuous,
                                                     )
            yield_curve.enableExtrapolation()

            self.yield_curves[date] = yield_curve

    def _update_all_curves(self):
        index = self.ts_collection[0].ts_values.index.tolist()
        for i in range(1, len(self.ts_collection)):
            index += self.ts_collection[i].ts_values.index.tolist()
        counted_dates = Counter(index)
        possible_dates = [date for date, count in counted_dates.items() if count >= 2]
        self.update_curves(possible_dates)

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
                    return self.yield_curve(latest_available_date)
            else:
                self.update_curves(date)
            return self.yield_curves[date]

    def spreaded_curve(self, date, spread, compounding, frequency):
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

        Returns
        -------
        QuantLib.ZeroSpreadedTermStructure
            The spreaded yield curve.
        """
        # TODO: Check if this is working.
        curve_handle = ql.YieldTermStructureHandle(self.yield_curve(date=date))
        spread_handle = ql.QuoteHandle(spread)
        return ql.ZeroSpreadedTermStructure(curve_handle, spread_handle, compounding, frequency, self.day_counter)

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

    @conditional_vectorize('date', 'to_date')
    def discount_to_date(self, date, to_date):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            The date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            The maturity for the discount rate.

        Returns
        -------
        scalar
            The discount rate for `to_date` implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        return self.yield_curve(date).discount(to_date)

    @conditional_vectorize('date', 'to_date')
    def zero_rate_to_date(self, date, to_date, compounding, frequency, extrapolate=True):
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

        Returns
        -------
        scalar
            Zero rate for `to_date`, implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        return self.yield_curve(date).zeroRate(to_date, self.day_counter, compounding, frequency, extrapolate).rate()

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
        return self.yield_curve(date).zeroRate(to_time, compounding, frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_date1', 'to_date2')
    def forward_rate_date_to_date(self, date, to_date1, to_date2, compounding, frequency, extrapolate=True):
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

        Returns
        -------
        scalar
            Forward rate between `to_date1` and `to_date2`, implied by the yield curve at `date`.
        """
        to_date1 = to_ql_date(to_date1)
        to_date2 = to_ql_date(to_date2)
        return self.yield_curve(date).forwardRate(to_date1, to_date2, self.day_counter, compounding, frequency,
                                                  extrapolate).rate()

    @conditional_vectorize('date', 'to_date', 'to_time')
    def forward_rate_date_to_time(self, date, to_date, to_time, compounding, frequency, extrapolate=True):
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
        extrapolate: bool, optional
            Whether to enable extrapolation.

        Returns
        -------
        scalar
            Forward rate between `to_date` and `to_time`, implied by the yield curve at `date`.
        """
        to_date = to_ql_date(to_date)
        to_date2 = self.calendar.advance(to_date, to_time)
        return self.yield_curve(date).forwardRate(to_date, to_date2, self.day_counter, compounding, frequency,
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
        return self.yield_curve(date).forwardRate(to_time1, to_time2, compounding, frequency, extrapolate).rate()

    @staticmethod
    def _date_to_month_year(dt_object):
        return str(dt_object.month()) + '-' + str(dt_object.year())
