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

# ExtRateHelpers are a named tuples containing QuantLib RateHelpers objects and other meta-information.
ExtRateHelper = namedtuple('ExtRateHelper', ['ts_name', 'issue_date', 'tenor', 'maturity_date', 'helper'])


class CDSCurveTimeSeries:

    def __init__(self, ts_collection, base_yield_curve, calendar, day_counter, keep_only_on_the_run_month=False,
                 ignore_errors=False, **other_rate_helper_args):

        """Time series of QuantLib YieldTermStructures objects.

        The QuantLib YieldTermStructure objects are stored in the dict self.yield_curves and are 'lazy' created and
        stored when requested.

        Parameters
        ----------
        ts_collection: :py:obj:`TimeSeriesCollection`
            Collection of instruments for building the yield curves.
        base_yield_curve: :py:obj: 'YieldCurveTimeSeries'
            Base yield curve used for discounting the cash flows.
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
        self.base_yield_curve = base_yield_curve
        self.calendar = calendar
        self.day_counter = day_counter
        self.keep_only_on_the_run_month = keep_only_on_the_run_month
        self.hazard_curves = dict()
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
            yield_curve_handle = self.base_yield_curve_handle(date)
            helper = ts.cds_rate_helper(date=date, base_yield_curve_handle=yield_curve_handle,
                                        **self.other_rate_helper_args)
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
            ql_date = to_ql_date(date)
            ql.Settings.instance().evaluationDate = ql_date

            helpers_dict = self._get_helpers(date)

            # Instantiate the curve
            helpers = [ndhelper.helper for ndhelper in helpers_dict.values()]
            # Just bootstrapping the nodes
            hazard_curve = ql.PiecewiseFlatHazardRate(ql_date, helpers, self.day_counter)

            hazard_curve.enableExtrapolation()

            self.hazard_curves[date] = hazard_curve

    def _update_all_curves(self):
        index = self.ts_collection[0].ts_values.index.tolist()
        for i in range(1, len(self.ts_collection)):
            index += self.ts_collection[i].ts_values.index.tolist()
        counted_dates = Counter(index)
        possible_dates = [date for date, count in counted_dates.items() if count >= 2]
        self.update_curves(possible_dates)

    @conditional_vectorize('date')
    def hazard_curve(self, date):
        """ Get the QuantLib yield curve object at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the yield curve.

        Returns
        -------
        QuantLib.PiecewiseFlatHazardRate
            The hazard curve at `date`.
        """
        try:
            # Try to return the yield curves if it is stored in self.yield_curves.
            return self.hazard_curves[date]
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
                    curve_dates = list(self.hazard_curves.keys())
                    try:
                        # Try to find latest date less than 'date'.
                        latest_available_date = find_le(curve_dates, date)
                    except ValueError:
                        # No latest date less than 'date', try to find earliest date higher than 'date' instead.
                        latest_available_date = find_gt(curve_dates, date)
                    return self.hazard_curve(latest_available_date)
            else:
                self.update_curves(date)
            return self.hazard_curves[date]

    @conditional_vectorize('date')
    def probability_curve_handle(self, date):
        """ Handle for a yield curve at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            Date of the yield curve.

        Returns
        -------
        QuantLib.DefaultProbabilityTermStructureHandle
            A handle to the yield term structure object.
        """
        return ql.DefaultProbabilityTermStructureHandle(self.hazard_curve(date))

    @conditional_vectorize('date')
    def probability_curve_relinkable_handle(self, date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.

        Returns
        -------
        QuantLib.RelinkableDefaultProbabilityTermStructureHandle
            A relinkable handle to the yield term structure object.
        """
        return ql.RelinkableDefaultProbabilityTermStructureHandle(self.hazard_curve(date))

    @staticmethod
    def _date_to_month_year(dt_object):
        return str(dt_object.month()) + '-' + str(dt_object.year())

    @conditional_vectorize('date')
    def base_yield_curve_handle(self, date):
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
        return self.base_yield_curve.yield_curve_handle(date)

    @conditional_vectorize('date')
    def base_yield_curve_relinkable_handle(self, date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.

        Returns
        -------
        QuantLib.RelinkableYieldTermStructureHandle
            A relinkable handle to the yield term structure object.
        """
        return self.base_yield_curve.yield_curve_relinkable_handle(date)

    @conditional_vectorize('date')
    def survival_probability(self, date, period):
        """
        The survival probability given a date and period

        :param date: Date of the yield curve.
        :param period: The tenor for the maturity.
        :return: The % chance of survival given the date and tenor.
        """

        ql_date = to_ql_date(date)
        ql_period = ql.Period(period)

        return self.hazard_curves[date].survivalProbability(ql_date + ql_period)

    @conditional_vectorize('date')
    def default_probability(self, date, period):
        """
        The survival probability given a date and period

        :param date: Date of the yield curve.
        :param period: The tenor for the maturity.
        :return: The % chance of default given the date and tenor.
        """

        ql_date = to_ql_date(date)
        ql_period = ql.Period(period)

        return self.hazard_curves[date].defaultProbability(ql_date + ql_period)
