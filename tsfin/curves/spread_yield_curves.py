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
InterpolatedSpreadYieldCurveTimeSeries, a class to handle a time series of spreaded yield curves.
SingleSpreadYieldCurveTimeSeries, a class to handle a time series of single spread yield curves.
"""

import QuantLib as ql
from tsfin.base import to_ql_date, conditional_vectorize, to_list


class InterpolatedSpreadYieldCurveTimeSeries:

    def __init__(self, yield_curve_time_series, spreads_ts_collection=None, spreads=None, day_counter=None,
                 compounding=ql.Compounded, frequency=ql.Annual, **kwargs):
        self.yield_curve_time_series = yield_curve_time_series
        self.spreads_ts_collection = spreads_ts_collection
        self.spreads = spreads
        self.day_counter = day_counter
        self.calendar = yield_curve_time_series.calendar
        self.compounding = compounding
        self.frequency = frequency
        self.spreaded_curves = dict()

    def update_curves(self, dates):

        dates = to_list(dates)
        for date in dates:
            date = to_ql_date(date)

            ql.Settings.instance().evaluationDate = date
            spread_dict = dict()
            if self.spreads_ts_collection is not None:
                for ts in self.spreads_ts_collection:
                    maturity = ts.maturity(date=date)
                    rate = ts.spread_rate(date=date).equivalentRate(self.day_counter, self.compounding, self.frequency,
                                                                    date, maturity).rate()
                    spread_dict[maturity] = ql.SimpleQuote(rate)
            elif self.spreads is not None:
                # passing the dict with dates and simple quotes directly.
                spread_dict = self.spreads

            date_list = list()
            spread_list = list()
            for maturity in sorted(spread_dict.keys()):
                date_list.append(maturity)
                spread_list.append(ql.QuoteHandle(spread_dict[maturity]))

            curve_handle = ql.YieldTermStructureHandle(self.yield_curve_time_series.yield_curve(date=date))
            spread_curve = ql.SpreadedLinearZeroInterpolatedTermStructure(curve_handle, spread_list, date_list,
                                                                          self.compounding, self.frequency,
                                                                          self.day_counter)
            spread_curve.enableExtrapolation()

            self.spreaded_curves[date] = spread_curve

    def yield_curve(self, date):

        """ Get the QuantLib yield curve object at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the yield curve.

        Returns
        -------
        QuantLib.SpreadedLinearZeroInterpolatedTermStructure
            The yield curve at `date`.
        """

        date = to_ql_date(date)
        try:
            return self.spreaded_curves[date]

        except KeyError:
            self.update_curves(dates=date)
            return self.spreaded_curves[date]

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

    @conditional_vectorize('future_date')
    def implied_term_structure_handle(self, date, future_date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.
        future_date: Date of the Implied Yield Curve
        Returns
        -------
        QuantLib.RelinkableYieldTermStructureHandle
            A relinkable handle to the yield term structure object.
        """
        future_date = to_ql_date(future_date)
        return ql.ImpliedTermStructure(self.yield_curve_handle(date), future_date)

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
        date = to_ql_date(date)
        to_date = to_ql_date(to_date)
        day_counter = day_counter if day_counter is not None else self.day_counter
        return self.yield_curve(date).zeroRate(to_date, day_counter, compounding, frequency, extrapolate).rate()

    @conditional_vectorize('date', 'to_time')
    def zero_rate_to_time(self, date, to_time, compounding, frequency, extrapolate=True, **kwargs):
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
        return self.yield_curve(date, **kwargs).zeroRate(to_time, compounding, frequency, extrapolate).rate()

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


class SingleSpreadYieldCurveTimeSeries:

    def __init__(self, yield_curve_time_series, spreads, compounding, frequency, day_counter):
        self.yield_curve_time_series = yield_curve_time_series
        self.calendar = yield_curve_time_series.calendar
        self.spreads = spreads
        self.compounding = compounding
        self.frequency = frequency
        self.day_counter = day_counter
        self.spreaded_curves = dict()

    def spreaded_curve(self, date, spreads=None, compounding=None, frequency=None, day_counter=None):

        ql_date = to_ql_date(date)
        if compounding is None:
            compounding = self.compounding
        if frequency is None:
            frequency = self.frequency
        if day_counter is None:
            day_counter = self.day_counter
        if spreads is None:
            spreads_at_date = self.spreads[ql_date]
        else:
            spreads_at_date = spreads[ql_date]

        curve_handle = self.yield_curve_time_series.yield_curve_handle(date=ql_date)
        yield_curve = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(spreads_at_date), compounding,
                                                   frequency, day_counter)
        yield_curve.enableExtrapolation()

        self.spreaded_curves[ql_date] = yield_curve

    def yield_curve(self, date):

        """ Get the QuantLib yield curve object at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the yield curve.

        Returns
        -------
        QuantLib.SpreadedLinearZeroInterpolatedTermStructure
            The yield curve at `date`.
        """

        ql_date = to_ql_date(date)
        try:
            return self.spreaded_curves[ql_date]

        except KeyError:
            self.spreaded_curve(date=date)
            return self.spreaded_curves[ql_date]

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

    @conditional_vectorize('future_date')
    def implied_term_structure_handle(self, date, future_date):
        """ A relinkable handle for a yield curve at a given date.

        Parameters
        ----------
        date: Date of the yield curve.
        future_date: Date of the Implied Yield Curve
        Returns
        -------
        QuantLib.RelinkableYieldTermStructureHandle
            A relinkable handle to the yield term structure object.
        """
        future_date = to_ql_date(future_date)
        return ql.ImpliedTermStructure(self.yield_curve_handle(date), future_date)

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

    @conditional_vectorize('date', 'to_time')
    def zero_rate_to_time(self, date, to_time, compounding, frequency, extrapolate=True, **kwargs):
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


