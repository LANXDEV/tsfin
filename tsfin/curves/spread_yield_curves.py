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
SpreadedYieldCurveTimeSeries, a class to handle a time series of spreaded yield curves.
"""

import QuantLib as ql
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import conditional_vectorize


class InterpolatedSpreadYieldCurveTimeSeries:

    def __init__(self, yield_curve_time_series, spreads, compounding, frequency, day_counter):
        self.yield_curve_time_series = yield_curve_time_series
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

        spread_list = list()
        date_list = list()
        for tenor_date in sorted(spreads_at_date.keys()):
            date_list.append(tenor_date)
            spread_list.append(ql.QuoteHandle(spreads_at_date[tenor_date]))

        curve_handle = self.yield_curve_time_series.yield_curve_handle(date=ql_date)

        self.spreaded_curves[ql_date] = ql.SpreadedLinearZeroInterpolatedTermStructure(curve_handle, spread_list,
                                                                                       date_list, compounding,
                                                                                       frequency, day_counter)

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


class SingleSpreadYieldCurveTimeSeries:

    def __init__(self, yield_curve_time_series, spreads, compounding, frequency, day_counter):
        self.yield_curve_time_series = yield_curve_time_series
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

        self.spreaded_curves[ql_date] = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(spreads_at_date),
                                                                     compounding, frequency, day_counter)

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
