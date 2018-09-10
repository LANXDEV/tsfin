"""
A temporary class to handle sums multiple curves. For example, you need this to calculate a fra of a (Treasury + CDS)
yield curve.

TODO: Implement an equivalent of this class in QuantLib and use it directly inside the YieldCurveTimeSeries class.
"""

import QuantLib as ql
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import conditional_vectorize


class HybridYieldCurveTimeSeries:

    def __init__(self, yield_curves, weights=None):
        """A class to handle sums of multiple yield curves. Has the same methods of the YieldCurveTimeSeries class.

        Parameters
        ----------
        yield_curves: list of :py:obj:`YieldCurveTimeSeries`
            The curves to be summed.
        weights: list of scalars, optional
            The weights of each yield curve in `yield_curves`.

        Note
        ----
        Uses by default the calendar and day counter of the first :py:obj:`YieldCurveTimeSeries` in `yield_curves`.
        """
        self.yield_curves = yield_curves
        if weights is None:
            self.weights = [1 for _ in yield_curves]
        else:
            self.weights = weights
        self.day_counter = self.yield_curves[0].day_counter

    @conditional_vectorize('date', 'to_date', 'zero_rate')
    def _discount_factor_to_date(self, date, to_date, zero_rate, compounding, frequency):
        """ Calculate discount factor to a given date, at a given date, given an interest rate.

        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Reference date of the interest rate.
        to_date: QuantLib.Date, (c-vectorized)
            Maturity of the discount rate.
        zero_rate: scalar, (c-vectorized)
            Interest rate at `date`, with maturity `to_date`.
        compounding: QuantLib.Compounding
            Compounding convention of the interest rate.
        frequency: QuantLib.Frequency
            Compounding frequency of the interest rate.

        Returns
        -------
        scalar
            The discount rate to `to_date`, equivalent to the given interest rate.
        """
        date = to_ql_date(date)
        to_date = to_ql_date(to_date)
        return ql.InterestRate(zero_rate, self.day_counter, compounding, frequency).discountFactor(date, to_date,
                                                                                                   date, to_date)

    @conditional_vectorize('time', 'zero_rate')
    def _discount_factor_to_time(self, time, zero_rate, compounding, frequency):
        """ Calculate discount factor to a given time, at a given date, given an interest rate.

        Parameters
        ----------
        time: scalar, (c-vectorized)
            Time to maturity.
        zero_rate: scalar, (c-vectorized)
            Interest rate.
        compounding: QuantLib.Compounding
            Compounding convention of the interest rate.
        frequency: QuantLib.Frequency
            Compounding frequency of the interest rate.

        Returns
        -------
        scalar
            The discount rate to `time`, equivalent to the given interest rate.
        """
        return ql.InterestRate(zero_rate, self.day_counter, compounding, frequency).discountFactor(time)

    @conditional_vectorize('date1', 'date2', 'discount1', 'discount2')
    def _forward_rate_from_discounts_to_date(self, date1, date2, discount1, discount2, compounding, frequency):
        """ Calculate forward rate given two discount rates at two dates.

        Parameters
        ----------
        date1: QuantLib.Date, (c-vectorized)
            First maturity date.
        date2: QuantLib.Date, (c-vectorized)
            Second maturity date.
        discount1: scalar, (c-vectorized)
            The first discount rate.
        discount2: scalar, (c-vectorized)
            The second discount rate.
        compounding: QuantLib.Compounding
            Compounding convention of the fra.
        frequency: QuantLib.Frequency
            Compounding frequency of the fra.

        Returns
        -------
        scalar
            The forward rate to between `date1` and `date2`.
        """
        return ql.InterestRate.impliedRate(discount1/discount2, self.day_counter, compounding, frequency, date1, date2,
                                           date1, date2).rate()

    @conditional_vectorize('time1', 'time2', 'discount1', 'discount2')
    def _forward_rate_from_discounts_to_time(self, time1, time2, discount1, discount2, compounding, frequency):
        """ Calculate forward rate given two discount rates to two times.

        Parameters
        ----------
        time1: QuantLib.Date, (c-vectorized)
            First time to maturity.
        time2: QuantLib.Date, (c-vectorized)
            Second time to maturity.
        discount1: scalar, (c-vectorized)
            The first discount rate.
        discount2: scalar, (c-vectorized)
            The second discount rate.
        compounding: QuantLib.Compounding
            Compounding convention of the fra.
        frequency: QuantLib.Frequency
            Compounding frequency of the fra.

        Returns
        -------
        scalar
            The forward rate between `time1` and `time2`.
        """
        return ql.InterestRate.impliedRate(discount1/discount2, self.day_counter, compounding, frequency,
                                           time2-time1).rate()

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
        return sum(weight * curve.zero_rate_to_date(date=date, to_date=to_date, compounding=compounding,
                                                    frequency=frequency, extrapolate=extrapolate)
                   for weight, curve in zip(self.weights, self.yield_curves))

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
        return sum(weight * curve.zero_rate_to_time(date=date, to_time=to_time, compounding=compounding,
                                                    frequency=frequency, extrapolate=extrapolate)
                   for weight, curve in zip(self.weights, self.yield_curves))

    def discount_to_date(self, date, to_date, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_date: QuantLib.Date, (c-vectorized)
            Maturity for the discount rate.

        Returns
        -------
        scalar
            Discount rate for `to_date` implied by the yield curve at `date`.
        """
        zero_rate = self.zero_rate_to_date(date=date, to_date=to_date, compounding=ql.Compounded,
                                           frequency=ql.Continuous, extrapolate=extrapolate)
        return self._discount_factor_to_date(date=date, to_date=to_date, zero_rate=zero_rate, compounding=ql.Compounded,
                                             frequency=ql.Continuous)

    def discount_to_time(self, date, to_time, extrapolate=True):
        """
        Parameters
        ----------
        date: QuantLib.Date, (c-vectorized)
            Date of the yield curve.
        to_time: scalar, (c-vectorized)
            Time to maturity for the discount rate.

        Returns
        -------
        scalar
            Discount rate to `to_time` implied by the yield curve at `date`.
        """
        zero_rate = self.zero_rate_to_time(date=date, to_time=to_time, compounding=ql.Compounded,
                                           frequency=ql.Continuous, extrapolate=extrapolate)
        return self._discount_factor_to_time(time=to_time, zero_rate=zero_rate, compounding=ql.Compounded,
                                             frequency=ql.Continuous)

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
        discounts_date1 = self.discount_to_date(date=date, to_date=to_date1, extrapolate=extrapolate)
        discounts_date2 = self.discount_to_date(date=date, to_date=to_date2, extrapolate=extrapolate)
        return self._forward_rate_from_discounts_to_date(date1=to_date1, date2=to_date2, discount1=discounts_date1,
                                                         discount2=discounts_date2, compounding=compounding,
                                                         frequency=frequency)

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
        discounts_date1 = self.discount_to_time(date=date, to_time=to_time1, extrapolate=extrapolate)
        discounts_date2 = self.discount_to_time(date=date, to_time=to_time2, extrapolate=extrapolate)
        return self._forward_rate_from_discounts_to_time(time1=to_time1, time2=to_time2, discount1=discounts_date1,
                                                         discount2=discounts_date2, compounding=compounding,
                                                         frequency=frequency)
