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
Classes to model currency exchange rate curves, and build the yield curves of their base and counter currencies.
Maybe in the future should be added to QuantLib.
"""
from collections import namedtuple, Counter
from operator import attrgetter
import numpy as np
import pandas as pd
import QuantLib as ql
from tsio import TimeSeries, TimeSeriesCollection
from tsfin.instruments import DepositRate
from tsfin.curves import YieldCurveTimeSeries
from tsfin.base import to_ql_date, to_list, find_le, find_gt

# namedtuple representing a 'dated value' for the currency curve:
ExtCurrencyHelper = namedtuple('ExtCurrencyHelper', ['ts_name', 'maturity_date', 'helper'])


class CurrencyCurveTimeSeries:

    def __init__(self, spot_timeseries, currency_futures):
        """ A currency curve capable of generating yield curves for its base and counter currencies.

        Parameters
        ----------
        spot_timeseries: :py:obj:`TimeSeries`
            The time series of spot exchange rates.
        currency_futures: list of py:obj:`CurrencyFuture`
            List with the currency future objects to build the curve.
        """

        self.spot = spot_timeseries
        self.ts_collection = currency_futures
        self.currency_curves = dict()

        # Use one of the currency futures to infer rate attributes.
        a_currency_future = self.ts_collection[0]
        self.calendar = a_currency_future.calendar
        self.counter_rate_day_counter = a_currency_future.counter_rate_day_counter
        self.counter_rate_compounding = a_currency_future.counter_rate_compounding
        self.counter_rate_frequency = a_currency_future.counter_rate_frequency
        self.base_rate_day_counter = a_currency_future.base_rate_day_counter
        self.base_rate_compounding = a_currency_future.base_rate_compounding
        self.base_rate_frequency = a_currency_future.base_rate_frequency

    def _get_spot_and_named_dated_helpers(self, date):
        named_dated_helpers = dict()

        for ts in self.ts_collection:
            if not ts.is_expired(date):
                ts_name = ts.ts_name
                maturity_date = ts.maturity
                helper = ts.helper(date)
                if helper is None:
                    continue
                ndhelper = ExtCurrencyHelper(ts_name=ts_name, maturity_date=maturity_date, helper=helper)
                named_dated_helpers[maturity_date] = ndhelper

        # Retrieving spot value.
        spot_value = self.spot.get_values(index=date, last_available=False, fill_value=np.nan)

        return spot_value, named_dated_helpers

    def update_curves(self, dates):
        """ Update ``self.currency_curves`` with the currency curves of each date in `dates`.

        Parameters
        ----------
        dates: list of QuantLib.Dates
        """
        dates = to_list(dates)

        for date in dates:
            spot_value, named_dated_helpers_dict = self._get_spot_and_named_dated_helpers(date)
            # Instantiate the curve.
            helpers = [ndhelper.helper for ndhelper in named_dated_helpers_dict.values()]
            currency_curve = CurrencyCurve(date, spot_value, helpers, self.counter_rate_day_counter,
                                           self.counter_rate_compounding, self.counter_rate_frequency,
                                           self.base_rate_day_counter, self.base_rate_compounding,
                                           self.base_rate_frequency)
            self.currency_curves[date] = currency_curve

    def _update_all_curves(self):
        index = self.ts_collection[0].ts_values.index.tolist()
        for i in range(1, len(self.ts_collection)):
            index += self.ts_collection[i].ts_values.index.tolist()
        counted_dates = Counter(index)
        possible_dates = [date for date, count in counted_dates.items() if count >= 2]
        self.update_curves(possible_dates)

    def currency_curve(self, date):
        """ Get the currency curve object at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The date of the currency curve.

        Returns
        -------
        :py:obj:CurrencyCurve
            The currency curve at `date`.
        """
        try:
            # Try to return the currency curves if it is stored in self.currency_curves.
            return self.currency_curves[date]
        except KeyError:
            # If the required currency curve is not in self.currency_curves,
            # then create it and return it, updating self.currency_curves along the way.
            self.update_curves(date)
            return self.currency_curves[date]

    def exchange_rate_to_date(self, date, to_date):
        """ Get exchange rate implied by the curve at a given date, to a given date, with linear interpolation.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date of the currency curve.
        to_date: QuantLib.Date
            Maturity date for the future exchange rate.

        Returns
        -------
        scalar
            Exchange rate to `to_date`, at `date`.
        """
        to_date = to_ql_date(to_date)
        return self.currency_curve(date).exchange_rate_to_date(to_date)

    def counter_rate_to_date(self, date, to_date, base_rate):
        """ Get interest rate of the counter currency at a given date, to a given date.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date for the currency curve.
        to_date: QuantLib.Date
            Maturity date of the interest rate.
        base_rate: scalar
            Interest rate of the base currency to `to_date`.

        Returns
        -------
        scalar
            Interest rate of the counter currency to `to_date`.
        """
        to_date = to_ql_date(to_date)
        currency_curve = self.currency_curve(date)
        helper = currency_curve.helpers[0]
        exchange_rate = currency_curve.exchange_rate_to_date(date)
        return helper.counter_rate(quote=exchange_rate, spot=currency_curve.spot, base_rate=base_rate, maturity=to_date)

    def base_rate_to_date(self, date, to_date, counter_rate):
        """ Get interest rate of the base currency at a given date, to a given date.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date for the currency curve.
        to_date: QuantLib.Date
            Maturity date of the interest rate.
        counter_rate: scalar
            Interest rate of the counter currency to `to_date`.

        Returns
        -------
        scalar
            Interest rate of the base currency to `to_date`.
        """
        to_date = to_ql_date(to_date)
        currency_curve = self.currency_curve(date)
        helper = currency_curve.helpers[0]
        exchange_rate = currency_curve.exchange_rate_to_date(date)
        return helper.base_rate(quote=exchange_rate, spot=currency_curve.spot, counter_rate=counter_rate,
                                maturity=to_date)

    def counter_rate_curve(self, date, base_rate_curve):
        """ Get a yield curve for counter currency interest rate at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The reference date for the yield curve.
        base_rate_curve: QuantLib.YieldTermStructure
            The yield curve of base currency interest rate.

        Returns
        -------
        QuantLib.YieldTermStructure
            Yield curve of the counter currency interest rates.
        """
        return self.currency_curve(date).counter_rate_curve(base_rate_curve)

    def base_rate_curve(self, date, counter_rate_curve):
        """ Get a yield curve for base currency interest rate at a given date.

        Parameters
        ----------
        date: QuantLib.Date
            The reference date for the yield curve.
        counter_rate_curve: QuantLib.YieldTermStructure
            The yield curve of counter currency interest rate.

        Returns
        -------
        QuantLib.YieldTermStructure
            Yield curve of the base currency interest rates.
        """
        return self.currency_curve(date).base_rate_curve(counter_rate_curve)

    def counter_rate_curve_time_series(self, base_rate_curve_time_series):
        """ Get a yield curve time series of the counter currency interest rate.

        Parameters
        ----------
        base_rate_curve_time_series: :py:obj:`YieldCurveTimeSeries`

        Returns
        -------
        :py:obj:YieldCurveTimeSeries
            Yield curve time series object of the counter currency interest rate.
        """
        deposit_rates = TimeSeriesCollection()
        for ts in self.ts_collection:
            deposit_rate = TimeSeries('({0})({1})'.format(ts.ts_name, 'COUNTER_RATE'))
            deposit_rate.set_attribute("CALENDAR", ts.get_attribute("CALENDAR"))
            deposit_rate.set_attribute("MATURITY", ts.get_attribute("MATURITY"))
            deposit_rate.set_attribute("DAY_COUNTER", ts.get_attribute("COUNTER_RATE_DAY_COUNTER"))
            deposit_rate.set_attribute("COMPOUNDING", ts.get_attribute("COUNTER_RATE_COMPOUNDING"))
            deposit_rate.set_attribute("FREQUENCY", ts.get_attribute("COUNTER_RATE_FREQUENCY"))
            dates = ts.quotes.index
            maturity_date = ts.maturity_date
            deposit_rate.ts_values = pd.Series(index=dates,
                                               data=ts.counter_rate(quote=ts.quotes.values,
                                                                    spot=self.spot.get_values(index=dates),
                                   base_rate=base_rate_curve_time_series.zero_rate_to_date(date=dates,
                                                                                 to_date=maturity_date,
                                                                                 compounding=ts.base_rate_compounding,
                                                                                 frequency=ts.base_rate_frequency,
                                                                                 ),
                                                                    date=dates)
                                               )
            deposit_rate = DepositRate(deposit_rate)
            deposit_rates.add(deposit_rate)

        return YieldCurveTimeSeries(deposit_rates, calendar=self.calendar, day_counter=self.counter_rate_day_counter)

    def base_rate_curve_time_series(self, counter_rate_curve_time_series):
        """ Get a yield curve time series of the base currency interest rate.

        Parameters
        ----------
        counter_rate_curve_time_series: :py:obj:`YieldCurveTimeSeries`

        Returns
        -------
        :py:obj:YieldCurveTimeSeries
            Yield curve time series object of the base currency interest rate.
        """
        deposit_rates = TimeSeriesCollection()
        for ts in self.ts_collection:
            deposit_rate = TimeSeries('({0})({1})'.format(ts.ts_name, 'COUNTER_RATE'))
            deposit_rate.set_attribute("CALENDAR", ts.get_attribute("CALENDAR"))
            deposit_rate.set_attribute("MATURITY", ts.get_attribute("MATURITY"))
            deposit_rate.set_attribute("DAY_COUNTER", ts.get_attribute("COUNTER_RATE_DAY_COUNTER"))
            deposit_rate.set_attribute("COMPOUNDING", ts.get_attribute("COUNTER_RATE_COMPOUNDING"))
            deposit_rate.set_attribute("FREQUENCY", ts.get_attribute("COUNTER_RATE_FREQUENCY"))
            dates = ts.quotes.index
            maturity_date = ts.maturity_date
            deposit_rate.ts_values = pd.Series(index=dates,
                                               data=ts.base_rate(quote=ts.quotes.values,
                                                                 spot=self.spot.get_values(index=dates),
                        counter_rate=counter_rate_curve_time_series.zero_rate_to_date(date=dates,
                                                                                   to_date=maturity_date,
                                                                                   compounding=ts.base_rate_compounding,
                                                                                   frequency=ts.base_rate_frequency,
                                                                                   ),
                                                                 date=dates)
                                               )
            deposit_rate = DepositRate(deposit_rate)
            deposit_rates.add(deposit_rate)

        return YieldCurveTimeSeries(deposit_rates, calendar=self.calendar, day_counter=self.base_rate_day_counter)


class CurrencyCurve:

    def __init__(self, reference_date, spot, helpers, counter_rate_day_counter, counter_rate_compounding,
                 counter_rate_frequency, base_rate_day_counter, base_rate_compounding, base_rate_frequency):
        """ Currency curve class analogous to QuantLib yield term structures.

        Parameters
        ----------
        reference_date: QuantLib.Date
            The reference date for the curve.
        spot: scalar
            Value of the spot exchange rate at `reference_date`.
        helpers: list of :py:pbj:CurrencyFutureHelper
            Currency future helpers for construction of the curve.
        counter_rate_day_counter: QuantLib.DayCounter
            Day counter for the counter currency interest rates.
        counter_rate_compounding: QuantLib.Compounding
            Compounding convention for the counter currency interest rates.
        counter_rate_frequency: QuantLib.Frequency
            Compounding frequency for the counter currency interest rates.
        counter_rate_day_counter: QuantLib.DayCounter
            Day counter for the counter currency interest rates.
        counter_rate_compounding: QuantLib.Compounding
            Compounding convention for the counter currency interest rates.
        counter_rate_frequency: QuantLib.Frequency
            Compounding frequency for the counter currency interest rates.
        """
        # TODO: Propose a similar class in QuantLib.
        self.reference_date = to_ql_date(reference_date)
        self.spot = spot
        self.helpers = sorted(helpers, key=attrgetter('maturity'))
        self.maturity_dates = [self.reference_date] + [helper.maturity for helper in self.helpers]
        self.nodes = {helper.maturity: helper.quote for helper in self.helpers}
        self.nodes[reference_date] = spot
        self.max_date = to_ql_date(self.maturity_dates[-1])
        self._day_counter = ql.Actual365Fixed()
        self.counter_rate_day_counter = counter_rate_day_counter
        self.counter_rate_compounding = counter_rate_compounding
        self.counter_rate_frequency = counter_rate_frequency
        self.base_rate_day_counter = base_rate_day_counter
        self.base_rate_compounding = base_rate_compounding
        self.base_rate_frequency = base_rate_frequency

        # These will be initialized when necessary.
        self._base_rate_curve = None
        self._counter_rate_curve = None

    def dates(self):
        """ Date nodes of the currency curve.

        Returns
        -------
        list of QuantLib.Date
            Date nodes of the currency curve.
        """
        return self.maturity_dates

    def exchange_rate_to_date(self, date):
        """ Exchange rate to a given date, using linear interpolation between nodes.

        Parameters
        ----------
        date: QuantLib.Date

        Returns
        -------
        scalar
            The exchange rate implied by the currency curve, using linear interpolation between nodes.
        """
        date = to_ql_date(date)
        if date > self.max_date:
            raise ValueError("The requested date ({0}) is after the curve's max date ({1})".format(date,
                                                                                                   self.max_date))
        try:
            # If the date is in the nodes, use it.
            return self.nodes[date]
        except KeyError:
            # Use linear interpolation to obtain the result.
            lower_date_bound = find_le(self.maturity_dates, date)
            upper_date_bound = find_gt(self.maturity_dates, date)
            lower_bound = self.nodes[lower_date_bound]
            upper_bound = self.nodes[upper_date_bound]
            lower_interval = self._day_counter.yearFraction(lower_date_bound, date)
            interval = self._day_counter.yearFraction(lower_date_bound, upper_date_bound)
            return lower_bound + lower_interval * (upper_bound - lower_bound) / interval

    def spot_rate(self):
        """ Exchange rate at the reference date (spot rate).

        Returns
        -------
        scalar
            Spot rate.
        """
        return self.nodes[self.maturity_dates[0]]

    def counter_rate_curve(self, base_rate_curve=None):
        """ Yield term structure object of the counter currency.

        Parameters
        ----------
        base_rate_curve: QuantLib.YieldTermStructure

        Returns
        -------
        QuantLib.YieldTermStructure
            Yield curve of counter currency interest rates, implied by the currency curve and base currency interest
            rates curve.
        """
        if base_rate_curve is None:
            if self._counter_rate_curve is None:
                raise ValueError("CurrencyCurve needs a base_rate_curve to build the counter_rate_curve.")
            return self._counter_rate_curve
        # Build the curve if necessary.
        node_dates = self.maturity_dates
        discounts = [helper.counter_rate(helper.quote, self.spot,
                                         base_rate_curve.zeroRate(helper.maturity, helper.counter_rate_compounding,
                                                                  helper.counter_rate_frequency),
                                         self.reference_date).discountFactor(self.reference_date, helper.maturity)
                     for helper in self.helpers]
        self._counter_rate_curve = ql.DiscountCurve(node_dates, discounts, self.counter_rate_day_counter)
        return self._counter_rate_curve

    def base_rate_curve(self, counter_rate_curve=None):
        """ Yield term structure object of the base currency.

        Parameters
        ----------
        counter_rate_curve: QuantLib.YieldTermStructure

        Returns
        -------
        QuantLib.YieldTermStructure
            Yield curve of base currency interest rates, implied by the currency curve and counter currency interest
            rates curve.
        """
        if counter_rate_curve is None:
            if self._base_rate_curve is None:
                raise ValueError("CurrencyCurve needs a base_rate_curve to build the counter_rate_curve.")
            return self._base_rate_curve
        # Build the curve if necessary.
        node_dates = self.maturity_dates
        discounts = [helper.base_rate(helper.quote, self.spot,
                                      counter_rate_curve.zeroRate(helper.maturity, helper.counter_rate_compounding,
                                                                  helper.counter_rate_frequency),
                                      self.reference_date).discountFactor(self.reference_date, helper.maturity)
                     for helper in self.helpers]
        self._base_rate_curve = ql.DiscountCurve(node_dates, discounts, self.base_rate_day_counter)
        return self._base_rate_curve