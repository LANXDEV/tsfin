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
from bisect import bisect_left, bisect_right
import numpy as np
import pandas as pd
import collections
from tsio.tools import at_index
from tsfin.base.qlconverters import to_ql_date, to_ql_duration

trade = collections.namedtuple('trade', 'qty price')


def merge_trades(old, new):
    merged_trade = trade(old.qty + new.qty, (old.qty * old.price + new.qty * new.price)/(old.qty + new.qty))
    return merged_trade


def find_gt(date, sorted_dates):
    # Find leftmost value greater than x
    i = bisect_right(sorted_dates, date)
    if i != len(sorted_dates):
        return sorted_dates[i]
    raise ValueError('Could not find leftmost value greater than date.')


def find_lt(date, sorted_dates):
    # Find rightmost value less than x
    i = bisect_left(sorted_dates, date)
    if i:
        # print('returning {}'.format(sorted_dates[i-1]))
        return sorted_dates[i - 1]
    raise ValueError('Could not find rightmost value less than date.')


class Portfolio:
    """Model of a Simple Portfolio of Securities
    Attributes
    ----------
    portfolio : DataFrame
        DataFrame with columns = ['DATE', 'TS_NAME', 'NMV'].
        TS_NAME : Name of the TimeSeries.
        NMV : Total value of the TimeSeries at Date.

    """

    def __init__(self, currency, security_objects=None, base_yield_curve=None):
        self.positions = dict()
        self.trades = dict()
        self.currency = currency
        self.base_yield_curve = base_yield_curve
        if security_objects is None:
            self.security_objects = []
        else:
            self.security_objects = security_objects

    def copy(self):
        copied_portfolio = Portfolio(self.currency, self.security_objects)
        copied_portfolio.positions = self.positions.copy()
        copied_portfolio.trades = self.trades.copy()
        return copied_portfolio

    def add_position(self, date, name, qty):
        if date in self.positions.keys():
            self.positions[date][name] = self.positions[date].get(name, 0) + qty
        else:
            self.positions[date] = {name: qty}

    def remove_position(self, date, name, qty=None):
        date = pd.to_datetime(date)
        if date in self.positions.keys():
            if qty is None:
                self.positions[date].pop(date, None)
            else:
                self.positions[date][name] = self.positions[date].get(name, 0) - qty

    def get_security(self, security_name, security_objects=None):
        if security_objects is None:
            security_objects = self.security_objects
        security = next((obj for obj in security_objects if
                         security_name == getattr(obj, 'ts_name', None) or
                         security_name == getattr(obj, 'name', None)), [None])
        return security

    def carry_to(self, date, security_objects=None):
        if security_objects is None:
            security_objects = self.security_objects
        if date not in self.positions.keys():
            sorted_dates = sorted(list(self.positions.keys()))
            previous_date = find_lt(date, sorted_dates)
            for security_name in self.positions[previous_date].keys():
                # print('carrying ' + security_name)
                security = self.get_security(security_name, security_objects)
                self.add_position(date, self.currency, security.cash_to_date(start_date=previous_date,
                                                                             date=date))
                if not security.is_expired(date):
                    self.add_position(date, security_name, self.positions[previous_date][security_name])

    def add_trade(self, date, name, new_trade, security_objects=None):
        date = pd.to_datetime(date)
        if date in self.trades.keys():
            self.trades[date][name] = merge_trades(self.trades[date].get(name, trade(0, 0)), new_trade)
        else:
            self.trades[date] = {name: new_trade}
        # Now apply the new trade to the positions dict
        self._apply_trade(date, name, new_trade, security_objects)

    def _apply_trade(self, date, name, new_trade, security_objects):
        if date not in self.positions.keys():
            self.carry_to(date, security_objects)
        self.positions[date][name] = self.positions[date].get(name, 0) + new_trade.qty
        self.positions[date][self.currency] = self.positions[date].get(self.currency, 0) - \
            new_trade.qty * new_trade.price
        if self.positions[date][name] == 0 and name != self.currency:
            del self.positions[date][name]

    def value(self, date, security_objects=None, **kwargs):
        # print('Portfolio: valuating in ' + the_date.strftime('%Y-%m-%d'))
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        value_dict = dict()
        for security_name in self.positions[date].keys():
            if security_name == self.currency:
                unit_value = 1
            else:
                unit_value = self.get_security(security_name, security_objects).value(date=date, last_available=True)
                if np.isnan(unit_value):
                    print("Security {0} is returning null value in {1}, replacing by zero..".format(security_name,
                                                                                                    date))
                    unit_value = 0
            value_dict[security_name] = unit_value * self.positions[date][security_name]
        value = sum(value_dict.values())
        return value, value_dict

    def ytm(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).ytm(date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null ytm in {1}, replacing by zero..".format(security_name,
                                                                                                  date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result
        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def ytw(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).ytw(date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null ytw in {1}, replacing by zero..".format(security_name,
                                                                                                  date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def ytw_rolling_call(self, date, security_objects=None, **kwargs):
        pass

    def zspread_to_mat(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        yield_curve_timeseries = kwargs['yield_curve_timeseries']
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name,
                                                security_objects).zspread_to_mat(date=date,
                                                                        yield_curve_timeseries=yield_curve_timeseries)
                if np.isnan(unit_result):
                    print("Security {0} is returning null zspread_to_mat {1}, replacing by zero..".format(
                        security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result
        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def zspread_to_worst(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        yield_curve_timeseries = kwargs['yield_curve_timeseries']
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name,
                                                security_objects).zspread_to_worst(date=date,
                                                                        yield_curve_timeseries=yield_curve_timeseries)
                if np.isnan(unit_result):
                    print("Security {0} is returning null zspread_to_worst {1}, replacing by zero..".format(
                        security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def zspread_to_worst_rolling_call(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        yield_curve_timeseries = kwargs['yield_curve_timeseries']
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name,
                                                security_objects).zspread_to_worst_rolling_call(date=date,
                                                                        yield_curve_timeseries=yield_curve_timeseries)
                if np.isnan(unit_result):
                    print("Security {0} is returning null zspread_to_worst_rolling_call {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mac_duration_to_mat(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_mat(
                    duration_type=to_ql_duration('Macaulay'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_mat {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mac_duration_to_worst(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_worst(
                    duration_type=to_ql_duration('Macaulay'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_worst {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mac_duration_to_worst_rolling_call(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_worst_rolling_call(
                    duration_type=to_ql_duration('Macaulay'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_worst_rolling_call {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mod_duration_to_mat(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_mat(
                    duration_type=to_ql_duration('Modified'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_mat {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mod_duration_to_worst(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_worst(
                    duration_type=to_ql_duration('Modified'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_worst {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def mod_duration_to_worst_rolling_call(self, date, security_objects=None, **kwargs):
        if security_objects is None:
            security_objects = self.security_objects
        self.carry_to(date, security_objects)
        result_dict = dict()
        value, value_dict = self.value(date, security_objects)
        for security_name in self.positions[date].keys():
            try:
                unit_result = self.get_security(security_name, security_objects).duration_to_worst_rolling_call(
                    duration_type=to_ql_duration('Modified'), date=date)
                if np.isnan(unit_result):
                    print("Security {0} is returning null mod_duration_to_worst_rolling_call {1}, replacing by "
                          "zero..".format(security_name, date))
                    unit_result = 0
            except AttributeError:
                unit_result = 0
            result_dict[security_name] = unit_result

        result = sum(value_dict[name] * result_dict[name] / value for name in self.positions[date])
        return result, result_dict

    def convexity_to_mat(self, date, security_objects=None, **kwargs):
        pass

    def convexity_to_worst(self, date, security_objects=None, **kwargs):
        pass

    def convexity_to_worst_rolling_call(self, date, security_objects=None, **kwargs):
        pass

    def percentage_of_amount_outstanding(self, date, security_objects=None, **kwargs):
        pass

    def liquidity_index(self, date, security_objects=None):
        # total_liquidity_index = total_amt_outstd * number_of_securities.
        pass
