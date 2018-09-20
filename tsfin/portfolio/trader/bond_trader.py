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
A trader model that trades instruments following a pre-defined cost mapping.

WARNING: THIS CLASS IS NOT WORKING.

TODO: Finish this class and test it.
"""
import pandas as pd
import scipy.optimize as optimize
from tsio import TimeSeries
from tsfin.instruments.bonds.callablefixedratebond import CallableFixedRateBond
from tsfin.instruments.bonds.fixedratebond import FixedRateBond


class CostTrader:

    def __init__(self, cost_dict, default_security_objects=None):
        if default_security_objects is None:
            self.default_security_objects = [None]
        else:
            self.default_security_objects = default_security_objects
        self.cost_dict = cost_dict

    def trade(self, the_date, old_portfolio, objective_portfolio_dict, security_objects=None):
        if objective_portfolio_dict == 'no_trade':
            # If the optimizer sent a no-trade signal, don't do anything.
            print('Cost Trader received a no-trade signal, passing...')
            return
        print('CostTrader: executing trades for ' + the_date.strftime('%Y-%m-%d'))
        if security_objects is None:
            security_objects = self.default_security_objects

        self.sell_unwanted_securities(the_date, objective_portfolio_dict, old_portfolio, security_objects)
        # Selling securities that should not be in the portfolio anymore

        final_nav_guess, _ = old_portfolio.valuate(the_date, security_objects, pricing_function=self.get_price)
        final_nav = optimize.newton(self._objective_function, final_nav_guess, args=(the_date, objective_portfolio_dict,
                                                                                     old_portfolio, security_objects))
        # Discovering what will be the final NAV after the trades
        self._trade_for_final_nav(final_nav, the_date, objective_portfolio_dict, old_portfolio, security_objects)

    def _trade_for_final_nav(self, final_nav, the_date, objective_portfolio_dict, old_portfolio, security_objects):
        for trade_name, trade_percent in objective_portfolio_dict.items():
            new_security_name = trade_name
            new_security_nmv = trade_percent * final_nav
            new_security_qty = new_security_nmv / self.get_price(the_date, new_security_name,
                                                                 security_objects=security_objects)
            # This will be the new security quantity
            old_security_qty = old_portfolio.positions[the_date].get(new_security_name, 0)
            if old_security_qty > new_security_qty:
                new_security_trading_price = self.get_price(the_date, new_security_name, self.cost_dict, 'SELL',
                                                            security_objects)
            else:
                new_security_trading_price = self.get_price(the_date, new_security_name, self.cost_dict, 'BUY',
                                                            security_objects)
            new_trade = self.trade(new_security_qty - old_security_qty, new_security_trading_price)
            old_portfolio.add_trade(the_date, new_security_name, new_trade)

    def _objective_function(self, final_nav, the_date, objective_portfolio_dict, old_portfolio, security_objects):
        old_portfolio_copy = old_portfolio.copy()
        self._trade_for_final_nav(final_nav, the_date, objective_portfolio_dict, old_portfolio_copy, security_objects)
        the_actual_final_nav, _ = old_portfolio_copy.valuate(the_date, security_objects,
                                                             pricing_function=self.get_price)
        return final_nav - the_actual_final_nav

    def sell_unwanted_securities(self, the_date, objective_portfolio_dict, old_portfolio, security_objects):
        old_securities = set([security_name for security_name in old_portfolio.positions[the_date].keys()])
        new_securities = set([x for x in objective_portfolio_dict])
        securities_to_get_rid_of = old_securities.difference(new_securities)
        for security_name in securities_to_get_rid_of:
            # Getting rid of the securities that should not be in the portfolio anymore.
            price = self.get_price(the_date, security_name, self.cost_dict, 'SELL', security_objects)
            the_trade = self.trade(-old_portfolio.positions[the_date][security_name], price)
            old_portfolio.add_trade(the_date, security_name, the_trade)

    def get_price(self, the_date, security_name, cost_dict=None, transaction_type=None, security_objects=None):
        if cost_dict is None:
            cost_dict = self.cost_dict
        if security_objects is None:
            security_objects = self.default_security_objects
        security = next((obj for obj in security_objects if security_name == getattr(obj, 'ts_name', None) or
                         security_name == getattr(obj, 'name', None)), [None])

        if isinstance(security, FixedRateBond) or isinstance(security, CallableFixedRateBond):
            clean_price = security.clean_prices.get_value(date=the_date)
            par = security.attributes['PAR']
            price = security.dirty_price_from_clean_price(clean_price, the_date) / par
            # TODO: Add a convenient attribute to Bond objects (e.g.: price-to-value factor)
            consider_defaulted = False
            try:  # Verifying if bond has defaulted
                default_date = pd.to_datetime(security.get_attribute('DEFAULT_DATE'))
                consider_defaulted = (default_date <= the_date)
            except:
                pass
            if consider_defaulted is True:
                price = price * cost_dict.get('DEFAULT', 0.0)
                # input('this bond has defaulted: ' + str(security_name) + ' in ' + str(the_date))
            elif transaction_type == 'BUY':
                trade_spread = cost_dict.get('BUY', 0.0)
                price = price + (trade_spread / par)
            elif transaction_type == 'SELL':
                trade_spread = cost_dict.get('SELL', 0.0)
                price = price - (trade_spread / par)

        elif isinstance(security, TimeSeries):
            price = security.get_value(date=the_date)
        else:
            price = 0.0
        return price

