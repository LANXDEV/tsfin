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
A trader model that trades any instrument with no spreads, costs, or fees.

WARNING: THIS CLASS IS NOT WORKING.

TODO: Finish this class and test it.
"""
from tsio import TimeSeries
from tsfin.instruments.bonds.callablefixedratebond import CallableFixedRateBond
from tsfin.instruments.bonds.fixedratebond import FixedRateBond


class PerfectTrader:
    def __init__(self, default_security_objects=None):
        if default_security_objects is None:
            self.default_security_objects = [None]
        else:
            self.default_security_objects = default_security_objects
        pass

    def trade(self, the_date, old_portfolio, objective_portfolio_list, security_objects=None):
        print('PerfectTrader: executing trades for ' + the_date.strftime('%Y-%m-%d'))
        if security_objects is None:
            security_objects = self.default_security_objects

        if the_date not in old_portfolio.positions.keys():
            old_portfolio.carry_to(the_date, security_objects=security_objects)

        # print('old portfolio: ' + str(the_date))
        # pprint(old_portfolio.positions[the_date])

        # print('These are the portfolio weights:')
        # _, positions_percent = old_portfolio.positions_percent(the_date=the_date, security_objects=security_objects)
        # pprint(positions_percent)
        # print('This is the sum of percentages: ' + str(round(sum(float(x.replace('%', '')) for x in
        #                                                          positions_percent.values()), 2)) + '%')

        old_securities = set([security_name for security_name in old_portfolio.positions[the_date].keys()])
        new_securities = set([x[0] for x in objective_portfolio_list])

        securities_to_get_rid_of = old_securities.difference(new_securities)
        # print('securities to get rid of:')
        # input(securities_to_get_rid_of)

        for security_name in securities_to_get_rid_of:
            # Doing this first because security_objects can be different in Trader Class and Portfolio Class
            price = self.get_price(the_date, security_name, security_objects)
            the_trade = self.trade(-old_portfolio.positions[the_date][security_name], price)
            old_portfolio.add_trade(the_date, security_name, the_trade)

        portfolio_value, _ = old_portfolio.valuate(the_date, security_objects)
        # print('This is the portfolio value')
        # pprint(portfolio_value)
        # input(' press enter')

        for trade_item in objective_portfolio_list:
            new_security_name = trade_item[0]
            new_security_nmv = trade_item[1] * portfolio_value
            new_security_price = self.get_price(the_date, new_security_name, security_objects)
            new_security_qty = new_security_nmv / new_security_price
            new_trade = self.trade(new_security_qty - old_portfolio.positions[the_date].get(new_security_name, 0),
                              new_security_price)
            old_portfolio.add_trade(the_date, new_security_name, new_trade)

            # pprint('the trades: ' + str(the_date))
            # pprint(old_portfolio.trades[the_date])
            #
            # print('new portfolio: ' + str(the_date))
            # pprint(old_portfolio.positions[the_date])
            # input()

    def get_price(self, the_date, security_name, security_objects=None):
        if security_objects is None:
            security_objects = self.default_security_objects
        security = next((obj for obj in security_objects if security_name == getattr(obj, 'ts_name', None) or
                            security_name == getattr(obj, 'name', None)), [None])

        if isinstance(security, FixedRateBond) or isinstance(security, CallableFixedRateBond):
            clean_price = security.clean_prices.get_value(date=the_date)
            price = security.dirty_price_from_clean_price(clean_price, the_date) / security.attributes['PAR']
            # TODO: Add a convenient attribute to Bond objects (e.g.: price-to-value factor)
        elif isinstance(security, TimeSeries):
            price = security.get_value(date=the_date)
        else:
            price = 0.0
        return price







