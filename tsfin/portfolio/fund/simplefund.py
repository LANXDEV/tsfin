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
Model for a fund, which may have portfolios, portfolio optimizers, and traders.
"""
import pandas as pd


class SimpleFund(object):

    def __init__(self, initial_portfolio, return_calculator, portfolio_optimizer, trader, rebalance_freq,
                 security_space):
        self.portfolio = initial_portfolio
        self.return_calculator = return_calculator
        self.portfolio_optimizer = portfolio_optimizer
        self.trader = trader
        self.rebalance_freq = rebalance_freq
        self.security_space = security_space

    def simulate(self, initial_date, final_date):
        initial_date = pd.to_datetime(initial_date)
        final_date = pd.to_datetime(final_date)
        trade_dates = list(pd.date_range(initial_date, final_date, freq=self.rebalance_freq))

        for the_date in trade_dates:
            if the_date not in self.portfolio.positions.keys():
                # First we carry the portfolio to the desired date
                self.portfolio.carry_to(the_date, security_objects=self.security_space)
            desired_portfolio = self.portfolio_optimizer.optimize(self.security_space, the_date,
                                                                  self.portfolio)
            self.trader.trade(the_date, self.portfolio, desired_portfolio, self.security_space)

    def performance(self, date_list=None):
        inception_date = min(self.portfolio.positions.keys())
        inception_value, _ = self.portfolio.valuate(inception_date, self.security_space)

        if date_list is None:
            date_list = self.portfolio.positions.keys()

        perf_dict = {the_date: self.portfolio.valuate(the_date, self.security_space)[0] / inception_value - 1 for
                     the_date in date_list}
        return perf_dict

    def duration(self, date_list=None):
        inception_date = min(self.portfolio.positions.keys())
        inception_value, _ = self.portfolio.valuate(inception_date, self.security_space)

        if date_list is None:
            date_list = self.portfolio.positions.keys()

        duration_dict = {the_date: self.portfolio.duration(the_date, self.security_space)[0] for the_date in date_list}

        return duration_dict

    def yield_to_worst(self, date_list=None):
        inception_date = min(self.portfolio.positions.keys())
        inception_value, _ = self.portfolio.valuate(inception_date, self.security_space)

        if date_list is None:
            date_list = self.portfolio.positions.keys()

        yield_to_worst_dict = {the_date: self.portfolio.yield_to_worst(the_date, self.security_space)[0] for the_date
                               in date_list}

        return yield_to_worst_dict
