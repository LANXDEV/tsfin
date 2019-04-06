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
A portfolio optimizer model that selects instruments that have had the best returns in the past.

WARNING: THIS CLASS IS NOT WORKING.

TODO: Finish this class and test it.
"""
from operator import itemgetter


class BestReturnPortfolioOptimizer(object):

    def __init__(self, number_of_securities, return_calculator):
        self.number_of_securities = number_of_securities
        self.return_calculator = return_calculator

    def optimize(self, ts_collection, the_date, time_horizon):
        security_return_list = [(ts.ts_name, self.return_calculator.calculate_return(ts, the_date, time_horizon))
                                for ts in ts_collection]
        security_return_list = list(reversed(sorted(security_return_list,
                                                    key=itemgetter(1))))[0:self.number_of_securities]

        optimized_portfolio = [(x[0], 1/self.number_of_securities, x[1]) for x in security_return_list]

        return optimized_portfolio
