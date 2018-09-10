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
        security_return_list = list(reversed(sorted(security_return_list, key=itemgetter(1))))[0:self.number_of_securities]

        optimized_portfolio = [(x[0], 1/self.number_of_securities, x[1]) for x in security_return_list]

        return optimized_portfolio
