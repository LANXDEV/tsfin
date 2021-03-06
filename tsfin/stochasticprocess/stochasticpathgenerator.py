# Copyright (C) 2016-2019 Lanx Capital Investimentos LTDA.
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
A class for generating paths from one or multiple stochastic process.
"""
import QuantLib as ql
import numpy as np
from tsfin.base import to_list, to_ql_date, to_ql_calendar, to_ql_day_counter, to_ql_business_convention


# class for hosting schedule-related information (dates, times)
class TimeGrid:

    def __init__(self, start_date, end_date, tenor, calendar, day_counter, business_convention):
        self.start_date = to_ql_date(start_date)
        self.end_date = to_ql_date(end_date)
        self.tenor = ql.PeriodParser.parse(tenor)
        self.calendar = to_ql_calendar(calendar)
        self.day_counter = to_ql_day_counter(day_counter)
        self.business_convention = to_ql_business_convention(business_convention)
        self.schedule = ql.Schedule(self.start_date, self.end_date, self.tenor, self.calendar, self.business_convention,
                                    self.business_convention, ql.DateGeneration.Forward, False)

    def get_times(self):
        # get list of scheduled times
        return [self.day_counter.yearFraction(self.schedule.startDate(), date) for date in self.schedule]

    def get_dates(self):
        # get list of scheduled dates
        return [date for date in self.schedule]

    def get_maturity(self):
        # get maturity in time units
        return self.day_counter.yearFraction(self.schedule.startDate(), self.schedule.endDate())

    def get_steps(self):
        # get number of steps in schedule
        return self.get_size() - 1

    def get_size(self):
        # get total number of items in schedule
        return len(self.schedule)

    def get_time_grid(self):
        # get QuantLib TimeGrid object, constructed by using list of scheduled times
        return ql.TimeGrid(self.get_times(), self.get_size())

    def get_dt(self):
        # get constant time step
        return self.get_maturity() / self.get_steps()


class PathGenerator:

    def __init__(self, process_list, correlation_matrix, start_date, end_date, tenor='1D', calendar='NULL',
                 day_counter='ACTUAL365', business_convention='UNADJUSTED', low_discrepancy=False,
                 brownian_bridge=True, seed=0):
        """

        :param process_list: list of QuantLib.StochasticProcess1D
            A list with the QuantLib class of the stochastic process
        :param correlation_matrix: list
            A list with the correlation matrix.
        :param start_date: QuantLib.Date
            The start date of the path
        :param end_date: QuantLib.Date
            The end date of the path
        :param tenor: str
            The fixed interval used for path generation. Default '1D', daily intervals.
        :param calendar: str
            The calendar code of the calendar to be used. Default 'NULL'.
        :param day_counter: str
            The day count of the path. Default 'ACTUAL365'
        :param low_discrepancy: bool
            Used to choose the type of Gaussian Sequence Generator.
        :param seed: int
            Used to fix a certain seed of the random number.
        """
        self.process_list = to_list(process_list)
        self.correlation_matrix = correlation_matrix
        self.start_date = to_ql_date(start_date)
        self.end_date = to_ql_date(end_date)
        self.tenor = tenor.upper()
        self.calendar = calendar.upper()
        self.day_counter = day_counter.upper()
        self.business_convention = business_convention.upper()
        self.time_grid = TimeGrid(self.start_date, self.end_date, self.tenor, self.calendar, self.day_counter,
                                  self.business_convention)
        self.low_discrepancy = low_discrepancy
        self.brownian_bridge = brownian_bridge
        self.seed = seed
        if len(self.process_list) == 1:
            self.process = process_list[0]
        elif len(self.process_list) > 1:
            self.process = ql.StochasticProcessArray(self.process_list, self.correlation_matrix)
        else:
            raise print("No process")

    def initial_values(self):

        return self.process.initialValues()

    def _generator(self):

        dimensionality = self.time_grid.get_steps() * self.process.factors()
        if self.low_discrepancy:
            uniform_sequence = ql.UniformLowDiscrepancySequenceGenerator(dimensionality)
            generator = ql.GaussianLowDiscrepancySequenceGenerator(uniform_sequence)
        else:
            uniform_sequence = ql.UniformRandomSequenceGenerator(dimensionality, ql.UniformRandomGenerator(self.seed))
            generator = ql.GaussianRandomSequenceGenerator(uniform_sequence)
        return generator

    def generate_paths(self, n, antihetic=False):

        if isinstance(self.process, ql.StochasticProcessArray) or isinstance(self.process, ql.HestonProcess):
            path_generator = ql.GaussianMultiPathGenerator(self.process, self.time_grid.get_times(), self._generator(),
                                                           self.brownian_bridge)

            antihetic_const = 1
            if antihetic:
                antihetic_const = 2

            paths = np.zeros(shape=(n*antihetic_const, self.process.size(), self.time_grid.get_size()))
            # loop through number of paths
            k = 0
            for i in range(n):
                # request multiPath, which contains the list of paths for each process
                multi_path = path_generator.next().value()
                if antihetic:
                    multi_antithetic = path_generator.antithetic().value()
                # loop through number of processes
                for j in range(multi_path.assetNumber()):
                    # request path, which contains the list of simulated prices for a process
                    path = np.array(multi_path[j])
                    paths[k, j, :] = np.array(path)
                    if antihetic:
                        antithetic = np.array(multi_antithetic[j])
                        paths[k + 1, j, :] = np.array(antithetic)
                    # push prices to array
                    if j == multi_path.assetNumber() - 1:
                        k = k + antihetic_const
        else:
            if self.low_discrepancy:
                path_generator = ql.GaussianSobolPathGenerator(self.process, self.time_grid.get_time_grid(),
                                                               self._generator(), self.brownian_bridge)
            else:
                path_generator = ql.GaussianPathGenerator(self.process, self.time_grid.get_time_grid(),
                                                          self._generator(),
                                                          self.brownian_bridge)
            paths = np.zeros(shape=(2 * n, self.time_grid.get_size()))
            k = 0
            for i in range(n):
                path = path_generator.next().value()
                antithetic_path = path_generator.antithetic().value()
                paths[k, :] = path
                paths[k + 1, :] = antithetic_path
                k = k + 2
        # resulting array dimension: n, len(timeGrid)
        return paths
