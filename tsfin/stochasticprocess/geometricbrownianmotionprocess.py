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
A class for modelling geometric brownian motion processes.
"""
import QuantLib as ql
from tsfin.base import to_ql_date


class GeometricBrownianMotion:

    def __init__(self):
        self.process_name = "GBM"
        self.process = None
        self.mean = 0
        self.sigma = 0

    def update_process(self, initial_value, mean, sigma):

        self.mean = mean
        self.sigma = sigma
        initial_value = float(initial_value)
        self.process = ql.GeometricBrownianMotionProcess(initial_value, self.mean, self.sigma)
