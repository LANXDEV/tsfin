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
A class for modelling interest rate swaption.
"""
import numpy as np
import QuantLib as ql
from tsfin.instruments import SwapRate
from tsfin.constants import MATURITY_TENOR


class Swaption(SwapRate):

    def __init__(self, timeseries):
        super().__init__(timeseries)
        # Database Attributes
        self.maturity_tenor = ql.PeriodParser.parse(self.ts_attributes[MATURITY_TENOR])

    def set_rate_helper(self):
        """Set Rate Helper if None has been defined yet

        Returns
        -------
        QuantLib.RateHelper
        """
        self._rate_helper = ql.SwaptionHelper(self.maturity_tenor, self._tenor, ql.QuoteHandle(self.final_rate),
                                              self.index, self.fixed_leg_tenor, self.fixed_leg_day_counter,
                                              self.index.dayCounter(), self.term_structure)

    def is_expired(self, date, *args, **kwargs):
        """ Returns False.

        Parameters
        ----------
        date: QuantLib.Date
            The date.

        Returns
        -------
        bool
            Always False.
        """
        return False
