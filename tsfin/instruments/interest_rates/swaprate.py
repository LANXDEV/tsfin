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
A class for modelling interest rate swaps.
"""
import numpy as np
import QuantLib as ql
from tsfin.instruments.interest_rates.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_calendar, to_ql_day_counter, to_ql_business_convention
from tsfin.constants import CALENDAR, DAY_COUNTER, BUSINESS_CONVENTION, SETTLEMENT_DAYS, \
    FIXED_LEG_TENOR


class SwapRate(DepositRate):
    """ Model for rolling interest rate swap rates (fixed tenor, like the ones quoted in Bloomberg).

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        # Database Attributes
        self.fixed_business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.fixed_calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.fixed_leg_tenor = ql.PeriodParser.parse(self.ts_attributes[FIXED_LEG_TENOR])
        self.fixed_leg_day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])
        # QuantLib Attributes
        self.index_calendar = self.index.fixingCalendar()
        self.calendar = ql.JointCalendar(self.fixed_calendar, self.index_calendar)
        # Swap Index
        self.swap_index = ql.SwapIndex(self.ts_name, self._tenor, self.settlement_days, self.currency, self.calendar,
                                       self.fixed_leg_tenor, self.fixed_business_convention, self.fixed_leg_day_counter,
                                       self.index)

    def set_rate_helper(self):
        """Set Rate Helper if None has been defined yet

        Returns
        -------
        QuantLib.RateHelper
        """
        self._rate_helper = ql.SwapRateHelper(ql.QuoteHandle(self.final_rate), self.swap_index,
                                              ql.QuoteHandle(self.final_spread))

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

    def rate_helper(self, date, last_available=True, *args, **kwargs):
        """ Helper for yield curve construction.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool, optional
            Whether to use last available quotes if missing data.

        Returns
        -------
        QuantLib.RateHelper
            Rate helper for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series

        if self._rate_helper is None:
            self.set_rate_helper()

        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)

        if np.isnan(rate):
            return None
        self.final_rate.setValue(rate)
        return self._rate_helper
