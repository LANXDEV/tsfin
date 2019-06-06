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
from functools import wraps
import numpy as np
import QuantLib as ql
from tsfin.instruments.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_calendar, to_ql_day_counter, to_ql_rate_index, to_ql_business_convention
from tsfin.constants import CALENDAR, INDEX, DAY_COUNTER, TENOR_PERIOD, BUSINESS_CONVENTION, INDEX_TENOR, QUOTE_TYPE


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
        self.business_convention = to_ql_business_convention(self.ts_attributes[BUSINESS_CONVENTION])
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        self._index_tenor = ql.PeriodParser.parse(self.ts_attributes[INDEX_TENOR])
        self.index = to_ql_rate_index(self.ts_attributes[INDEX], self._index_tenor)
        self.day_counter = to_ql_day_counter(self.ts_attributes[DAY_COUNTER])

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

        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)

        if np.isnan(rate):
            return None
        return ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), self._tenor, self.calendar, self.frequency,
                                 self.business_convention, self.day_counter, self.index)
