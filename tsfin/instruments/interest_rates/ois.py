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
A class for modelling OIS (Overnight Indexed Swap) rates.
"""
import numpy as np
import QuantLib as ql
from tsfin.instruments.interest_rates.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_date, to_ql_rate_index, to_ql_calendar
from tsfin.constants import INDEX, TENOR_PERIOD, SETTLEMENT_DAYS, PAYMENT_LAG, CALENDAR


class OISRate(DepositRate):
    """ Class to model OIS (Overnight Indexed Swap) rates.

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`
        TimeSeries object representing the instrument.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.overnight_index = to_ql_rate_index(self.ts_attributes[INDEX])
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self._tenor = ql.PeriodParser.parse(self.ts_attributes[TENOR_PERIOD])
        self.settlement_days = int(self.ts_attributes[SETTLEMENT_DAYS])
        self.payment_lag = int(self.ts_attributes[PAYMENT_LAG])

    def rate_helper(self, date, last_available=True, spread=0, *args, **kwargs):
        """ Rate helper object for yield curve building.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool
            Whether to use last available information if missing data.
        spread: float
            The spread to be added to the OIS rate. Default 0.
        Returns
        -------
        QuantLib.RateHelper
            Rate helper object for yield curve construction.
        """
        # Returns None if impossible to obtain a rate helper from this time series
        if self.is_expired(date):
            return None
        rate = self.get_values(index=date, last_available=last_available, fill_value=np.nan)
        if np.isnan(rate):
            return None
        date = to_ql_date(date)
        try:
            tenor = self.tenor(date)
        except ValueError:
            # Return none if the deposit rate can't retrieve a tenor (i.e. is expired).
            return None
        final_rate = ql.SimpleQuote(rate)
        return ql.OISRateHelper(self.settlement_days, tenor, ql.QuoteHandle(final_rate),
                                self.overnight_index, ql.YieldTermStructureHandle(), True, 0,
                                self.business_convention, self.frequency, self.calendar, ql.Period(0, ql.Days),
                                float(spread))
