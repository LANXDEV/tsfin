"""
A class for modelling OIS (Overnight Indexed Swap) rates.
"""
import numpy as np
import QuantLib as ql
from tsfin.instruments.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_date, to_ql_overnight_index
from tsfin.constants import INDEX, TENOR_PERIOD, SETTLEMENT_DAYS, PAYMENT_LAG


class OISRate(DepositRate):
    """ Class to model OIS (Overnight Indexed Swap) rates.

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`
        TimeSeries object representing the instrument.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.overnight_index = to_ql_overnight_index(self.attributes[INDEX])
        self._tenor = ql.PeriodParser.parse(self.attributes[TENOR_PERIOD])
        self.settlement_days = int(self.attributes[SETTLEMENT_DAYS])
        self.payment_lag = int(self.attributes[PAYMENT_LAG])

    def rate_helper(self, date, last_available=True, *args, **kwargs):
        """ Rate helper object for yield curve building.

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.
        last_available: bool
            Whether to use last available information if missing data.

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
        # Convert rate to simple compounding because DepositRateHelper expects simple rates.
        return ql.OISRateHelper(self.settlement_days, tenor, ql.QuoteHandle(ql.SimpleQuote(rate)),
                                self.overnight_index(), ql.YieldTermStructureHandle(), False, 0,
                                ql.ModifiedFollowing)
