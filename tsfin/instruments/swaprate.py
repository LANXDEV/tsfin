"""
A class for modelling interest rate swaps.
"""
from functools import wraps
import numpy as np
import QuantLib as ql
from tsfin.instruments.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_calendar, to_ql_day_counter, to_ql_index, to_ql_business_convention
from tsfin.constants import CALENDAR, INDEX, DAY_COUNTER, TENOR_PERIOD, BUSINESS_CONVENTION


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
        self.business_convention = to_ql_business_convention(self.attributes[BUSINESS_CONVENTION])
        self.calendar = to_ql_calendar(self.attributes[CALENDAR])
        self._tenor = ql.PeriodParser.parse(self.attributes[TENOR_PERIOD])
        self.index = to_ql_index(self.attributes[INDEX])(ql.Period(3, ql.Months))  # TODO: needs to be parametrized.
        self.day_counter = to_ql_day_counter(self.attributes[DAY_COUNTER])

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
        rate = self.get_value(date=date, last_available=last_available, default=np.nan)
        if np.isnan(rate):
            return None
        return ql.SwapRateHelper(rate, self._tenor, self.calendar, self.frequency, self.business_convention,
                                 self.day_counter, self.index)