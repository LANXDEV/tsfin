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
Currency class, to represent Currency Spot.
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from tsio.tools import at_index
from tsfin.constants import CALENDAR, CURRENCY, BASE_CURRENCY, COUNTRY, BASE_CALENDAR
from tsfin.base import Instrument, to_datetime, to_ql_date, to_ql_calendar, ql_holiday_list, conditional_vectorize, \
    to_list, filter_series


class Currency(Instrument):
    """ Model Currency Spot.

    :param timeseries: :py:class:`TimeSeries`
        The TimeSeries representing the Currency.
    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.currency = self.ts_attributes[CURRENCY]
        self.base_currency = self.ts_attributes[BASE_CURRENCY]
        self.calendar = to_ql_calendar(self.ts_attributes[CALENDAR])
        self.base_calendar = to_ql_calendar(self.ts_attributes[BASE_CALENDAR])
        self.country = self.ts_attributes[COUNTRY]

