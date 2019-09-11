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
CupomCambial class, USD interest rate in Brazil.
TODO: Propose implementation of this rate type in QuantLib.
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from tsfin.instruments.interest_rates.depositrate import DepositRate
from tsfin.base.qlconverters import to_ql_date


def next_cc_maturity(date):
    """Next DDI future maturity (DDI future contracts dealt by BMF in Brazil).

    Parameters
    ----------
    date: date-like

    Returns
    -------
    QuantLib.Date

    """
    date = to_ql_date(date)
    calendar = ql.Brazil()
    return calendar.advance(calendar.endOfMonth(calendar.advance(date, 2, ql.Days)), 1, ql.Days)


class CupomCambial(DepositRate):
    """Class to represent USD interest rate in Brazil.

    Note that this deposit rate is different from Libor and other typical interest rates, in that it has fixed
    maturity (rolling each month), instead of a fixed tenor. The rates calculated in the ``ts_values`` of this object
    represent the "cupom cambial" that is closest to maturity (i.e. the "next cupom cambial").

    Parameters
    ----------
    ts_name: str
        Name of the TimeSeries that will be built.
    currency_curve: :py:obj:`CurrencyCurveTimeSeries`
        USDBRL time series of currency curves, needed to obtain the interest rates.
    DI_curve: :py:obj:`YieldCurveTimeSeries`
        Brazilian interest rate time series of yield curves, needed to obtain the interest rates.
    """

    def __init__(self, ts_name, currency_curve, di_curve):
        # Create a BaseInstrument (base class of DepositRate) with the given ts_name.
        super(DepositRate, self).__init__(timeseries=ts_name)
        self.calendar = ql.Brazil()
        self.day_counter = ql.Actual360()
        self.compounding = ql.Simple
        self.frequency = ql.Annual
        self.business_convention = ql.Following
        self.fixing_days = 0
        spot = currency_curve.spot
        dates = spot.ts_values.index
        spot_values = spot.ts_values.values
        values = np.vectorize(self._calculate_cc, excluded=['currency_curve', 'DI_curve'])(dates, spot_values,
                                                                                           currency_curve, di_curve)
        self.ts_values = pd.Series(index=dates, data=values)

    def _calculate_cc(self, date, spot_price, currency_curve, di_curve):
        maturity_date = self._maturity_on_the_run(date)
        future_price = currency_curve.exchange_rate_to_date(date, maturity_date)
        di = di_curve.zero_rate_to_date(date, maturity_date, ql.Compounded, ql.Annual)
        di_rate = ql.InterestRate(di, ql.Business252(), ql.Compounded, ql.Annual)
        date = to_ql_date(date)
        compound = spot_price * di_rate.compoundFactor(date, maturity_date) / future_price
        rate = ql.InterestRate.impliedRate(compound, ql.Actual360(), ql.Simple, ql.Annual, date, maturity_date)
        return rate.rate()

    def maturity(self, date, *args, **kwargs):
        """Maturity of the "next cupom cambial".

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        QuantLib.Date
            Maturity date.
        """
        date = to_ql_date(date)
        calendar = self.calendar
        return calendar.advance(calendar.endOfMonth(calendar.advance(date, 2, ql.Days)), 1, ql.Days)

    def tenor(self, date, **kwargs):
        """Tenor of the "next cupom cambial".

        Parameters
        ----------
        date: QuantLib.Date
            Reference date.

        Returns
        -------
        QuantLib.Period
            The tenor (period) to maturity.


        """
        date = to_ql_date(date)
        maturity = self._maturity_on_the_run(date)
        days = self.calendar.businessDaysBetween(date, maturity)
        return ql.Period(days, ql.Days)

    def rate_helper(self, date, last_available=True, **other_args):
        """Helper for yield curve construction.

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
        rate = self.get_value(date=date, last_available=last_available, default=np.nan)
        tenor = self.tenor(date=date)
        return ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), tenor, 0, self.calendar, ql.Following, False,
                                    self.day_counter)
