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
from tsfin.base.qlconverters import to_ql_date, to_ql_frequency, to_ql_date_generation
from tsfin.constants import FREQUENCY, QUOTE_TYPE, DATE_GENERATION, RECOVERY_RATE, COUPONS
from tsfin.base.basetools import conditional_vectorize


class CDSRate(DepositRate):
    """ Model for rolling interest cds rates (fixed tenor, like the ones quoted in Bloomberg).

    Parameters
    ----------
    timeseries: :py:class:`TimeSeries`

    Note
    ----
    See the :py:mod:`constants` for required attributes in `timeseries` and their possible values.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.date_generation = to_ql_date_generation(self.ts_attributes[DATE_GENERATION])
        self.coupon_frequency = ql.Period(to_ql_frequency(self.ts_attributes[FREQUENCY]))
        self.recovery_rate = float(self.ts_attributes[RECOVERY_RATE])
        self.coupon = float(self.ts_attributes[COUPONS])

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

        quote_type_dict = {'BPS': float(10000), 'PERCENT': float(100)}
        quote_type = self.get_attribute(QUOTE_TYPE)
        rate /= float(quote_type_dict.get(quote_type, 100))

        if np.isnan(rate):
            return None
        return ql.SpreadCdsHelper(ql.QuoteHandle(ql.SimpleQuote(rate)), self._tenor, 0, self.calendar, self.frequency,
                                  self.business_convention, self.date_generation, self.day_counter, self.recovery_rate,
                                  *args)

    @conditional_vectorize('date')
    def credit_default_swap(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            The CDS Instrument for calculation purposes.
        """

        recovery_rate = self.recovery_rate
        if 'recovery_rate' in kwargs.keys():
            recovery_rate = kwargs['recovery_rate']

        # maybe the user passed a quote multiplied by 100, if so we divide by 100 to be correctly used by the CDS.
        if upfront_price > 1:
            upfront_price = upfront_price/100
        upfront = 1 - upfront_price

        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)
        quote_type_dict = {'BPS': float(10000), 'PERCENT': float(100)}
        quote_type = self.get_attribute(QUOTE_TYPE)
        rate /= float(quote_type_dict.get(quote_type, 100))

        maturity = to_ql_date(date) + self._tenor
        schedule = ql.Schedule(to_ql_date(date), maturity, self.coupon_frequency, self.calendar,
                               self.business_convention, ql.Unadjusted, self.date_generation, False)

        cds = ql.CreditDefaultSwap(ql.Protection.Buyer, notional, upfront, rate, schedule, self.business_convention,
                                   self.day_counter)
        probability = ql.DefaultProbabilityTermStructureHandle(hazard_curve)
        engine = ql.MidPointCdsEngine(probability, recovery_rate, *args)
        cds.setPricingEngine(engine)

        return cds

    @conditional_vectorize('date')
    def net_present_value(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            CDS Net Present Value.
        """

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.credit_default_swap(date, notional, hazard_curve, upfront_price, last_available, *args, **kwargs)

        return cds.NPV()

    @conditional_vectorize('date')
    def cds_spread(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            CDS fair Spread.
        """

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.credit_default_swap(date, notional, hazard_curve, upfront_price, last_available, *args, **kwargs)

        return cds.fairSpread()

    @conditional_vectorize('date')
    def default_leg_npv(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            CDS Net Present Value of the default Leg.
        """

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.credit_default_swap(date, notional, hazard_curve, upfront_price, last_available, *args, **kwargs)

        return cds.defaultLegNPV()

    @conditional_vectorize('date')
    def coupon_leg_npv(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            CDS Net Present Value of the coupon Leg.
        """

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.credit_default_swap(date, notional, hazard_curve, upfront_price, last_available, *args, **kwargs)

        return cds.couponLegNPV()

    @conditional_vectorize('date')
    def upfront_npv(self, date, notional, hazard_curve, upfront_price=1, last_available=True, *args, **kwargs):

        """
        :param date: pd.Datetime or QuantLib.Date
            Reference Date
        :param notional: float
            Size of the contract
        :param hazard_curve:
            the curve used for the calculation
        :param upfront_price: float
            The par value of the upfront payment.
        :param last_available: bool, optional
            Whether to use last available quotes if missing data.
        :return:
        QuantLib CDS Instrument
            CDS Net Present Value of the coupon Leg.
        """

        ql.Settings.instance().evaluationDate = to_ql_date(date)
        cds = self.credit_default_swap(date, notional, hazard_curve, upfront_price, last_available, *args, **kwargs)

        return cds.upfrontNPV()
