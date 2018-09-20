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

import QuantLib as ql
from tsfin.instruments.bonds._basebond import _BaseBond


class FixedRateBond(_BaseBond):
    """ Fixed rate bond.

    Parameters
    ----------
    timeseries: :py:obj:`TimeSeries`
        TimeSeries object representing the bond.
    attribute_overrides: dict
        Attributes to override ts_attributes in `timeseries`.

    Note
    ----
    See the :py:mod:`constants` for required key-value pairs in ts_attributes of `timeseries` and their possible values.
    """
    def __init__(self, timeseries):
        super().__init__(timeseries=timeseries)
        self.bond = ql.FixedRateBond(self.settlement_days, self.face_amount, self.schedule, self.coupons,
                                     self.day_counter, self.business_convention, self.redemption, self.issue_date)
        self.bond_components[self.maturity_date] = self.bond
        self._bond_components_backup = self.bond_components.copy()
