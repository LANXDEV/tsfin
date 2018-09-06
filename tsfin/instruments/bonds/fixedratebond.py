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
