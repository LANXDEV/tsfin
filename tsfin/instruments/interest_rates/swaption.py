import numpy as np
import QuantLib as ql
from tsfin.instruments.interest_rates.swaprate import SwapRate
from tsfin.base.qlconverters import to_ql_float_index, to_ql_currency
from tsfin.constants import MATURITY_TENOR, CURRENCY, FIXED_LEG_TENOR, INDEX


class SwapOption(SwapRate):

    def __init__(self, timeseries):
        super().__init__(timeseries)
        self.term_structure = ql.RelinkableYieldTermStructureHandle()
        self.currency = to_ql_currency(self.ts_attributes[CURRENCY])
        self.month_end = False
        self.index = to_ql_float_index(self.ts_attributes[INDEX], self._index_tenor, self.term_structure)
        self.maturity_tenor = ql.PeriodParser.parse(self.ts_attributes[MATURITY_TENOR])
        self.fixed_leg_tenor = ql.PeriodParser.parse(self.ts_attributes[FIXED_LEG_TENOR])

    def rate_helper(self, date, last_available=True, *args, **kwargs):

        rate = self.quotes.get_values(index=date, last_available=last_available, fill_value=np.nan)

        if np.isnan(rate):
            return None

        final_rate = ql.SimpleQuote(rate)
        return ql.SwaptionHelper(self.maturity_tenor, self._tenor, ql.QuoteHandle(final_rate), self.index,
                                 self.fixed_leg_tenor, self.day_counter, self.index.dayCounter(), self.term_structure)

    def set_yield_curve(self, yield_curve):

        self.term_structure.linkTo(yield_curve)
