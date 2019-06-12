
import QuantLib as ql
from tsfin.constants import TENOR_PERIOD, MATURITY_DATE
from tsfin.base.qlconverters import to_ql_date
from tsfin.base.basetools import conditional_vectorize


class QuoteHandle:

    def __init__(self, value):
        self.value = ql.SimpleQuote(value)
        self.quote_handle = ql.RelinkableQuoteHandle(self.value)

    def set_value(self, value):
        self.value = ql.SimpleQuote(value)

    def link_to(self, simple_quote):
        self.quote_handle.linkTo(simple_quote)

    def quote_value(self):
        return self.quote_handle.value()

    def quote_handle(self):
        return self.quote_handle


class SpreadHandle:

    def __init__(self, ts_collection):
        self.ts_collection = ts_collection
        self.spreads = dict()

    @conditional_vectorize('date')
    def update_spread_from_timeseries(self, date, last_available=True):

        date = to_ql_date(date)
        self.spreads[date] = dict()
        for ts in self.ts_collection:
            spread = ts.get_values(date, last_available=last_available)
            try:
                spread_date_or_tenor = date + ql.Period(ts.ts_attributes[TENOR_PERIOD])
            except AttributeError:
                spread_date_or_tenor = to_ql_date(ts.ts_attributes[MATURITY_DATE])

            self.spreads[date][spread_date_or_tenor] = QuoteHandle(spread)

    @conditional_vectorize('date')
    def spread_handle(self, date, last_available=True):

        date = to_ql_date(date)
        try:
            return self.spreads[date]

        except KeyError:
            self.update_spread_from_timeseries(date=date, last_available=last_available)
            return self.spreads[date]
