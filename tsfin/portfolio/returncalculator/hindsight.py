from lanxad.base.timeseries import TimeSeries


class HindsightReturnCalculator(object):

    def __init__(self):
        pass

    def calculate_return(self, security, date, time_horizon):
        if isinstance(security, TimeSeries):
            calculated_return = (security.get_value(date=date + time_horizon) / security.get_value(date)) - 1
            # extrapolates prices for dates with no price
            return calculated_return, 0.0  # risk is zero
        else:
            raise ValueError("I dont know how to calculate return for this security")
