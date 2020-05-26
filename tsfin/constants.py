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
Override according to attribute values in your database.
TODO: Implement a good way to override these constants (e.g.: with a config file).
"""

'''
Special names.
'''
QUOTES = 'PRICE'  # Name (in the database) of the component representing the quotes of an instrument.

'''
Database Instrument Types
'''
BOND = 'BOND'
FIXEDRATE = 'FIXEDRATE'
CALLABLEFIXEDRATE = 'CALLABLEFIXEDRATE'
FLOATINGRATE = 'FLOATINGRATE'
CONTINGENTCONVERTIBLE = 'CONTINGENTCONVERTIBLE'
DEPOSIT_RATE = 'DEPOSIT_RATE'
DEPOSIT_RATE_FUTURE = 'DEPOSIT_RATE_FUTURE'
CURRENCY_FUTURE = 'CURRENCY_FUTURE'
SWAP_RATE = 'SWAP_RATE'
SWAP_VOL = 'SWAP_VOL'
OIS_RATE = 'OIS'
EQUITY_OPTION = 'EQUITY_OPTION'
RATE_INDEX = 'RATE_INDEX'
FUND = 'FUND'
ETF = 'ETF'
EQUITY = 'EQUITY'
EXCHANGE_TRADED_FUND = 'EXCHANGE_TRADED_FUND'
CDS = 'CDS'
CDX = 'CDX'
ZERO_RATE = 'ZERO_RATE'
EURODOLLAR_FUTURE = 'EURODOLLAR_FUTURE'
CURRENCY = 'CURRENCY'
NDF = 'NDF'
# generic classification
INSTRUMENT = 'INSTRUMENT'

'''
Database attribute keys
'''
TYPE = 'TYPE'
BOND_TYPE = 'SUBTYPE'
FUND_TYPE = 'FUND_TYPE'
QUOTE_TYPE = 'QUOTE_TYPE'
YIELD_QUOTE_COMPOUNDING = 'YIELD_QUOTE_COMPOUNDING'
COMPOUNDING = 'COMPOUNDING'
YIELD_QUOTE_FREQUENCY = 'YIELD_QUOTE_FREQUENCY'
FREQUENCY = 'FREQUENCY'
ISSUE_DATE = 'ISSUE_DATE'
FIRST_ACCRUAL_DATE = 'FIRST_ACCRUAL_DATE'
MATURITY_DATE = 'MATURITY'
MATURITY_TENOR = 'MATURITY_TENOR'
TENOR_PERIOD = 'TENOR'
COUPON_FREQUENCY = 'FREQUENCY'
CALENDAR = 'CALENDAR'
BASE_CALENDAR = 'BASE_CALENDAR'
BUSINESS_CONVENTION = 'BUSINESS_CONVENTION'
DATE_GENERATION = 'DATE_GENERATION'
SETTLEMENT_DAYS = 'SETTLEMENT_DAYS'
FACE_AMOUNT = 'FACE_AMOUNT'
COUPONS = 'COUPON'
COUPON_TYPE = 'COUPON_TYPE'
DAY_COUNTER = 'DAY_COUNT'
REDEMPTION = 'REDEMPTION'
INDEX = 'INDEX'
INDEX_TENOR = 'INDEX_TENOR'
INDEX_TIME_SERIES = 'INTEREST_TIME_SERIES'
INDEX_DAY_COUNT = 'INDEX_DAY_COUNT'
SPREAD = 'SPREAD'
BASE_CURRENCY = 'BASE_CURRENCY'
COUNTER_CURRENCY = 'COUNTER_CURRENCY'
BASE_RATE_DAY_COUNTER = 'BASE_RATE_DAY_COUNTER'
BASE_RATE_COMPOUNDING = 'BASE_RATE_COMPOUNDING'
BASE_RATE_FREQUENCY = 'BASE_RATE_FREQUENCY'
COUNTER_RATE_DAY_COUNTER = 'COUNTER_RATE_DAY_COUNTER'
COUNTER_RATE_COMPOUNDING = 'COUNTER_RATE_COMPOUNDING'
COUNTER_RATE_FREQUENCY = 'COUNTER_RATE_FREQUENCY'
PAYMENT_LAG = 'PAYMENT_LAG'
FIXING_DAYS = 'FIXING_DAYS'
EXERCISE_TYPE = 'EXERCISE_TYPE'
OPTION_TYPE = 'OPTION_TYPE'
STRIKE_PRICE = 'STRIKE_PRICE'
UNDERLYING_INSTRUMENT = 'UNDERLYING_INSTRUMENT'
OPTION_CONTRACT_SIZE = 'OPTION_CONTRACT_SIZE'
PAYOFF_TYPE = 'PAYOFF_TYPE'
RECOVERY_RATE = 'RECOVERY_RATE'
SPREAD_TAG = 'SPREAD_TAG'
BASE_SPREAD_TAG = 'BASE_SPREAD_TAG'
FIXED_LEG_TENOR = 'FIXED_LEG_TENOR'
CALLED_DATE = 'CALLED_DATE'
EXPIRE_DATE_OVRD = 'EXPIRE_DATE_OVRD'
FUTURE_CONTRACT_SIZE = 'FUTURE_CONTRACT_SIZE'
EARLIEST_DATE = 'EARLIEST_DATE'
TICK_SIZE = 'TICK_SIZE'
TICK_VALUE = 'TICK_VALUE'
TERM_NUMBER = 'TERM_NUMBER'
TERM_PERIOD = 'TERM_PERIOD'
MONTH_END = 'MONTH_END'
COUNTRY = 'COUNTRY'
TICKER = 'TICKER'
COUPON_TYPE_RESET_DATE = 'COUPON_TYPE_RESET_DATE'
LAST_DELIVERY = 'LAST_DELIVERY'
CONTRACT_SIZE = 'CONTRACT_SIZE'
VALUE = 'VALUE'
RISK_VALUE = 'RISK_VALUE'
FUTURE_CONTRACT = 'FUTURE_CONTRACT'
# Equity and ETF specific
UNADJUSTED_PRICE = 'UNADJUSTED_PRICE'
EX_DIVIDENDS = 'DIVIDEND_SCHEDULE'
PAYABLE_DIVIDENDS = 'DIVIDEND_PAYABLE'
DIVIDENDS = 'DIVIDENDS'
DIVIDEND_YIELD = 'EQY_DVD_YLD_12M'
# Option specific
MID_PRICE = 'PX_MID'
IMPLIED_VOL = 'IVOL_MID'

# Trade Blotter
SIDE = 'SIDE'
DATE = 'DATE'
SETTLE_DATE = 'SETTLE_DATE'
QUANTITY = 'QUANTITY'
MULTIPLIER = 'MULTIPLIER'
TRADE_PRICE = 'TRADE_PRICE'
ACCRUED_INTEREST = 'ACCRUED_INTEREST'
COMMISSION = 'COMMISSION'
FEES = 'FEES'
BUY = 'BUY'
SELL = 'SELL'

'''
Database attribute values
'''
DISCOUNT = 'DISCOUNT'
CLEAN_PRICE = 'CLEAN_PRICE'
DIRTY_PRICE = 'DIRTY_PRICE'
YIELD = 'YIELD'
YIELD_CURVE = 'YIELD_CURVE'

'''
Configuration for yield curve classes
'''
# Names of attributes representing issue dates of securities in increasing order of precedence.
ISSUE_DATE_ATTRIBUTES = ['EFFECTIVE_PRECEDENCE_ISSUE', 'ISSUE', 'ISSUE_DATE']

'''
Equity Process Names
'''
BLACK_SCHOLES_MERTON = 'BLACK_SCHOLES_MERTON'
BLACK_SCHOLES = 'BLACK_SCHOLES'
HESTON = 'HESTON'
GJR_GARCH = 'GJR_GARCH'
