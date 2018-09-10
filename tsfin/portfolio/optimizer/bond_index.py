"""
A portfolio optimizer for building bond indexes.

WARNING: THIS CLASS IS NOT WORKING.

TODO: Finish this class and test it.
"""
import random

import numpy as np
import pandas as pd

from tsio import TimeSeriesCollection
from tsfin.portfolio import ptools as ptools


class BondIndex(object):
    def __init__(self, return_calculator=None, options=None):
        """A portfolio selector for bond indexes

        """

        self.options = options
        self.return_calculator = return_calculator
        self.unnalocated_symbol = 'UNNALOCATED_SYMBOL'
        self.concentration_rules = dict()
        # now checking the options
        if 'MAX_MATURITY' in options:
            self.max_maturity = options['MAX_MATURITY']
        else:
            self.max_maturity = pd.DateOffset(years=200)
        if 'MIN_MATURITY' in options:
            self.min_maturity = options['MIN_MATURITY']
        else:
            self.min_maturity = pd.DateOffset(days=0)
        if 'MIN_ELIGIBLE_MATURITY' in options:
            # Bonds with maturity closer than 'MIN_ELIGIBLE_MATURITY' will not enter the portfolio if they are not
            # already in.
            self.min_eligible_maturity = options['MIN_ELIGIBLE_MATURITY']
        else:
            self.min_eligible_maturity = None
        if 'SELL_DEFAULTED' in options:
            self.sell_defaulted = options['SELL_DEFAULTED']
        else:
            self.sell_defaulted = False
        if 'CORPORATE_MIN_OUTSTANDING' in options:
            self.corp_min_outstanding = options['CORPORATE_MIN_OUTSTANDING']
        else:
            self.corp_min_outstanding = 0
        if 'GOVT_MIN_OUTSTANDING' in options:
            self.govt_min_outstanding = options['GOVT_MIN_OUTSTANDING']
        else:
            self.govt_min_outstanding = 0
        if 'CORP_WEIGHT' in options:
            self.corp_weight = options['CORP_WEIGHT']
        else:
            self.corp_weight = 0.8
        if 'GOVT_WEIGHT' in options:
            self.govt_weight = options['GOVT_WEIGHT']
        else:
            self.govt_weight = 1 - self.corp_weight
        if 'MAX_TOLERANCE_AMONG_PORTFOLIOS' in options:
            self.max_tolerance_among_portfolios = options['MAX_TOLERANCE_AMONG_PORTFOLIOS']
        else:
            self.max_tolerance_among_portfolios = 0.0
        if 'MAX_TOLERANCE_AMONG_SECURITIES' in options:
            self.max_tolerance_among_securities = options['MAX_TOLERANCE_AMONG_SECURITIES']
        else:
            self.max_tolerance_among_securities = 0.0
        # Now checking the concentration rules
        if any('CONCENTRATION_RULES' in x for x in options.keys()):
            # Building concentration_rules dict
            for key, value in options.items():
                if 'CONCENTRATION_RULES' in key:
                    # Remove the first two words, which are CONCENTRATION and RULES
                    arguments = list(key.split('_'))[2:]
                    current_dict = self.concentration_rules
                    if len(arguments) >= 2:
                        for arg in arguments[:-1]:
                            try:
                                converted_arg = float(arg)
                            except:
                                converted_arg = arg
                            if converted_arg in current_dict.keys():
                                pass
                            else:
                                current_dict[converted_arg] = dict()
                            current_dict = current_dict[converted_arg]
                    last_arg = arguments[-1]
                    try:
                        last_arg = float(last_arg)
                    except:
                        pass
                    current_dict[last_arg] = value

        print('Successfully instantiated BondIndex Class with options:')
        print('max maturity: ' + str(self.max_maturity))
        print('min maturity: ' + str(self.min_maturity))
        print('min eligible maturity: {}'.format(self.min_eligible_maturity))
        print('sell defaulted : ' + str(self.sell_defaulted))
        print('min govt outstanding: {}'.format(self.govt_min_outstanding))
        print('corp weight: {}'.format(self.corp_weight))
        print('govt weight: {}'.format(self.govt_weight))
        print('rebalancing tolerance between portfolios: {}'.format(self.max_tolerance_among_portfolios))
        print('rebalancing tolerance between securities: {}'.format(self.max_tolerance_among_securities))
        print('Concentration rules:')
        print(self.concentration_rules)

    def optimize(self, ts_list, the_date, portfolio=None):
        print('BondIndex: optimizing for date ' + the_date.strftime('%Y-%m-%d'))
        # 1st Elimination: Get only Bonds
        ts_collection = TimeSeriesCollection()
        ts_collection.collection = ts_list
        available_securities = [ts for ts in ts_collection if
                                all((ts.get_attribute('TYPE') == 'BOND',
                                     ts.get_attribute('FIELD') == 'PX_LAST'))]
        # 2nd Elimination:
        # Applying max/min maturity filters,
        # Only bonds with equity TODO: Verify if the bonds have the BOND_TO_EQY TICKER in DB
        # Only Bonds with 'amount issued' information TODO: Verify in DB
        # Only bonds that have not defaulted until 'the_date'
        #
        govt_and_quasi_govt_securities = [ts for ts in available_securities if
                                          all((pd.to_datetime(ts.get_attribute('MATURITY')) > the_date,
                                               pd.to_datetime(
                                                   ts.get_attribute('MATURITY')) < the_date + self.max_maturity,
                                               pd.to_datetime(
                                                   ts.get_attribute('MATURITY')) >= the_date + self.min_maturity,
                                               ts.get_attribute('AMOUNT_ISSUED') is not None,
                                               self._consider_defaulted(ts.get_attribute('DEFAULT_DATE'),
                                                                       the_date) is False,
                                               ts.get_attribute('SUBTYPE_2') in ['SOVEREIGN', 'QUASI_SOVEREIGN'],
                                               self._get_bond_current_outstanding(ts, the_date, ts_collection) >=
                                               self.govt_min_outstanding,
                                               ))]
        corp_available_securities = [(ts, self._get_in_list('(' + ts.get_attribute('BOND_TO_EQY_TICKER') +
                                                           ')(CUR_MKT_CAP)', ts_collection))
                                     for ts in available_securities if
                                     all((pd.to_datetime(ts.get_attribute('MATURITY')) > the_date,
                                          pd.to_datetime(ts.get_attribute('MATURITY')) < the_date + self.max_maturity,
                                          pd.to_datetime(ts.get_attribute('MATURITY')) >= the_date + self.min_maturity,
                                          ts.get_attribute('BOND_TO_EQY_TICKER') is not None,
                                          ts.get_attribute('AMOUNT_ISSUED') is not None,
                                          self._consider_defaulted(ts.get_attribute('DEFAULT_DATE'), the_date) is
                                          False,
                                          ts.get_attribute('SUBTYPE_2') == 'CORPORATE',
                                          self._get_bond_current_outstanding(ts, the_date, ts_collection) >=
                                          self.corp_min_outstanding,
                                          ))]  # Elimination (2nd)
        # 3rd Elimination:
        # Only bonds which corresponding equity has (CUR_MKT_CAP) time series in DB
        corp_available_securities = [ts_pair for ts_pair in corp_available_securities if ts_pair[1] not in
                                     [None, [None]]]
        # 4th Elimination:
        # Only bonds which corresponding equity (CUR_MKT_CAP) time series ins not empty
        corp_available_securities = [ts_pair for ts_pair in corp_available_securities if ts_pair[1].ts_values.empty
                                     is not True]
        # 5th Elimination:
        # Only bonds which have price and cur_mkt_cap since before the_date
        # Only bonds which have price and cur_mkt_cap until before the_date
        corp_available_securities = [ts_pair for ts_pair in corp_available_securities if
                                     all((
                                         ts_pair[1].ts_values.last_valid_index() > the_date,
                                         ts_pair[1].ts_values.first_valid_index() <= the_date,
                                         ts_pair[0].ts_values.last_valid_index() > the_date,
                                         ts_pair[0].ts_values.first_valid_index() <= the_date,
                                     ))
                                     ]

        govt_and_quasi_govt_securities = [ts for ts in govt_and_quasi_govt_securities if
                                          all((
                                              ts.ts_values.last_valid_index() > the_date,
                                              ts.ts_values.first_valid_index() <= the_date,
                                              ts.ts_values.last_valid_index() > the_date,
                                              ts.ts_values.first_valid_index() <= the_date,
                                          ))
                                          ]
        # 6th Elimination:
        # If self.min_eligible_maturity is not None, then do not add NEW timeseries that mature before this date
        # Using 0.00001 to check wether the security was already in portfolio to account for rounding
        # problems
        if self.min_eligible_maturity is not None:
            corp_available_securities = [ts_pair for ts_pair in corp_available_securities if
                                         any((pd.to_datetime(ts_pair[0].get_attribute('MATURITY')) >= the_date +
                                              self.min_eligible_maturity,
                                              portfolio.positions[the_date].get(ts_pair[0].ts_name, 0) >= 0.00001
                                              ))
                                         ]

            govt_and_quasi_govt_securities = [ts for ts in govt_and_quasi_govt_securities if
                                              any((pd.to_datetime(ts.get_attribute('MATURITY')) >= the_date +
                                              self.min_eligible_maturity,
                                              portfolio.positions[the_date].get(ts.ts_name, 0) >= 0.00001
                                              ))
                                         ]
        optimized_govt_portfolio = dict()
        optimized_corp_portfolio = dict()
        if self.corp_weight > 0:
            corp_concentration_rules = self.concentration_rules.get('CORP', {})
            optimized_corp_portfolio, _ = self._recursive_distribute(self.corp_weight, corp_available_securities,
                                                                     the_date,
                                                                     'STOCK_MARKET_CAP', corp_concentration_rules,
                                                                     ts_collection, optimized_corp_portfolio, None)

        if self.govt_weight > 0:
            govt_concentration_rules = self.concentration_rules.get('GOVT', {})
            optimized_govt_portfolio, _ = self._recursive_distribute(self.govt_weight, govt_and_quasi_govt_securities,
                                                                     the_date, 'BOND_MARKET_OUTSTANDING',
                                                                     govt_concentration_rules, ts_collection,
                                                                     optimized_govt_portfolio, None)

        try:
            del optimized_corp_portfolio[self.unnalocated_symbol]
        except:
            pass
        try:
            del optimized_govt_portfolio[self.unnalocated_symbol]
        except:
            pass
        final_target_portfolio_dict = {**optimized_corp_portfolio, **optimized_govt_portfolio}
        # print('This is the optimized portfolio on ' + str(the_date))
        # pprint(final_target_portfolio_dict)
        # if the_date == pd.to_datetime('2013-12-31'):
        #     input('ok?')

        # Checking if we should actually change the portfolio, regarding the minimum divergence for rebalancing
        _, actual_portfolio = ptools.positions_percent(portfolio=portfolio, the_date=the_date,
                                                               security_objects=ts_list)
        corp_weight_in_actual_portfolio = sum(value for key, value in actual_portfolio.items() if ts_collection.get(
            key).get_attribute('SUBTYPE_2') == 'CORPORATE')
        govt_weight_in_actual_portfolio = sum(value for key, value in actual_portfolio.items() if ts_collection.get(
            key).get_attribute('SUBTYPE_2') in ['SOVEREIGN', 'QUASI-SOVEREIGN'])
        corp_diff_from_target = abs(self.corp_weight - corp_weight_in_actual_portfolio)
        govt_diff_from_target = abs(self.govt_weight - govt_weight_in_actual_portfolio)
        # pprint(actual_portfolio)
        if (corp_diff_from_target >= self.max_tolerance_among_portfolios or govt_diff_from_target >=
            self.max_tolerance_among_portfolios):
            return final_target_portfolio_dict
        diff_dict = dict()
        all_securities = set(list(actual_portfolio.keys()) + list(final_target_portfolio_dict.keys()))
        for security in all_securities:
            diff_dict[security] = abs(final_target_portfolio_dict.get(security, 0) - actual_portfolio.get(security, 0))
        if max(diff_dict.values()) >= self.max_tolerance_among_securities:
            return final_target_portfolio_dict
        # Else, if limits were not reached, then send a no-trading instruction
        return 'no_trade'

    def _get_in_list(self, ts_name, ts_collection):
        security = next((obj for obj in ts_collection if ts_name == getattr(obj, 'ts_name', None) or
                         ts_name == getattr(obj, 'name', None)), [None])
        # print("returning" + str(security))
        return security

    def _consider_defaulted(self, default_date, the_date):
        if self.sell_defaulted is True:
            try:
                if pd.to_datetime(default_date) <= the_date:
                    return True
            except:
                pass
        return False

    def _get_bond_current_outstanding(self, bond, the_date, ts_collection):
        amount_outstanding_ts_name = bond.ts_name.replace('PX_LAST', 'AMOUNT_OUTSTANDING_HISTORY')
        amt_outstd = self._get_in_list(amount_outstanding_ts_name, ts_collection)
        try:
            amount = amt_outstd.get_value(the_date, last_available=True)
        except:
            amount = np.nan
        if np.isnan(amount):
            amount = bond.get_attribute('AMOUNT_ISSUED')

        # Checking for errors!!
        if amount in [None, [None]]:
            raise ValueError('We have a problem in calculating the current outstanding...' + bond.ts_name)
        return amount

    def _recursive_distribute(self, final_proportion, available_securities, the_date, distribution_type,
                              concentration_rules, ts_collection, target_portfolio_dict, excluded_names):
        if not target_portfolio_dict:
            target_portfolio_dict = dict()
        if excluded_names is None:
            excluded_names = []
        total_proportion = 1 - sum((x[1] for x in target_portfolio_dict.items() if x[0] in excluded_names))
        # print('This is the total_proportion: {}'.format(total_proportion))
        # print("this is what i have to allocate:{}".format(total_proportion))
        # First we calculate
        if isinstance(available_securities[0], tuple):
            current_available_securities = [ts_pair for ts_pair in available_securities if ts_pair[0].ts_name not in
                                            excluded_names]
            current_available_securities_names = [ts_pair[0].ts_name for ts_pair in current_available_securities]
        else:
            current_available_securities = [ts for ts in available_securities if ts.ts_name not in excluded_names]
            current_available_securities_names = [ts.ts_name for ts in current_available_securities]

        if distribution_type == 'STOCK_MARKET_CAP':
            self._distribute_by_stock_market_cap(the_date, current_available_securities, target_portfolio_dict,
                                                 ts_collection, total_proportion)
            # print('Recursive Distribute has distributed in STOCK_MARKET_CAP mode, this how it looks:')
            # pprint(target_portfolio_dict)
            # print('This is the sum: {}'.format(sum(x[1] for x in target_portfolio_dict.items())))

            exclude_list = self._exclude_and_unconcentrate(target_portfolio_dict, concentration_rules,
                                                           ts_collection, excluded_names, the_date)
            if exclude_list == 'OK':
                # print('Recursive Distribute has received OK, returning...')
                # print(exclude_list)
                # print('returning....')
                for ts_name, ts_qty in target_portfolio_dict.copy().items():
                    if ts_name in current_available_securities_names:
                        target_portfolio_dict[ts_name] *= final_proportion
                return target_portfolio_dict, exclude_list
            else:
                #  means that exclude_list is not None, going to recursion
                return self._recursive_distribute(final_proportion, available_securities, the_date, distribution_type,
                                                  concentration_rules, ts_collection,
                                                  target_portfolio_dict, excluded_names=exclude_list)

        elif distribution_type == 'BOND_MARKET_OUTSTANDING':
            self._distribute_by_bond_market_outstanding(the_date, current_available_securities, target_portfolio_dict,
                                                        ts_collection, total_proportion)
            # print('Recursive Distribute has distributed in BOND_MARKET_OUTSTANDING mode, this how it looks:')
            # pprint(target_portfolio_dict)
            # print('This is the sum: {}'.format(sum(x[1] for x in target_portfolio_dict.items())))
            exclude_list = self._exclude_and_unconcentrate(target_portfolio_dict, concentration_rules,
                                                           ts_collection, excluded_names, the_date)
            if exclude_list == excluded_names:
                # print('Recursive Distribute has received same exclude_list from exclude_names:')
                # print(exclude_list)
                # print('returning....')
                for ts_name, ts_qty in target_portfolio_dict.copy().items():
                    if ts_name in current_available_securities_names:
                        target_portfolio_dict[ts_name] *= final_proportion
                return target_portfolio_dict, exclude_list
            else:
                # Going to recursion
                return self._recursive_distribute(final_proportion, available_securities, the_date,
                                                  distribution_type, concentration_rules,
                                                  ts_collection, target_portfolio_dict, excluded_names=exclude_list)

        else:
            raise ValueError('This distribution_type is not currently supported by BondIndex: ' + str(distribution_type
                                                                                                          ))

    def _distribute_by_stock_market_cap(self, the_date, available_securities, target_portfolio_dict,
                                        ts_collection, total_proportion):
        stock_bond_relational = {}
        # Relating equity ticker and bonds
        for bond, stock in available_securities:
            bond_outstanding = self._get_bond_current_outstanding(bond, the_date, ts_collection)
            stock_bond_relational[stock.ts_name] = stock_bond_relational.get(stock.ts_name, []) + [(bond,
                                                                                                 bond_outstanding)]
        # Removing duplicates to form a stock 'set'
        set_of_stocks = set([ts_pair[1] for ts_pair in available_securities])
        # Grouping stocks and their market caps
        stocks_with_market_cap = [(stock, stock.get_value(date=the_date)) for stock in set_of_stocks]
        # Calculating full market cap
        full_market_cap = sum((x[1] for x in stocks_with_market_cap))

        for stock, market_cap in stocks_with_market_cap:
            stock_fraction = market_cap / full_market_cap
            bond_total_issued = sum(x[1] for x in stock_bond_relational[stock.ts_name])
            for bond, bond_issued in stock_bond_relational[stock.ts_name]:
                bond_fraction = bond_issued / bond_total_issued
                target_portfolio_dict[bond.ts_name] = stock_fraction * bond_fraction * total_proportion

    def _distribute_by_bond_market_outstanding(self, the_date, available_securities, target_portfolio_dict,
                                               ts_collection, total_proportion):
        bonds_with_outstanding = [(bond, self._get_bond_current_outstanding(bond, the_date, ts_collection)) for bond
                                  in available_securities]
        total_outstanding = sum((x[1] for x in bonds_with_outstanding))
        for bond, bond_outstanding in bonds_with_outstanding:
            # print('This is the bond:{0}'.format(bond.ts_name))
            # print('This is the bond outstanding {}'.format(bond_outstanding))
            bond_fraction = bond_outstanding / total_outstanding
            # print('This is the bonds fraction: {}'.format(bond_fraction))
            target_portfolio_dict[bond.ts_name] = bond_fraction * total_proportion

    def _exclude_and_unconcentrate(self, target_portfolio_dict, concentration_rules, ts_collection, excluded_list,
                                  the_date):
        excluded_list = excluded_list.copy()
        # Necessary to avoid changing the external list
        if 'ISSUER-OUTSTANDING' in concentration_rules:
            return self._exclude_and_unconcentrate_by_issuer_and_bond(target_portfolio_dict, concentration_rules,
                                                                       ts_collection, excluded_list, the_date)
        elif 'OUTSTANDING' in concentration_rules:
            return self._exclude_and_unconcentrate_by_bond(target_portfolio_dict, concentration_rules, ts_collection,
                                                           excluded_list, the_date)
        else:
            return 'OK'

    def _exclude_and_unconcentrate_by_issuer_and_bond(self, target_portfolio_dict, concentration_rules, ts_collection,
                                                      source_excluded_list, the_date):
        # print('exclude_and_unconcentrate_by_issuer is running...')
        excluded_list = source_excluded_list.copy()
        max_concentration_per_issuer = concentration_rules['ISSUER-OUTSTANDING']['ISSUER']
        max_concentration_per_outstanding = concentration_rules['ISSUER-OUTSTANDING']['OUTSTANDING']
        issuer_concentration_dict = dict()
        non_blocked_issuer_concentration_dict = dict()
        issuer_bond_names = dict()
        non_blocked_issuer_bond_names = dict()
        issuer_concentration_field = 'BOND_TO_EQY_TICKER'
        for bond_name, bond_percent in target_portfolio_dict.items():
            if bond_name != self.unnalocated_symbol:
                issuer = ts_collection.get(bond_name).get_attribute(issuer_concentration_field)
                issuer_concentration_dict[issuer] = issuer_concentration_dict.get(issuer, 0) + bond_percent
                issuer_bond_names[issuer] = issuer_bond_names.get(issuer, []) + [bond_name]
                if bond_name not in excluded_list:
                    non_blocked_issuer_concentration_dict[issuer] = non_blocked_issuer_concentration_dict.get(issuer,
                                                                                                                 0) + \
                                                                      bond_percent
                    non_blocked_issuer_bond_names[issuer] = non_blocked_issuer_bond_names.get(issuer, []) + [bond_name]

        for issuer, issuer_concentration in sorted(issuer_concentration_dict.items(), key=lambda x: random.random()):
            if issuer_concentration > max_concentration_per_issuer + 0.00000001:
                # print('Found a problem with {}'.format(issuer))
                # print('This is the concentration_dict:')
                # pprint(issuer_concentration_dict)
                # Removing excluded bonds from the issuer adjustment
                this_issuer_total_qty_blocked_bonds = sum(target_portfolio_dict[name] for name in
                                                          issuer_bond_names[issuer] if name in excluded_list)
                non_blocked_max_concentration_per_issuer = max_concentration_per_issuer - \
                                                           this_issuer_total_qty_blocked_bonds
                non_blocked_issuer_concentration = non_blocked_issuer_concentration_dict[issuer]
                issuer_multiplication_factor = non_blocked_max_concentration_per_issuer / \
                                               non_blocked_issuer_concentration
                # print('Multiplying every available bond by: {}'.format(issuer_multiplication_factor))

                for bond_name in non_blocked_issuer_bond_names[issuer]:
                    target_portfolio_dict[bond_name] *= issuer_multiplication_factor
                    excluded_list.append(bond_name)
                target_portfolio_dict[self.unnalocated_symbol] = target_portfolio_dict.get(
                                                    self.unnalocated_symbol, 0.0) + non_blocked_issuer_concentration - \
                                                     non_blocked_max_concentration_per_issuer
                disp_bonds_in_portfolio_for_this_issuer = [ts_collection.get(x) for x in non_blocked_issuer_bond_names[
                    issuer]]
                for bond in disp_bonds_in_portfolio_for_this_issuer:
                    bond_name = bond.ts_name
                    bond_percent = target_portfolio_dict[bond.ts_name]
                    this_bond_outstanding = self._get_bond_current_outstanding(bond, the_date, ts_collection)
                    this_bond_outstanding_category = max(x for x in max_concentration_per_outstanding if x <
                                                         this_bond_outstanding)
                    this_bond_outstanding_limit = max_concentration_per_outstanding[this_bond_outstanding_category]
                    if bond_percent > this_bond_outstanding_limit + 0.00000001:
                        # print('Correcting a bond!!!: {0}. {1}'.format(bond_name, bond_percent))
                        new_concentration_rules = dict()
                        new_concentration_rules['OUTSTANDING'] = concentration_rules['ISSUER-OUTSTANDING'][
                            'OUTSTANDING']
                        disp_bonds_in_portfolio_for_this_issuer_names = [x.ts_name for x in
                                                                         disp_bonds_in_portfolio_for_this_issuer]
                        partial_excluded_names = [x for x in target_portfolio_dict if x not in
                                                  disp_bonds_in_portfolio_for_this_issuer_names]
                        partial_excluded_names.append(self.unnalocated_symbol)
                        self._recursive_distribute(1.0,
                                                   available_securities=disp_bonds_in_portfolio_for_this_issuer,
                                                   the_date=the_date, distribution_type='BOND_MARKET_OUTSTANDING',
                                                   concentration_rules=new_concentration_rules,
                                                   ts_collection=ts_collection,
                                                   target_portfolio_dict=target_portfolio_dict,
                                                   excluded_names=partial_excluded_names)
                        check = sum(value for key, value in target_portfolio_dict.items() if key in
                                    disp_bonds_in_portfolio_for_this_issuer_names)
                        if check < non_blocked_max_concentration_per_issuer:
                            target_portfolio_dict[self.unnalocated_symbol] += \
                                non_blocked_max_concentration_per_issuer - check

        if excluded_list != source_excluded_list:
            # Meaning at least one emissor was touched upon
            try:
                del target_portfolio_dict[self.unnalocated_symbol]
            except:
                pass
            return excluded_list

        for issuer, issuer_concentration in sorted(issuer_concentration_dict.items(), key=lambda x: random.random()):
            disp_bonds_in_portfolio_for_this_issuer = [ts_collection.get(x) for x in issuer_bond_names[issuer] if x
                                                       not in excluded_list]
            for bond in disp_bonds_in_portfolio_for_this_issuer:
                bond_name = bond.ts_name
                bond_percent = target_portfolio_dict[bond.ts_name]
                this_bond_outstanding = self._get_bond_current_outstanding(bond, the_date, ts_collection)
                this_bond_outstanding_category = max(x for x in max_concentration_per_outstanding if x <
                                                     this_bond_outstanding)
                this_bond_outstanding_limit = max_concentration_per_outstanding[this_bond_outstanding_category]
                if bond_percent > this_bond_outstanding_limit + 0.00000001:
                    # print('Correcting a bond!!!: {0}. {1}'.format(bond, bond_percent))
                    new_concentration_rules = dict()
                    new_concentration_rules['OUTSTANDING'] = concentration_rules['ISSUER-OUTSTANDING'][
                        'OUTSTANDING']
                    disp_bonds_in_portfolio_for_this_issuer_names = [x.ts_name for x in
                                                                     disp_bonds_in_portfolio_for_this_issuer]
                    total_allocation_in_disp_bonds = sum(value for key, value in target_portfolio_dict.items() if key
                                                         in disp_bonds_in_portfolio_for_this_issuer_names)
                    partial_excluded_names = [x for x in target_portfolio_dict if x not in
                                              disp_bonds_in_portfolio_for_this_issuer_names]
                    _, new_excluded_list = self._recursive_distribute(1.0,
                                                       available_securities=disp_bonds_in_portfolio_for_this_issuer,
                                                       the_date=the_date, distribution_type='BOND_MARKET_OUTSTANDING',
                                                       concentration_rules=new_concentration_rules,
                                                       ts_collection=ts_collection,
                                                       target_portfolio_dict=target_portfolio_dict,
                                                       excluded_names=partial_excluded_names)
                    # print('This is the new_excludedlist')
                    # input(new_excluded_list)
                    # print('This is the disp_bonds')
                    # input(disp_bonds_in_portfolio_for_this_issuer_names)
                    if all((x in new_excluded_list for x in disp_bonds_in_portfolio_for_this_issuer_names)):
                        # print('Going to all, it is ok')
                        # Meaning all the bonds for this emissor are now blocked!!
                        excluded_list += [x for x in new_excluded_list if x not in partial_excluded_names]
                        # After running the above _recursive_distribute only for the bonds of this issuer, this bond is
                        # sure to be at its maximum allocation (per individual bond rules), so add it to excluded list.
                        return excluded_list
        # If nothing is wrong, return the same excluded_list that was received. This will end the recursion in
        # the _recursive_distribute method.
        return 'OK'

    def _exclude_and_unconcentrate_by_bond(self, target_portfolio_dict, concentration_rules, ts_collection,
                                           source_excluded_list, the_date):
        excluded_list = source_excluded_list.copy()
        max_concentration_per_outstanding = concentration_rules['OUTSTANDING']
        for bond_name, bond_percent in target_portfolio_dict.copy().items():
            if bond_name not in excluded_list:
                bond = ts_collection.get(bond_name)
                this_bond_outstanding = self._get_bond_current_outstanding(bond, the_date, ts_collection)
                this_bond_outstanding_category = max(x for x in max_concentration_per_outstanding if x <
                                                     this_bond_outstanding)
                this_bond_outstanding_limit = max_concentration_per_outstanding[this_bond_outstanding_category]
                if bond_percent > this_bond_outstanding_limit + 0.00000001:
                    target_portfolio_dict[bond_name] = this_bond_outstanding_limit
                    # print('exclude_and_unconcentrate_by_bond is adding this bond to excluded_list:')
                    # print(bond_name)
                    excluded_list.append(bond_name)
                    # Check if the following changes anything
                    # return excluded_list
        return excluded_list

