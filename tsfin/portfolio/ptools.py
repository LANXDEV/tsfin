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
Tools for the portfolio, trading, and back-testing framework.
"""
# Routines based on the Portfolio class.
import pandas as pd
from tsio.tools import create_folder
from pprint import pprint


def export_summary(portfolio, name, path, the_dates, other_args=None, function_list=None, security_objects=None,
                   save=True):
    # Exports portfolio characteristics to Excel

    if function_list is None:
        function_list = ['valuate', 'positions_percent']
    if other_args is None:
        other_args = dict()

    print(20*'-')
    print("Portfolio: exporting summary for the following list of functions:")
    pprint(function_list)
    print(20*'-')
    create_folder(path)

    result_dict = {}
    detailed_info = {}
    total_info = {}
    detailed_df_dict = {}
    total_info_df_dict = {}
    for func_and_options in function_list:
        if '_by_' in func_and_options:
            func_name, agg_attribute = func_and_options.split('_by_')
        else:
            # If there are no options
            func_name = func_and_options
        print("Portfolio: exporting {0}'s summary to file {1}".format(func_name, func_and_options+'_' + name))
        file_name = path + func_and_options + '_' + name + '.xlsx'
        result_dict[func_and_options] = {dt: getattr(portfolio, func_name)(date=dt,
                                                                           security_objects=security_objects,
                                                                           **other_args)
                                         for dt in the_dates}
        detailed_info[func_and_options] = {key: value[1] for key, value in result_dict[func_and_options].items()}
        total_info[func_and_options] = {key: value[0] for key, value in result_dict[func_and_options].items()}

        detailed_df_dict[func_and_options] = pd.DataFrame().from_dict(detailed_info[func_name]).unstack()
        detailed_df_dict[func_and_options].index.rename(['DATE', 'SECURITY'], inplace=True)
        detailed_df_dict[func_and_options] = detailed_df_dict[func_name].reset_index()
        detailed_df_dict[func_and_options].rename(columns={0: func_and_options}, inplace=True)
        total_info_df_dict[func_and_options] = pd.DataFrame().from_dict(total_info[func_name],
                                                                        orient='index').sort_index()
        total_info_df_dict[func_and_options].rename(columns={0: func_and_options}, inplace=True)

        if save is True:
            writer = pd.ExcelWriter(file_name)
            detailed_df_dict[func_and_options].to_excel(writer, sheet_name=func_name, index=False)
            total_info_df_dict[func_and_options].to_excel(writer, sheet_name='total ' + func_name)
            writer.save()

    # Now exporting full summary
    detailed_df_list = list()
    total_info_df_list = list()
    for func in function_list:
        if '_by_' not in func:
            # Building a list of the DataFrames generated above
            detailed_df_list.append(detailed_df_dict[func])
            total_info_df_list.append(total_info_df_dict[func])
    complete_detailed_df = pd.concat(detailed_df_list, axis=1)
    complete_detailed_df = complete_detailed_df.loc[:, ~complete_detailed_df.columns.duplicated()]
    complete_total_info_df = pd.concat(total_info_df_list, axis=1)
    if save is True:
        file_name = path + 'COMPLETE' + '_' + name + '.xlsx'
        writer = pd.ExcelWriter(file_name)
        complete_detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        complete_total_info_df.to_excel(writer, sheet_name='Total')
        writer.save()

    return path, detailed_df_dict, total_info_df_dict, complete_detailed_df, complete_total_info_df, result_dict
