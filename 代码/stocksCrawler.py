#!-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import pandas_datareader.data as web
import datetime as dt
import os
import shutil
from time import sleep
start = dt.datetime(2016, 3, 10)
end =  dt.datetime(2016, 6, 15)

# stocks = ['AAL','ADP','CERN','CSCO','EA','EBAY','EXPE','FISV','TMUS','TSLA','TXN','WDC']

path_list1 = ['aal_2016_06_15_12_01_41.xlsx', 'aapl_2016_06_15_14_30_09.xlsx', 'adbe_2016_06_15_12_03_58.xlsx', 'adp_2016_06_15_12_06_34.xlsx',
             'adsk_2016_06_15_12_07_20.xlsx', 'akam_2016_06_15_12_08_04.xlsx', 'alxn_2016_06_15_12_08_55.xlsx', 'amat_2016_06_15_12_09_54.xlsx',
             'amgn_2016_06_15_12_10_36.xlsx', 'amzn_2016_06_15_12_13_10.xlsx', 'atvi_2016_06_15_14_34_02.xlsx', 'avgo_2016_06_15_12_15_32.xlsx',
             'bbby_2016_06_15_12_16_08.xlsx', 'bidu_2016_06_15_12_16_43.xlsx', 'bmrn_2016_06_15_12_18_36.xlsx', 'ca_2016_06_15_12_19_11.xlsx',
             'celg_2016_06_15_12_20_12.xlsx', 'cern_2016_06_15_12_21_43.xlsx', 'chkp_2016_06_15_12_22_01.xlsx', 'chtr_2016_06_15_12_22_16.xlsx',
             'cmcsa_2016_06_15_12_22_45.xlsx', 'cost_2016_06_15_12_24_55.xlsx', 'csco_2016_06_15_12_26_22.xlsx', 'ctrp_2016_06_15_12_28_57.xlsx',
             'ctsh_2016_06_15_12_31_21.xlsx', 'disca_2016_06_15_12_32_37.xlsx', 'disck_2016_06_15_12_33_30.xlsx', 'dish_2016_06_15_12_36_19.xlsx',
             'dltr_2016_06_15_12_42_30.xlsx', 'ea_2016_06_15_12_43_03.xlsx', 'ebay_2016_06_15_12_44_05.xlsx', 'endp_2016_06_15_12_44_49.xlsx',
             'esrx_2016_06_15_12_45_25.xlsx', 'expe_2016_06_15_12_46_18.xlsx', 'fast_2016_06_15_12_46_49.xlsx', 'fb_2016_06_15_12_49_41.xlsx',
             'fisv_2016_06_15_12_50_12.xlsx', 'fox_2016_06_15_12_50_40.xlsx']
path_list2 = ['foxa_2016_06_15_12_51_17.xlsx', 'gild_2016_06_15_12_53_38.xlsx', 'googl_2016_06_15_13_06_09.xlsx', 'goog_2016_06_15_12_56_37.xlsx',
              'hsic_2016_06_15_13_12_31.xlsx', 'ilmn_2016_06_15_13_12_41.xlsx', 'inct_2016_06_15_13_13_27.xlsx', 'incy_2016_06_15_13_13_08.xlsx',
              'intu_2016_06_15_13_13_39.xlsx', 'isrg_2016_06_15_13_13_56.xlsx', 'jd_2016_06_15_13_14_18.xlsx', 'khc_2016_06_15_13_14_47.xlsx',
              'lbtya_2016_06_15_13_15_06.xlsx', 'lbtyk_2016_06_15_13_15_16.xlsx', 'lltc_2016_06_15_13_15_22.xlsx', 'lmca_2016_06_15_13_15_31.xlsx',
              'lmck_2016_06_15_13_15_42.xlsx', 'lrcx_2016_06_15_13_15_48.xlsx', 'lrcx_2016_06_15_13_16_01.xlsx', 'lvnta_2016_06_15_13_16_23.xlsx',
              'mar_2016_06_15_13_16_33.xlsx', 'mat_2016_06_15_13_17_09.xlsx', 'mdlz_2016_06_15_13_18_09.xlsx', 'mnst_2016_06_15_13_19_06.xlsx',
              'msft_2016_06_15_14_35_59.xlsx', 'mu_2016_06_15_13_23_44.xlsx', 'mxim_2016_06_15_13_27_24.xlsx', 'myl_2016_06_15_13_31_59.xlsx',
              'nclh_2016_06_15_13_32_29.xlsx', 'nflx_2016_06_15_13_37_15.xlsx', 'ntap_2016_06_15_13_41_12.xlsx', 'ntes_2016_06_15_13_43_09.xlsx',
              'nvda_2016_06_15_13_43_57.xlsx', 'nxpi_2016_06_15_13_45_40.xlsx', 'orly_2016_06_15_13_45_48.xlsx', 'payx_2016_06_15_13_48_59.xlsx',
              'pcar_2016_06_15_13_49_29.xlsx', 'pcln_2016_06_15_13_50_12.xlsx', 'pypl_2016_06_15_13_51_08.xlsx', 'qcom_2016_06_15_13_51_56.xlsx',
              'qvca_2016_06_15_13_52_19.xlsx', 'regn_2016_06_15_13_53_08.xlsx', 'rost_2016_06_15_13_54_00.xlsx', 'sbac_2016_06_15_13_54_30.xlsx',
              'sbux_2016_06_15_13_55_41.xlsx', 'sndk_2016_06_15_18_25_14.xlsx', 'srcl_2016_06_15_13_56_25.xlsx', 'stx_2016_06_15_13_56_49.xlsx',
              'swks_2016_06_15_13_57_26.xlsx', 'symc_2016_06_15_18_19_59.xlsx', 'tmus_2016_06_15_13_58_03.xlsx', 'trip_2016_06_15_13_58_32.xlsx',
              'tsco_2016_06_15_13_59_15.xlsx', 'tsla_2016_06_15_14_01_51.xlsx', 'txn_2016_06_15_14_03_07.xlsx', 'ulta_2016_06_15_14_03_52.xlsx',
              'viab_2016_06_15_14_04_32.xlsx', 'vod_2016_06_15_14_05_09.xlsx', 'vrsk_2016_06_15_14_05_35.xlsx', 'vrtx_2016_06_15_19_00_10.xlsx',
              'wba_2016_06_15_14_06_38.xlsx', 'wdc_2016_06_15_14_07_37.xlsx', 'wfm_2016_06_15_14_08_18.xlsx', 'xlnx_2016_06_15_14_10_06.xlsx', 'yhoo_2016_06_15_18_17_31.xlsx']
def iter_files(root_dir):
    #遍历根目录
    all_pic_path = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            file_name = os.path.join(root,file)
            all_pic_path.append(file_name)
    return all_pic_path
def iter_directory(root_dir):
    #遍历根目录
    all_pic_path = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            file_name = root
            all_pic_path.append(file_name)
    return all_pic_path
all_stock_path1 = iter_files('stock')
print(len(all_stock_path1))

# print(all_stock_path)
# stocks_no_data = ['CA', 'ESRX', 'FOX', 'FOXA', 'LLTC', 'LMCA', 'LMCK', 'LVNTA', 'PCLN', 'QVCA', 'SNDK', 'WFM', 'YHOO']
# stocks = []
# stocks_path = []
# print(len(path_list1) + len(path_list2))
# 获得所有股票简写list
# for path in path_list1:
#     stock = path.split('_')[0].upper()
#     if stock in stocks_no_data:
#         continue
#     stocks.append(stock)
#
# for path in path_list2:
#     stock = path.split('_')[0].upper()
#     if stock in stocks_no_data:
#         continue
#     stocks.append(stock)

# print(stocks)
# print(len(stocks))
# 获得所有股票的路径将其移动到stock的all_stock_path文件夹中
# stock_get = []
# for stock in path_list1:
#     stock_name = 'export_dashboard_' + stock
#     for stock_path in all_stock_path:
#         if stock_path.find(stock_name) != -1:
#             stocks_path.append(stock_path)
#             stock_get.append(stock)
# for stock in path_list2:
#     stock_name = 'export_dashboard_' + stock
#     for stock_path in all_stock_path:
#         if stock_path.find(stock_name) != -1:
#             stocks_path.append(stock_path)
#             stock_get.append(stock)
# print(stocks_path)
# print(stock_get)
# for stock_path, path in zip(stocks_path, stock_get):
#     stock = path.split('_')[0].upper()
#     if os.path.exists('./stock/'+stock):
#         print('./'+stock_path)
#         shutil.move('./'+stock_path, './stock/'+stock+'/'+stock+'.xlsx')
# CA ESRX FOX FOXA LLTC LMCA LMCK LVNTA PCLN QVCA SNDK WFM YHOO未获取成功
# for stock in stocks:
#     try:
#         path = './stock/' + stock
#         os.makedirs(path)
#     except Exception as e:
#         print(e)

# for stock in stocks:
#     try:
#         df_stock = web.DataReader(stock, 'yahoo', start, end)
#
#         df_stock.to_excel('./stock/'+stock+'/'+stock+'_stock.xlsx')
#
#
#     except Exception as e:
#         print(e)