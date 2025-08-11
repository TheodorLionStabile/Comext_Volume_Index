# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:24:32 2025

@author: stabith
"""

import os
import pandas as pd
import numpy as np
import sdmx
import requests
import numpy as np
import glob
import dask.dataframe as dd

for i in range(24,25):
    if i >= 10:
        file_pattern = f"full_20{i}*.dat"
        file_name= f"full_20{i}"
    elif i <=9:
        file_pattern = f"full_200{i}*.dat"
        file_name= f"full_200{i}"
        
    file_list = glob.glob(file_pattern)
    file_list= file_list[:-1]    
    
    df = pd.concat((pd.read_csv(file) for file in file_list), ignore_index=True)
    globals()[file_name] = df

del df

full_2023 = pd.read_csv("23_corrected.csv")

def data_pre_clean(df):
    # This selects all the values where values is but no quantity is traded.
    look = df[(df["QUANTITY_KG"]==0)&(df["QUANTITY_SUPPL_UNIT"]==0)&(df["VALUE_EUR"]>0)].index
    # The indices we selected before are dropped from the dataset
    df = df.drop(index=look)
    # This selects all the values where values is but no quantity is traded.
    look = df[df["VALUE_EUR"]==0].index
    # The indices we selected before are dropped from the dataset
    df = df.drop(index=look).reset_index(drop=True)
    # Selecting the variables we need to work with to improve performance
    df = df[["REPORTER","PARTNER","TRADE_TYPE","PERIOD","PRODUCT_NC","SUPPL_UNIT","VALUE_EUR","QUANTITY_KG","QUANTITY_SUPPL_UNIT"]]
    # Setting quantity to float to allow for future operations
    df["QUANTITY_KG"] =df["QUANTITY_KG"].astype(float)
    # Setting alternative indicators to float to perform operations
    df["QUANTITY_SUPPL_UNIT"] =df["QUANTITY_SUPPL_UNIT"].astype(float)
    df["PERIOD"] =df["PERIOD"].astype(str)
    #Dropping all total from relations to avoid double counting
    df = df[df["PRODUCT_NC"] != "TOTAL"].reset_index(drop=True)
    df = df[~df['PRODUCT_NC'].str.contains("X", case=False, na=False)].reset_index(drop=True)
    # Replacing all KG volume measures given that this is the Eurostat measure.
    df["QUANTITY_KG"] = np.where((df["SUPPL_UNIT"] != "NO_SUP") & (df["QUANTITY_SUPPL_UNIT"]>0),df["QUANTITY_SUPPL_UNIT"],df["QUANTITY_KG"])
    df = df[["REPORTER","PARTNER","TRADE_TYPE","PERIOD","PRODUCT_NC","VALUE_EUR","QUANTITY_KG"]]
    df = (
        df
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE","PRODUCT_NC","PERIOD"])
        .agg({"VALUE_EUR": "sum","QUANTITY_KG": "sum"})
        .reset_index()
    )
    df["UV"] = df["VALUE_EUR"]/df["QUANTITY_KG"]
    return df

full_2024 = data_pre_clean(full_2024)


def clean_cleaned(df):
    df = df[~df['PRODUCT_NC'].str.contains("X", case=False, na=False)].reset_index(drop=True)
    look = df[df["VALUE_EUR"]==0].index
    # The indices we selected before are dropped from the dataset
    df = df.drop(index=look).reset_index(drop=True)
    df["PERIOD"] =df["PERIOD"].astype(str)
    df["PERIOD"] =df["PERIOD"].astype(str)
    df = df[["REPORTER","PARTNER","TRADE_TYPE","PERIOD","PRODUCT_NC","QUANTITY_KG","VALUE_EUR","UV"]]
    df = (
        df
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE","PRODUCT_NC","PERIOD"])
        .agg({"VALUE_EUR": "sum","QUANTITY_KG": "sum","UV":"mean"})
        .reset_index()
    )
    return df

full_2023 = clean_cleaned(full_2023)



def gen_yearly_means(df):
    uv_means = (
        df
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE","PRODUCT_NC"])
        .agg({"VALUE_EUR": "mean","QUANTITY_KG": "mean"})
        .reset_index()
    )
    df2 = pd.merge(df, uv_means, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE',"PRODUCT_NC"], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE',"PRODUCT_NC"], how='inner')
    df2=df2.rename(columns = {"QUANTITY_KG_x":"QUANTITY_KG","VALUE_EUR_x":"VALUE_EUR","QUANTITY_KG_y":"QUANTITY_KG_M","VALUE_EUR_y":"VALUE_EUR_M"})
    df2["UV_M"] = df2["VALUE_EUR_M"]/df2["QUANTITY_KG_M"]
    return df2

full_2023 = gen_yearly_means(full_2023)


def split_month(row, col):
    x = row[col] #Call the respective product code value. 
    x = x[-2:].split()
    return x[0]  # Return the modified string without spaces

full_2023["Month"] = full_2023.apply(lambda row: split_month(row, "PERIOD"), axis=1)

full_2024["Month"] = full_2024.apply(lambda row: split_month(row, "PERIOD"), axis=1)

test = full_2023[full_2023["PARTNER"] == "US"]
test2 = full_2024[full_2024["PARTNER"] == "US"]
rep = list(test["REPORTER"].unique())
rep2 = list(test2["REPORTER"].unique())

ra = list(test["PRODUCT_NC"].unique())
ra2 = list(test2["PRODUCT_NC"].unique())

lol = [item for item in ra if item not in ra2]


def combined(y_1,y1):
    df3 = pd.merge(y_1, y1, left_on=['REPORTER','PARTNER','TRADE_TYPE',"PRODUCT_NC","Month"], right_on=['REPORTER','PARTNER', 'TRADE_TYPE',"PRODUCT_NC","Month"], how='inner')
    df3["uvi_lasp_nom"] = df3["UV_y"]*df3["QUANTITY_KG_M"]
    value_total_EXP = df3['VALUE_EUR_M'][df3["TRADE_TYPE"]=="E"].sum()
    value_total_IMP = df3['VALUE_EUR_M'][df3["TRADE_TYPE"]=="I"].sum()
    df3["uvi_lasp_denom"] = np.where(df3["TRADE_TYPE"]=="E",value_total_EXP-df3['VALUE_EUR_M'],value_total_IMP-df3['VALUE_EUR_M'])
    q_total_EXP = df3['QUANTITY_KG_M'][df3["TRADE_TYPE"]=="E"].sum()
    q_total_IMP = df3['QUANTITY_KG_M'][df3["TRADE_TYPE"]=="I"].sum()
    df3["Q_M_j"] = np.where(df3["TRADE_TYPE"]=="E",q_total_EXP-df3['QUANTITY_KG_M'],q_total_IMP-df3['QUANTITY_KG_M'])
    df3["UV_M_j"]=df3["uvi_lasp_denom"]/df3["Q_M_j"]
    df3["uvi_pas_denom"] =df3["UV_M_j"]*df3["QUANTITY_KG_y"]

    return df3  

combined_df = combined(full_2023,full_2024)



def gen_uvi(df):
    df_summed = (
        df
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE","Month"])
        .agg({"uvi_lasp_nom": "sum","uvi_lasp_denom": "sum","VALUE_EUR_y": "sum","uvi_pas_denom": "sum"})
        .reset_index()
    )

    df_summed["lasp"] = df_summed["uvi_lasp_nom"]/df_summed["uvi_lasp_denom"]

    df_summed["pas"] = df_summed["VALUE_EUR_y"]/df_summed["uvi_pas_denom"]

    df_summed["fisch"] = np.sqrt(df_summed["lasp"]*df_summed["pas"])

    df_summed=df_summed[["REPORTER","PARTNER","TRADE_TYPE","Month","lasp","pas","fisch"]].reset_index(drop=True)
    
    df_summed["Month"] = df_summed["Month"].astype(float)

    df_summed = df_summed[["REPORTER","PARTNER","TRADE_TYPE","Month","fisch"]]
    
    return df_summed


uvi_df = gen_uvi(combined_df)


def gen_IVAL_IVOL(uvi,df):
    df = df[["REPORTER","PARTNER","TRADE_TYPE","PRODUCT_NC","VALUE_EUR_x","QUANTITY_KG_x","UV_x","UV_y","VALUE_EUR_M","QUANTITY_KG_M","UV_M","Month","VALUE_EUR_y","QUANTITY_KG_y"]]
    df["Month"] = df["Month"].astype(float)
    df = pd.merge(df, uvi, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE',"Month"], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE',"Month"], how='outer')
    df["IVAL"] = df["VALUE_EUR_y"]/df["VALUE_EUR_M"]
    df["IVOL"] = df["IVAL"]/df["fisch"]
    return df

final_comb = gen_IVAL_IVOL(uvi_df,combined_df)

del combined_df

def gen_quart_res(df):

    col = 'Month'
    conditions = [
        df[col] <= 3, 
        (df[col] >= 4) & (df[col] <= 6),
        (df[col] >= 7) & (df[col] <= 9),
        (df[col] >= 10) & (df[col] <= 12)
    ]
    choices = ["Q1", 'Q2', 'Q3', "Q4"]
    df["Quarter"] = np.select(conditions, choices, default=np.nan)
    df2 = (
        df
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE","Quarter"])
        .agg({"IVOL": "mean"})
        .reset_index()
    )
    
    return df2


quarterly_vals = gen_quart_res(final_comb)

test = quarterly_vals[quarterly_vals["PARTNER"]=="ZA"]
len(test["REPORTER"].unique())
