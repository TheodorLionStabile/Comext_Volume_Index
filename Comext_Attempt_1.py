

import os
import pandas as pd
import numpy as np
import glob


# =============================================================================
# Title: Pre-Cleaning DF
# =============================================================================
# =============================================================================
# Description: This function is used to pre-clean any df to make sure that any next we perform do not have to deal with bad data structures
# =============================================================================

def data_pre_clean(df):
    # This selects all the values where values is but no quantity is traded.
    look = df[(df["QUANTITY_KG"]==0)&(df["QUANTITY_SUPPL_UNIT"]==0)&(df["VALUE_EUR"]>0)].index
    # The indices we selected before are dropped from the dataset
    df = df.drop(index=look)
    # Selecting the variables we need to work with to improve performance
    df = df[["REPORTER","PARTNER","TRADE_TYPE","FLOW","PERIOD","PRODUCT_NC","STAT_PROCEDURE","SUPPL_UNIT","VALUE_EUR","QUANTITY_KG","QUANTITY_SUPPL_UNIT"]]
    # Setting quantity to float to allow for future operations
    df["QUANTITY_KG"] =df["QUANTITY_KG"].astype(float)
    # Setting alternative indicators to float to perform operations
    df["QUANTITY_SUPPL_UNIT"] =df["QUANTITY_SUPPL_UNIT"].astype(float)
    #Dropping all total from relations to avoid double counting
    df = df[df["PRODUCT_NC"] != "TOTAL"].reset_index(drop=True)
    # Replacing all KG volume measures given that this is the Eurostat measure.
    df["QUANTITY_KG"] = np.where((df["SUPPL_UNIT"] != "NO_SUP") & (df["QUANTITY_SUPPL_UNIT"]>0),df["QUANTITY_SUPPL_UNIT"],df["QUANTITY_KG"])
    return df

# =============================================================================
# Title: Merging Codes
# =============================================================================
# =============================================================================
# Description: Since there is a need to merge codes in y-1 the simplest way is to apply the merge function below. 
# =============================================================================

def merge(df,dic,operation_type,code_list):
    
    #The operation_type condition allows to differentiate between normal merging or the use of the merge function in mergesplit.
    if operation_type == "list":
        for i in code_list:
            df.loc[df["PRODUCT_NC"] == i, "PRODUCT_NC"] = "new_code"
    else:
        #This takes the dictioanry we generate the determines the nature of the product code changes
        for key,value in dic.items():#This loops over the product code changes.
            list_of_cns = value[1]
            for j in list_of_cns:#Looping over the list of old product codes
                df.loc[df["PRODUCT_NC"] == j, "PRODUCT_NC"] = key #This line replaces the old product codes with the new one (in the case of simple merging)
    #This sums up all the new product codes changes, so that what was previosly multipl product codes becomes a single trade relationship
    result = df.groupby(
        ["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "PERIOD", "PRODUCT_NC","STAT_PROCEDURE"]
    ).agg({
        "QUANTITY_SUPPL_UNIT": "sum",
        "QUANTITY_KG": "sum",
        "VALUE_EUR": "sum",
        #"STAT_PROCEDURE": "first",
        "SUPPL_UNIT": "first"
    }).reset_index()
    
    return result


# =============================================================================
# Title: Splitting Code Weights
# =============================================================================
# =============================================================================
# Description: The function is able to determine the weights of the indvidiaul trade combinations for simple splitting operations. 
# =============================================================================

def split_weights(code_y_1,value,df_y,df_y_1,year_t):
    #This takes the dictionary that determined the nature of the product code changes and picks the values that need to be split into. 
    codes= value[1][0:]#This selects the values that need to be merged
    #This calculate hte number 
    #We change the Quantity_kg to ease the calculation later on.
    df_y["q"] = df_y["QUANTITY_KG"]
    #This selects the rows in the dataframe for year t. This allows us to generate the expected weighting of each product code. That is, we do not know the share of volume and value the new product codes would have. To correct for this we assume that the weights for the year t apply to year t-1
    filtered_y = df_y[df_y["PRODUCT_NC"].isin(codes)]#This takes all the rows we would be interested in.
    #This is the columns which we want to control for when splitting the product codes (i.e. identical lines and trade relations are identified through these group cols.)
    group_cols = ["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE", "PRODUCT_NC"]
    
    
    # Aggregate the data to sum VALUE_EUR and QUANTITY_SUPPL_UNIT over the relevant group
    aggregated_data = (
        filtered_y
        .groupby(group_cols)
        .agg({"VALUE_EUR": "sum", "q": "sum"})
        .reset_index()
    )
    #This determines the total value of the aggregated data over the year, allowing use to get the denominator of the shares. 
    total_values = (
        aggregated_data
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE"])
        .agg({"VALUE_EUR": "sum", "q": "sum"})
        .reset_index()
        .rename(columns={"VALUE_EUR": "TOTAL_VALUE_EUR", "q": "TOTAL_QUANTITY"})
    )
    # Now we can merge the two datasets we generates to build the weights.
    merged_data = pd.merge(aggregated_data, total_values, on=["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE"])
    #This calculate the weights we will apply to the different product/trade relation combinations to year t-1.
    merged_data["PERCENTAGE_VALUE_EUR"] = (merged_data["VALUE_EUR"] / merged_data["TOTAL_VALUE_EUR"])
    #Same thing as above but for volume measure. 
    merged_data["PERCENTAGE_QUANTITY"] = (merged_data["q"] / merged_data["TOTAL_QUANTITY"])
    #This filters the product codes that we want to split from in time t-1
    filtered_y_1 = df_y_1[df_y_1["PRODUCT_NC"].isin([code_y_1])]
    
    
    
    dfs_to_concat = []
    
    for i in filtered_y_1.index:
        # Filter the data according to your conditions
        temp_df = merged_data[
            (merged_data["REPORTER"] == filtered_y_1.loc[i, "REPORTER"]) &
            (merged_data["PARTNER"] == filtered_y_1.loc[i, "PARTNER"]) &
            (merged_data["TRADE_TYPE"] == filtered_y_1.loc[i, "TRADE_TYPE"])&
            (merged_data["FLOW"] == filtered_y_1.loc[i, "FLOW"])
        ]
        
        dfs_to_concat.append(temp_df)
    
    
    appended_data = pd.concat(dfs_to_concat, ignore_index=True)
    
    appended_data["PRODUCT_NC_from"] = code_y_1
    
    return appended_data


# =============================================================================
# Title: Selecting Products to Split
# =============================================================================
# =============================================================================
# Description: This function selects all the product codes that need to be split up.
# =============================================================================


def select_split(returned_dict):

    empty_dict = dict()
    
    for key,value in returned_dict.items():
        if value[0] == "split":
            empty_dict[key] = value
            
    return empty_dict



# =============================================================================
# Title: Combining all Products to Splits
# =============================================================================
# =============================================================================
# Description: This function merges all weights for the new CN8 codes.
# =============================================================================

def gen_total_df_for_split(keys,df_y,df_y_1,returned_dict,year_t):
    appended_data_merge_list = []
    
    for i in keys:
        
        appended_data = split_weights(i,returned_dict[i], df_y, df_y_1, year_t)
        appended_data = appended_data.drop_duplicates().reset_index(drop=True)
        appended_data_merge_list.append(appended_data)
    
    
    total_to_append = pd.concat(appended_data_merge_list, axis=0)
    total_to_append = total_to_append[["REPORTER","PARTNER","TRADE_TYPE","FLOW","STAT_PROCEDURE","PRODUCT_NC","PERCENTAGE_VALUE_EUR","PERCENTAGE_QUANTITY","PRODUCT_NC_from"]]

    return total_to_append





# =============================================================================
# Title: Splitting Products
# =============================================================================
# =============================================================================
# Description: This function splits up the products.
# =============================================================================
def split_product_codes(df,codes_to_split,append_dat):

    new_df = df[df["PRODUCT_NC"].isin(codes_to_split)].reset_index(drop=True)
    df = df[~df["PRODUCT_NC"].isin(codes_to_split)].reset_index(drop=True)
    res = pd.merge(new_df, append_dat, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW','STAT_PROCEDURE',"PRODUCT_NC"], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW', 'STAT_PROCEDURE',"PRODUCT_NC_from"], how='inner')
    res["QUANTITY_KG_new"] = res["QUANTITY_KG"] * res["PERCENTAGE_QUANTITY"]
    res["VALUE_EUR_new"] = res["VALUE_EUR"] * res["PERCENTAGE_VALUE_EUR"]
    res["UV"] = res["VALUE_EUR"]/res["QUANTITY_KG"]
    uv_means = (
        res
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE","PRODUCT_NC_x"])
        .agg({"UV": "mean"})
        .reset_index()
    )
    res2 = pd.merge(res, uv_means, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW','STAT_PROCEDURE',"PRODUCT_NC_x"], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW', 'STAT_PROCEDURE',"PRODUCT_NC_x"], how='inner')
    res2 = res2[["REPORTER","PARTNER","TRADE_TYPE","FLOW","PERIOD","STAT_PROCEDURE","QUANTITY_SUPPL_UNIT","SUPPL_UNIT","PRODUCT_NC_y","QUANTITY_KG_new","VALUE_EUR_new","UV_y"]].reset_index(drop=True)
    res2= res2.rename(columns = {"QUANTITY_KG_new":"QUANTITY_KG","VALUE_EUR_new":"VALUE_EUR","UV_y":"UV","PRODUCT_NC_y":"PRODUCT_NC"})
    df_fin = pd.concat([df, res2], ignore_index=True)
    
    return df_fin


# =============================================================================
# Section: Prepping Merge Split Codes
# =============================================================================
# =============================================================================
# Title: Selecting Mergesplit Codes
# =============================================================================
# =============================================================================
# Description: Using the select_mergesplit code allows us to select all codes that need to be merged and splitted after.
# =============================================================================



def select_mergesplit(returned_dict):

    empty_dict = dict()
    
    for key,value in returned_dict.items():
        if value[0] == "mergesplit":
            empty_dict[key] = value
            
    return empty_dict



# =============================================================================
# Title: Calculation frequencies of codes
# =============================================================================
# =============================================================================
# Description: frequency_calc returns the number of times an old product code is split into new product codes. This allows us to weigh their trade value and quantities later on. 
# =============================================================================


def frequency_calc(dictionary):
    code_list = list()
    for key, value in dictionary.items():
        cn_codes = value[1]
        for i in cn_codes:
            code_list.append(i)
    
    return code_list


# =============================================================================
# Title: Getting Frequencies
# =============================================================================
# =============================================================================
# Description: After getting the total population we need to calculate the actual frequencies 
# =============================================================================



# =============================================================================
# Title: Unique CN codes
# =============================================================================
# =============================================================================
# Description: These two lists are generated to have the unique set of olds codes and unique set of new codes they will be split into
# =============================================================================


def weight_counter_y(df,codes):
        df_filtered = df[df["PRODUCT_NC"].isin(codes)]
        df_filtered["q"] = df_filtered["QUANTITY_KG"]
        
        group_cols = ["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE", "PRODUCT_NC"]
        aggregated_data = (
            df_filtered
            .groupby(group_cols)
            .agg({"VALUE_EUR": "sum", "q": "sum"})
            .reset_index()
        )
        total_values = (
            aggregated_data
            .groupby(["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE"])
            .agg({"VALUE_EUR": "sum", "q": "sum"})
            .reset_index()
            .rename(columns={"VALUE_EUR": "TOTAL_VALUE_EUR", "q": "TOTAL_QUANTITY"})
        )
        
        merged_data = pd.merge(aggregated_data, total_values, on=["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE"])
        merged_data["PERCENTAGE_VALUE_EUR"] = (merged_data["VALUE_EUR"] / merged_data["TOTAL_VALUE_EUR"])
        merged_data["PERCENTAGE_QUANTITY"] = (merged_data["q"] / merged_data["TOTAL_QUANTITY"])
        merged_data = merged_data[["REPORTER","PARTNER","TRADE_TYPE","FLOW","STAT_PROCEDURE","PRODUCT_NC","PERCENTAGE_VALUE_EUR","PERCENTAGE_QUANTITY"]]
        merged_data = merged_data.drop_duplicates().reset_index(drop=True)
        return merged_data




# def lookup_values(row):
#     x = row['PRODUCT_NC_x']
#     y = row['PRODUCT_NC_y']
#     return match_matrix.loc[x, y]

# def lookup_values(row, match_matrix):
#     return match_matrix.loc[row['SomeColumn_X'], row['SomeColumn_Y']]


def lookup_values(row, match_matrix):
    x = row['PRODUCT_NC_x']
    y = row['PRODUCT_NC_y']
    return match_matrix.loc[x, y]



def apply_freq(row,frequency):
    x= row['PRODUCT_NC_x']
    ratio = frequency[x]
    return ratio




def reallocate_weights(df):
    df_value_0 = df[df['Value'] == 0]
    
    # Go over each row where Value == 0
    for index, row in df_value_0.iterrows():
        product_nc_x = row['PRODUCT_NC_x']
        product_nc_y = row["PRODUCT_NC_y"]
        calculated_value = row['PERCENTAGE_VALUE_EUR_x'] * row['PERCENTAGE_VALUE_EUR_y'] * row['Ratio']
        
        # Define the conditions for matching rows where Value == 1
        mask_value_1 = (
            (df['Value'] == 1) &
            (df['PRODUCT_NC_y'] == product_nc_y) &
            (df['REPORTER'] == row['REPORTER']) &
            (df['PARTNER'] == row['PARTNER']) &
            (df['TRADE_TYPE'] == row['TRADE_TYPE']) &
            (df['FLOW'] == row['FLOW']) &
            (df['STAT_PROCEDURE'] == row['STAT_PROCEDURE'])
        )
        
        # Update the PERCENTAGE_VALUE_EUR_y for these entries
        df.loc[mask_value_1, 'PERCENTAGE_VALUE_EUR_y'] += calculated_value
    
    
    for index, row in df_value_0.iterrows():
        product_nc_x = row['PRODUCT_NC_x']
        product_nc_y = row["PRODUCT_NC_y"]
        calculated_value = row['PERCENTAGE_QUANTITY_x'] * row['PERCENTAGE_QUANTITY_y'] * row['Ratio']
        
        # Define the conditions for matching rows where Value == 1
        mask_value_1 = (
            (df['Value'] == 1) &
            (df['PRODUCT_NC_y'] == product_nc_y) &
            (df['REPORTER'] == row['REPORTER']) &
            (df['PARTNER'] == row['PARTNER']) &
            (df['TRADE_TYPE'] == row['TRADE_TYPE']) &
            (df['FLOW'] == row['FLOW']) &
            (df['STAT_PROCEDURE'] == row['STAT_PROCEDURE'])
        )
        
        # Update the PERCENTAGE_VALUE_EUR_y for these entries
        df.loc[mask_value_1, 'PERCENTAGE_QUANTITY_y'] += calculated_value

    df = df[df["Value"]==1].reset_index(drop=True)
    df = df[["REPORTER","PARTNER","TRADE_TYPE","FLOW","STAT_PROCEDURE","PRODUCT_NC_y","PERCENTAGE_VALUE_EUR_y","PERCENTAGE_QUANTITY_y"]]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# =============================================================================
# Title Merge Split
# =============================================================================
# =============================================================================
# Description: This function performs the Merge Split task that we need when updating the CN8.
# =============================================================================



def merge_split(df,merge_split_codes,code_dictionary,df_merge):
    #This is the merge part of the merge split
    df = merge(df,code_dictionary,"list",merge_split_codes)
    #Here we pick the variables that will need to be split into multiple rows and multiplied by weights
   
    #This df 
    new_df = df[df["PRODUCT_NC"]=="new_code"].reset_index(drop=True)
    df = df[df["PRODUCT_NC"]!="new_code"].reset_index(drop=True)
    res = pd.merge(new_df, df_merge, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW','STAT_PROCEDURE'], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW', 'STAT_PROCEDURE'], how='inner')
    res["QUANTITY_KG_new"] = res["QUANTITY_KG"] * res["PERCENTAGE_QUANTITY_y"]
    res["VALUE_EUR_new"] = res["VALUE_EUR"] * res["PERCENTAGE_VALUE_EUR_y"]
    res["UV"] = res["VALUE_EUR"]/res["QUANTITY_KG"]
    uv_means = (
        res
        .groupby(["REPORTER", "PARTNER", "TRADE_TYPE", "FLOW", "STAT_PROCEDURE"])
        .agg({"UV": "mean"})
        .reset_index()
    )
    res2 = pd.merge(res, uv_means, left_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW','STAT_PROCEDURE'], right_on=['REPORTER', 'PARTNER', 'TRADE_TYPE', 'FLOW', 'STAT_PROCEDURE'], how='inner')
    res2 = res2[["REPORTER","PARTNER","TRADE_TYPE","FLOW","PERIOD","STAT_PROCEDURE","QUANTITY_SUPPL_UNIT","SUPPL_UNIT","PRODUCT_NC_y","QUANTITY_KG_new","VALUE_EUR_new","UV_y"]].reset_index(drop=True)
    res2= res2.rename(columns = {"QUANTITY_KG_new":"QUANTITY_KG","VALUE_EUR_new":"VALUE_EUR","UV_y":"UV","PRODUCT_NC_y":"PRODUCT_NC"})
    lol = pd.concat([df, res2], ignore_index=True)
    return lol
  
  


#def clean_dictionaries(years):
    


# =============================================================================
# Title: Performing 2024-2025 Operations
# =============================================================================

def sumfunc(row):
    tot = sum(row)
    return tot

def cleanup_prod_dict(row, col):
    x = row[col] #Call the respective product code value. 
    x = str(x) #Ensure the product code is a string.
    return x.replace(" ", "")  # Return the modified string without spaces


def gen_matrix(dictionary,y1,y_1):
    final_dict = dict()
    dictionary[f'CN20{y1}_CODE'] = dictionary.apply(lambda row: cleanup_prod_dict(row, f'CN20{y1}_CODE'), axis=1)
    dictionary[f'CN20{y_1}_CODE'] = dictionary.apply(lambda row: cleanup_prod_dict(row, f'CN20{y_1}_CODE'), axis=1)
    rows = dictionary[f"CN20{y1}_CODE"].unique()
    columns = dictionary[f"CN20{y_1}_CODE"].unique()
    df = pd.DataFrame(index=rows,columns=columns)
    
    for index, row in dictionary.iterrows():    
        r = dictionary.loc[index,f'CN20{y1}_CODE']
        c = dictionary.loc[index,f'CN20{y_1}_CODE']
        df.at[r,c] = 1
    
    df=df.fillna(0)
    
    df["Col_Sum"]=df.apply(sumfunc,axis=1)
    column_sums = df.sum()
    df.loc['Sum'] = column_sums
    
    empty_dic = dict()


    for i in df.index[:-1]:#Loops over the index to call the index values and store them in a dictionary
        if df.loc[i,"Col_Sum"]>1:#This selects all the roms with a row along sum greater than 1.
            empty_dic[i] = ["merge"]#For the rows with Col_Sum >1 we classify as a merge.
            values_row = df.loc[i,:]
            column_names = values_row.index[values_row == 1].tolist()
            empty_dic[i].append(column_names)
        else:
            empty_dic[i] = ["split"]
            values_row = df.loc[i,:]
            values_row = values_row[:-1]
            column_names = values_row.index[values_row == 1].tolist()
            empty_dic[i].append(column_names)

    empty_dic2 = dict()

    for i in df.columns[:-1]:
        if df.loc["Sum",i]<2:
            empty_dic2[i] = "merge"
        else:
            empty_dic2[i] = "split"

    for key,value in empty_dic.items():#Looping over the first dictionary we generated
        #The list of cases calls the column names in our relationship matrix df.
        list_of_cases = value[1]#The 1 is the because we are calling the list within the list. 
        for j in list_of_cases:#We loop over that list.
            if j in empty_dic2.keys():#This line loops over the empty_dic2 dictionary attempting to identify whether j insite its keys.
                empty_dic[key].append(empty_dic2[j])#This line appends the classification merge/split based on columns to empty_dic

    empty_dic3 = dict()#This allows us to do the final classification of the relations


    for key, value in empty_dic.items():#This loops over empty_dic which now both includes the classifcation generated usign the Col_Sum column as well as the Sum row. 
        # Gather all strings from the current list
        strings = [item for item in value if isinstance(item, str)]
        # Check if both "merge" and "split" are in the strings
        if "merge" in strings and "split" in strings:
            # Assuming value[1] is directly accessible and valid
            empty_dic3[key] = ["mergesplit", value[1]]
        elif "merge" in strings and "split" not in strings:
            empty_dic3[key] = ["merge", value[1]]
        else:
            reference_key = value[1][0] if isinstance(value[1], list) and len(value[1]) > 0 else None
            
            if reference_key is not None:
                if reference_key not in empty_dic3:
                    empty_dic3[reference_key] = ["split",[key]]
                else:
                    empty_dic3[reference_key][1].append(key)

    

    for key, value in empty_dic3.items():
        if (key in value[1]) and (len(value[1])==1):
            filtered_indices = df[(df[value[1][0]] == 1) & (df.index != "Sum")].index
            for ind in filtered_indices:
                if ind != key:
                    value[1].append(ind)
                    
        # Separate elif for different condition:
        elif value[0] == "mergesplit" and len(value[1]) == 1:
            value[0] = "merge"
                    
    for key, value in empty_dic3.items():
                 
        if value[0] == "split":
            short_term_storage = []
                
            # Iterate over other keys to find shared values in the other keys.
            for other_key, other_value in empty_dic3.items():
                # This line ensure that we do not refer to ourselves.
                if key == other_key:
                    continue
                    
                to_loop_in = other_value[1]
                for k in to_loop_in:
                    short_term_storage.append(k)
            #Checks whether the items are in the original 
            overlap = [item for item in short_term_storage if item in value[1]]
                
            if len(overlap) > 0:
                print(f"{key} is a mergesplit")
                value[0] = "mergesplit"
            else:
                print(f"{key} is a split")
                value[0] = "split"
            


    return empty_dic3


def merge_change_dictionary(dic):
    dic_wor = dict()
    for key,value in dic.items():
        if value[0] == "merge":
            dic_wor[key] = value
    
    return dic_wor



def run_funcs(y1,y_1):
    product_cn_nam = f"CN20{y1}_CN20{y_1}_Table"
    category_dic = gen_matrix(globals()[product_cn_nam],y1,y_1)
    merge_dictionary = merge_change_dictionary(category_dic)
    name_y_1 =f"full_20{y_1}"
    name_y1 =f"full_20{y1}"
    globals()[name_y1] = data_pre_clean(globals()[name_y1])
    globals()[name_y_1] = data_pre_clean(globals()[name_y_1])
    globals()[name_y_1] = merge(globals()[name_y_1],merge_dictionary,"NAN","NAN")
    split_list = select_split(category_dic)
    cn_codes_to_split = list(split_list.keys())
    total_to_append = gen_total_df_for_split(cn_codes_to_split, globals()[name_y1], globals()[name_y_1],category_dic,y_1)
    globals()[name_y_1]=split_product_codes(globals()[name_y_1],cn_codes_to_split,total_to_append)
    code_list = select_mergesplit(category_dic)
    cn_codes = frequency_calc(code_list)
    
    if len(cn_codes) > 0:
        frequency = {x: 1 / cn_codes.count(x) for x in cn_codes}
        unique_cn_codes_Y = list(code_list.keys())
        unique_cn_codes_Y_1 = list(set(cn_codes))
        match_matrix = pd.DataFrame(index=unique_cn_codes_Y_1, columns=unique_cn_codes_Y)
        
        for key, value in code_list.items():
            column = key
            for i in value[1]:
                match_matrix.loc[i,column] = 1
        
        match_matrix = match_matrix.fillna(0)
        weight_y = weight_counter_y(globals()[name_y1], unique_cn_codes_Y)
        weight_y_1 = weight_counter_y(globals()[name_y_1], unique_cn_codes_Y_1)
        merged_df = pd.merge(weight_y_1, weight_y, on=['REPORTER', 'PARTNER',"TRADE_TYPE","FLOW","STAT_PROCEDURE"], how='inner')
        merged_df['Value'] = merged_df.apply(lambda row: lookup_values(row, match_matrix), axis=1)
        #merged_df['Value'] = merged_df.apply(lookup_values, axis=1)
        merged_df['Ratio'] = merged_df.apply(lambda row: apply_freq(row,frequency), axis=1)
        merged_df = reallocate_weights(merged_df)
        globals()[name_y_1] = merge_split(globals()[name_y_1],unique_cn_codes_Y_1,category_dic,merged_df)
    globals()[name_y_1]["UV"] = np.where(globals()[name_y_1]["UV"].isna(), globals()[name_y_1]["VALUE_EUR"] / globals()[name_y_1]["QUANTITY_KG"], globals()[name_y_1]["UV"])
    globals()[name_y_1].to_csv(f'{y_1}_corrected.csv', index=False)


#,(18,19),(19,20),(22,23),(23,24),(24,25)

List_of_tuples = [(22,23),(23,24),(24,25)]

for o in List_of_tuples:
    

    for i in range(o[0],o[1]+1):
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
    
    
    for i in range(o[0],o[1]):
        if i >= 10 and i+1 <= 25:
            file_pattern = f"CN20{i+1}_CN20{i}_Table.xlsx"
            file_name= f"CN20{i+1}_CN20{i}_Table"
        elif i <=9:
            file_pattern = f"full_200{i}*.dat"
            file_name= f"full_200{i}"
            
        file_list = glob.glob(file_pattern)
        file_list= file_list[:-1]
    
        df_transform = pd.read_excel(file_pattern)
        
        globals()[file_name] = df_transform
    


    del df

    print(f"We have started runnin the code for {o[0]} and {o[1]}")
    run_funcs(o[1],o[0])
    
    del globals()[f"CN20{o[1]}_CN20{o[0]}_Table"]
    del globals()[f"full_20{o[0]}"]
    del globals()[f"full_20{o[1]}"]



