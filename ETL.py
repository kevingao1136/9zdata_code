import os
import pandas as pd
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
from datetime import timedelta
import warnings
import math
warnings.simplefilter("ignore")

export_path = '/app2/kevin_workspace/data03180414/'
# export_path = ''

def now(): 
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    # print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object and props[col].dtype != 'datetime64[ns]':  # Exclude strings
            
            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")
    
    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    # print("Memory usage is: ",mem_usg," MB")
    # print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

def impala_query(sql):
    '''
    Parameters:
        sql - sql query
    Returns:
        data - dataframe from sql query
    '''
    config = {'host': '10.82.28.136', 'port': 21051, 'database': 'default', 'use_ssl': False}
    impala_conn = connect(**config)
    impala_cursor = impala_conn.cursor()
    impala_cursor.execute(sql)
    data = as_pandas(impala_cursor)
    impala_conn.close()

    return data

def cartesian_product_basic(left, right):
    return (left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))

def get_datefeature_prominfo():
    """
    Returns:
        date_feature:   day features - mday, yday, holidays
        prom_info:      promotion features data
    """
    # Reduce date_feature
    date_feature = pd.read_csv('../data/date_feature.csv', parse_dates=['day_dt'])
    date_feature.drop(['day_of_mth','day_of_qtr','day_of_yr','wk_of_mth','wk_of_qtr','wk_of_yr'],axis=1,inplace=True)
    date_feature, _ = reduce_mem_usage(date_feature)

    # Reduce prom_info
    prom_path = '../data/ods_bo_dim_prmt.csv'
    prom_info = pd.read_csv(prom_path)
    prom_info = prom_info[['item', 'offer_code', 'offer_start_date', 'offer_end_date', 
                        'osd_type', 'outputs', 'x件y折', 'x元y件', '加x元多y件', '买x送y', '满x减y',
                        'x件减y', '第x件y折', '换购', 'is_vip', 'free_gift', 'unit_price', 
                        'required_num', 'amount', 'prom_price', 'retail_price','flag']]
    prom_info['offer_start_date'] = pd.to_datetime(prom_info['offer_start_date'], errors='coerce')
    prom_info['offer_end_date'] = pd.to_datetime(prom_info['offer_end_date'], errors='coerce')
    prom_info = prom_info.dropna(subset=['offer_end_date', 'offer_start_date'])
    prom_info['offer_code'] = prom_info['offer_code'].astype(int).astype(str)
    prom_info, _ = reduce_mem_usage(prom_info)

    return date_feature, prom_info

def gen_pog_and_date(item, date_feature):
    """
    Parameters:
        item: item code (int)
        date_feature: feature of calendar data
    Returns:
        data: POG data
    """
    pog_data = impala_query(f"select ITEM_IDNT, WK_IDNT, LOC_IDNT from scai.0318_0414_pog where ITEM_IDNT = '{str(item)}'")
    data = pog_data.merge(date_feature, on='wk_idnt', how='left')
    data.rename(columns={'item_idnt': 'item_code', 'loc_idnt': 'store_id'}, inplace=True)

    ####################### TEST DATA #######################
    item_code_df = pd.DataFrame(data=[item], columns=['item_code'])
    test_data = pd.read_csv('test_sample.csv') # TEST PERIOD AND ALL STORE ID
    cross_join_prom = cartesian_product_basic(item_code_df, test_data)
    cross_join_prom['day_dt'] = pd.to_datetime(cross_join_prom['day_dt'])
    test_data = pd.merge(cross_join_prom, date_feature, on='day_dt', how='left')
    data = pd.concat([data, test_data])
    data.drop(columns=['wk_idnt'], inplace=True)
    data['item_code'] = data['item_code'].apply(pd.to_numeric)
    data['day_dt'] = pd.to_datetime(data['day_dt'])
    data.drop_duplicates(inplace=True)

    return data

def get_store_info():
    '''
    Returns:
        store_info - store data
    '''
    #store info
    store_info = impala_query(f"""
    select loc_idnt as store_id,
                    distt_idnt,
                    regn_idnt,
                    area_idnt,
                    mkt_idnt,
                    store_type,
                    loc_fmt_cde,
                    loc_selling_area,
                    loc_tot_area,
                    pds_location_type_en,
                    pds_mtg_type,
                    pds_store_segmentation,
                    pds_grace,
                    pds_floor_type,
                    city_tier,
                    zone_id,
                    is_intracity_dlvr_store,
                    loc_wh,
                    is_central_store
    from ods_sc.dim_organization
    where is_e_store='N' and loc_type_cde='S' and loc_end_dt is null and area_idnt in ('1','2','3','4')
    """)

    store_info['store_id'] = store_info['store_id'].apply(pd.to_numeric)
    if store_info['store_id'].duplicated().any() == True:
        print(f"ALERT>>>>>>>>>>>>>>DUPLICATED STORE_ID")

    return store_info

def get_outlier(df,store,item,work_day,quantity):
    
    start_dt = work_day-timedelta(days=7)
    end_dt = work_day+timedelta(days=7)
    tmp = df[(df.store_id==store)&(df.item_code==item)&(df.day_dt>=start_dt)&(df.day_dt<=end_dt)&(df.ttl_quantity<=200)]
    all_quantity = tmp.ttl_quantity.sum()
    if quantity/all_quantity>1:
        return all_quantity/len(tmp)
    else:
        return quantity

def getsaletable(item):

    data = impala_query(
    f"select * from scai.0318_0414_item_sales where item_code = '{str(item)}'"
    )

    data['day_dt'] = pd.to_datetime(data['day_dt'])
    data[['item_code','store_id','ttl_quantity','avg_amountofsales','ttl_amt_payment']] = data[['item_code','store_id','ttl_quantity','avg_amountofsales','ttl_amt_payment']].apply(pd.to_numeric)
    outlier_df = data[data.ttl_quantity>=200]
    if len(outlier_df) == 0:
        return data
    else:
        outlier_df['sale_qty2'] = outlier_df.apply(lambda x:get_outlier(data,x.store_id,x.item_code,x.day_dt,x.ttl_quantity),axis=1)
        outlier_df['sale_qty2'] = outlier_df['sale_qty2'].fillna(0.0).apply(lambda x:math.ceil(x))
        outlier_df = outlier_df.drop(['ttl_quantity','avg_amountofsales','ttl_amt_payment'],axis=1)
        data = pd.merge(data,outlier_df,how='left',on=['store_id','day_dt','item_code'])
        data['sale_qty2'] = data['sale_qty2'].fillna(data['ttl_quantity'])
        data = data.drop(['ttl_quantity'],axis=1).rename({'sale_qty2':'ttl_quantity'},axis=1)
        if data[['store_id','item_code','day_dt']].duplicated().any() == True:
            print(f"ALERT>>>>>>>>>>>>>>> DUPLICATES!!!")
        
        return data

def get_prom(data, prom_info):
    """
    :parameter  day_dt datetime64
    """
    item = int(data['item_code'].unique()[0])
    prom_info = prom_info[prom_info['item'] == item]
    time_range = pd.date_range('2018-04-01', '2021-05-31')
    time_merge = pd.DataFrame()
    time_merge['day_dt'] = time_range

    cross_join_prom = cartesian_product_basic(time_merge, prom_info)
    cross_join_prom = cross_join_prom[(cross_join_prom['day_dt'] >= cross_join_prom['offer_start_date']) & (
            cross_join_prom['day_dt'] <= cross_join_prom['offer_end_date'])]

    cross_join_prom['p_rate'] = 1 - cross_join_prom['unit_price'] / cross_join_prom['retail_price']
    cross_join_prom.loc[cross_join_prom['p_rate'] < 0, 'p_rate'] = 0

    cross_join_prom = cross_join_prom.sort_values(['item', 'day_dt', 'unit_price'])
    cross_join_prom = cross_join_prom.groupby(['day_dt', 'item']).first().reset_index()

    data = pd.merge(data, cross_join_prom, right_on=['day_dt', 'item'], left_on=['day_dt', 'item_code'], how='left')

    # todo: merge store
    offer_list = data['offer_code'].dropna().unique().astype(float).astype(int).astype(str).tolist()
    if len(offer_list) == 0:
        offer_list = ['00000000']
    # offers_text = ','.join(offer_list)
    # loc_data = impala_query(f"select * from ods_sc.ods_bo_dim_prmt_pp_pkgloc where offer_code in ({str(offers_text)})")
    # loc_data['offer_code'] = loc_data['offer_code'].astype(str)
    # data['offer_code'] = data['offer_code'].astype(str)
    # loc_data['store'] = loc_data['store'].astype(int)
    # data['store_id'] = data['store_id'].astype(int)
    # data = pd.merge(data, loc_data, left_on=['offer_code', 'store_id'], right_on=['offer_code', 'store'],
    #                 how='left')
    # prom_cols = ['offer_code', 'offer_start_date', 'offer_end_date', 'osd_type',
    #              'outputs', 'x件y折', 'x元y件', '加x元多y件', '买x送y', '满x减y', 'x件减y', '第x件y折',
    #              '换购', 'is_vip', 'free_gift', 'unit_price', 'required_num', 'amount',
    #              'prom_price', 'retail_price', 'flag', 'p_rate']
    # data.loc[data['package'].isna(), prom_cols] = np.nan
    data = data.drop(
        ['item',
        #  'store', 'last_update_id', 'last_update_datetime', 'etl_update_time', 'package'
         ], axis=1)

    tmp = data[['item_code', 'retail_price']].drop_duplicates().dropna()
    d = dict(zip(tmp['item_code'], tmp['retail_price']))
    data['retail_price'] = data['item_code'].map(d)

    data['retail_price'] = data['retail_price'].fillna(data['retail_price'].median())
    prom_types = ['x件y折', 'x元y件', '加x元多y件', '买x送y', '满x减y', 'x件减y', '第x件y折', '换购', 'is_vip', 'free_gift']
    data[prom_types] = data[prom_types].fillna(0)

    data['required_num'] = data['required_num'].fillna(1)
    data['p_rate'] = data['p_rate'].fillna(0)
    data.loc[data['amount'].isna(), 'amount'] = data.loc[data['amount'].isna(), 'retail_price']
    data.loc[data['prom_price'].isna(), 'prom_price'] = data.loc[data['prom_price'].isna(), 'retail_price']
    data.loc[data['unit_price'].isna(), 'unit_price'] = data.loc[data['unit_price'].isna(), 'prom_price']
    if data.duplicated().any() == True:
        print(f"ALERT>>>>>>>>>>>>>>>> DUPLICATES!!!")

    return data

def get_item_list(item_class):

    temp = pd.read_csv('../data/new_top500_20201217.csv',sep='|')
    all_item_list = []

    for class_list in item_class:
        
        grp_desc = class_list[0]
        dept_desc = class_list[1]
        class_desc = class_list[2]
        sbclass_desc = class_list[3]

        item_list = temp[(temp.grp_desc == grp_desc) &
                    (temp.dept_desc == dept_desc) &
                    (temp.class_desc == class_desc) &
                    (temp.sbclass_desc == sbclass_desc)].prd_id.values
        
        all_item_list += list(item_list)

    return all_item_list

def main():

    # LOAD date_feature, prom_info
    date_feature, prom_info = get_datefeature_prominfo()

    # GET ITEM LIST
    data = impala_query(f"select distinct item_code from scai.0318_0414_item_sales")
    item_list = list(data.item_code)
    exported = [i[:-4] for i in os.listdir('/app2/kevin_workspace/data03180414')]
    item_list = [i for i in item_list if i not in exported]

    print(f"START ETL FOR {len(item_list)} ITEMS, ITEM LIST: {item_list} AT {now()}")

    for item in item_list: # ITEM: INT
        try:
            print(f"__________________________________________________________________________________________________")
            print(f"IMPORTING ITEM {item} AT {now()}, PROCESS: {item_list.index(item)+1} OUT OF {len(item_list)}")

            # Load data
            print(f"Loading POG data...")
            pog_df = gen_pog_and_date(item, date_feature)
            print(f"Loading store data...")
            store_df = get_store_info()
            print(f"Loading sales data...")
            sales_df = getsaletable(item)

            # Merge data
            print(f"Merging POG, store and sales data...")
            df = pog_df.merge(store_df,on='store_id',how='inner')
            df = df.merge(sales_df,on=['item_code','store_id','day_dt'],how='left')
            print(f"Merging promotions data...")
            res_df = get_prom(df, prom_info)

            # Export data
            print(f"THE RESULT DATA HAS {len(res_df)} ROWS")
            print(f"Exporting {item}...")
            res_df.to_pickle(f"{export_path}{item}.pk",index=False)
            print(f"exported to {export_path} at {now()}")

        except Exception as error:
            print(f"ERROR: {error}")
            continue
        
    print(f"IMPORT COMPLETED AT {now()}")

if __name__ == '__main__':
    main()