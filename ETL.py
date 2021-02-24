import os
import time
import pandas as pd
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
import traceback
import warnings
warnings.simplefilter("ignore")

def timefunc(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(f"RUN TIME: {round((end - start),4) / 60} MINUTES")
        return result
    return inner

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

def get_datefeature(path):
    """
    Returns:
        date_feature:   day features - mday, yday, holidays
        prom_info:      promotion features data
    """
    # Reduce date_feature
    date_feature = pd.read_csv(path, parse_dates=['day_dt'])
    date_feature.drop(['day_of_mth','day_of_qtr','day_of_yr','wk_of_mth','wk_of_qtr','wk_of_yr'],axis=1,inplace=True)
    date_feature, _ = reduce_mem_usage(date_feature)

    return date_feature

def get_prominfo(path): 
    # Reduce prom_info
    prom_info = pd.read_csv(path)

    prom_info.rename({'label_001':'x件y折',
                    'label_002':'x元y件',
                    'label_003':'加x元多y件',
                    'label_004':'买x送y',
                    'label_005':'满x减y',
                    'label_006':'x件减y',
                    'label_007':'第x件y折',
                    'label_008':'换购'
                    },axis=1,inplace=True)

    prom_info = prom_info[['item', 'offer_code', 'offer_start_date', 'offer_end_date',
                        'osd_type', 'outputs', 'x件y折', 'x元y件', '加x元多y件', '买x送y', '满x减y',
                        'x件减y', '第x件y折', '换购', 'is_vip', 'free_gift', 'unit_price',
                        'required_num', 'amount', 'prom_price', 'retail_price','flag']]
    prom_info['offer_start_date'] = pd.to_datetime(prom_info['offer_start_date'], errors='coerce')
    prom_info['offer_end_date'] = pd.to_datetime(prom_info['offer_end_date'], errors='coerce')
    prom_info = prom_info.dropna(subset=['item', 'offer_end_date', 'offer_start_date'])
    prom_info['offer_code'] = prom_info['offer_code'].astype(int).astype(str)
    prom_info['item'] = prom_info['item'].astype(int)
    prom_info, _ = reduce_mem_usage(prom_info)

    return prom_info

def gen_pog_and_date(item, date_feature, pog_path, test_path):
    """
    Parameters:
        item: item code (string)
        date_feature: feature of calendar data
    Returns:
        data: POG data
    """
    pog_data = impala_query(f"select ITEM_IDNT, WK_IDNT, LOC_IDNT from {pog_path} where ITEM_IDNT = '{item}'")
    data = pog_data.merge(date_feature, on='wk_idnt', how='left')
    data.rename(columns={'item_idnt': 'item_code', 'loc_idnt': 'store_id'}, inplace=True)

    ####################### TEST DATA #######################
    item_code_df = pd.DataFrame(data=[item], columns=['item_code'])
    test_data = pd.read_csv(test_path) # TEST PERIOD AND ALL STORE ID
    cross_join_prom = cartesian_product_basic(item_code_df, test_data)
    cross_join_prom['day_dt'] = pd.to_datetime(cross_join_prom['day_dt'])
    test_data = pd.merge(cross_join_prom, date_feature, on='day_dt', how='left')
    data = pd.concat([data, test_data])
    data.drop(columns=['wk_idnt'], inplace=True)
    data['item_code'] = data['item_code'].apply(pd.to_numeric)
    data['day_dt'] = pd.to_datetime(data['day_dt'])
    data.drop_duplicates(inplace=True)

    return data

def get_store_info(path):
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
    from {path}
    where is_e_store='N' and loc_type_cde='S' and loc_end_dt is null and area_idnt in ('1','2','3','4')
    """)

    store_info['store_id'] = store_info['store_id'].apply(pd.to_numeric)
    if store_info['store_id'].duplicated().any() == True:
        print(f"ALERT>>>>>>>>>>>>>>DUPLICATED STORE_ID")

    return store_info

def get_product_info(path):

    #product info
    product_info = impala_query(f"""
    select item_idnt as item_code,
            item_desc,
            sbclass_desc,
            item_brand
    from {path}
    """)

    product_info['item_code'] = product_info['item_code'].apply(pd.to_numeric)
    return product_info

def get_outlier(df,store,item,work_day,quantity):
    
    start_dt = work_day - pd.Timedelta(days=7)
    end_dt = work_day + pd.Timedelta(days=7)
    tmp = df[(df.store_id==store)&(df.item_code==item)&(df.day_dt>=start_dt)&(df.day_dt<=end_dt)&(df.ttl_quantity<=200)]
    all_quantity = tmp.ttl_quantity.sum()
    if quantity / all_quantity > 1:
        return all_quantity/len(tmp)
    else:
        return quantity

def getsaletable(item, path):

    data = impala_query(
    f"select * from {path} where item_code = '{str(item)}'"
    )

    data['day_dt'] = pd.to_datetime(data['day_dt'])
    data[['item_code','store_id','ttl_quantity','avg_amountofsales','ttl_amt_payment']] = data[['item_code','store_id','ttl_quantity','avg_amountofsales','ttl_amt_payment']].apply(pd.to_numeric)
    outlier_df = data[data.ttl_quantity>=200]
    if len(outlier_df) == 0:
        return data
    else:
        outlier_df['sale_qty2'] = outlier_df.apply(lambda x:get_outlier(data,x.store_id,x.item_code,x.day_dt,x.ttl_quantity),axis=1)
        outlier_df['sale_qty2'] = outlier_df['sale_qty2'].fillna(0.0).apply(lambda x:int(np.ceil(x)))
        outlier_df = outlier_df.drop(['ttl_quantity','avg_amountofsales','ttl_amt_payment'],axis=1)
        data = pd.merge(data,outlier_df,how='left',on=['store_id','day_dt','item_code'])
        data['sale_qty2'] = data['sale_qty2'].fillna(data['ttl_quantity'])
        data = data.drop(['ttl_quantity'],axis=1).rename({'sale_qty2':'ttl_quantity'},axis=1)
        if data[['store_id','item_code','day_dt']].duplicated().any() == True:
            print(f"ALERT>>>>>>>>>>>>>>> DUPLICATES!!!")
        
        return data

def get_prom(item, data, prom_info, date_range):
    """
    Parameters:
        item: item code (string)
        data: merged data from previous steps
        prom_info: prom_info data
        date_range: 2 dates for date range (tuple)
    Returns:
        data: data with promotion information merged to it
    """
    # GET PROM INFO
    prom_info = prom_info[prom_info['item'] == int(item)]

    # CREATE TIME RANGE DATAFRAME FOR CROSS JOIN
    time_range = pd.date_range(*date_range)
    time_merge = pd.DataFrame()
    time_merge['day_dt'] = time_range
    cross_join_prom = cartesian_product_basic(time_merge, prom_info)
    cross_join_prom = cross_join_prom[(cross_join_prom['day_dt'] >= cross_join_prom['offer_start_date']) & (
            cross_join_prom['day_dt'] <= cross_join_prom['offer_end_date'])]

    # REMOVE DUPLICATE UNIT PRICES - TAKE THE SMALLEST
    cross_join_prom['p_rate'] = 1 - cross_join_prom['unit_price'] / cross_join_prom['retail_price']
    cross_join_prom.loc[cross_join_prom['p_rate'] < 0, 'p_rate'] = 0
    cross_join_prom = cross_join_prom.sort_values(['item', 'day_dt', 'unit_price'])
    cross_join_prom = cross_join_prom.groupby(['day_dt', 'item']).first().reset_index()
    cross_join_prom['day_dt'] = pd.to_datetime(cross_join_prom['day_dt'], errors='coerce')
    cross_join_prom['item'] = cross_join_prom['item'].astype(int)

    # MERGE CROSS JOIN PROM AND DATA
    data = data.merge(cross_join_prom, left_on=['day_dt', 'item_code'], right_on=['day_dt', 'item'], how='left')
    data = data.drop(['item','offer_start_date', 'offer_end_date','outputs'], axis=1)

    # IMPUTE DATA
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

@timefunc
def main(item):

    # Load data
    print(f"Loading POG data at {now()}...")
    pog_df = gen_pog_and_date(item=item,
                            date_feature=date_feature,
                            pog_path='scai.0318_0414_pog',
                            test_path='assist_data/test_sample.csv')

    print(f"Loading store data at {now()}...")
    store_df = get_store_info(path='ods_sc.dim_organization')

    print(f"Loading product data at {now()}...")
    product_df = get_product_info(path='ods_sc.dim_product')

    print(f"Loading sales data at {now()}...")
    sales_df = getsaletable(item=item, path='scai.0318_0414_item_sales')

    print(f"Merging POG, store, product, and sales data at {now()}...")
    df = pog_df.merge(store_df,on='store_id',how='inner')
    df = df.merge(sales_df,on=['item_code','store_id','day_dt'],how='left')
    df = df.merge(product_df,on=['item_code'],how='left')

    print(f"Merging promotions data at {now()}...")
    res_df = get_prom(item=item, data=df, prom_info=prom_info, date_range=('2019-01-01', '2021-06-30'))
    assert res_df.unit_price.isna().mean() != 1, "UNIT PRICE IS NULL, GET PROM NOT MERGED CORRECTLY."

    # Export data
    print(f"THE RESULT DATA HAS {len(res_df)} ROWS")
    print(f"Exporting {item} at {now()}...")
    res_df.to_pickle(f"{export_path}{item}.pk")
    print(f"exported to {export_path} at {now()}")

if __name__ == '__main__':

    #* DEFINE EXPORT PATH
    # export_path = '/app2/kevin_workspace/NA/'
    export_path = './'

    #* GET ITEM LIST
    data = impala_query(f"select distinct item_code from scai.0318_0414_item_sales")
    item_list = list(data.item_code)[:2]
    # exported = [i[:-3] for i in os.listdir('/app2/kevin_workspace/data0304')]
    # item_list = [i for i in item_list if i not in exported]

    # LOAD date_feature, prom_info
    date_feature, prom_info = get_datefeature(path='../data/date_feature.csv'), get_prominfo(path='assist_data/prom_data.csv')

    # MAKE IT A LIST OF STRINGS
    item_list = [str(item) for item in item_list]
    print(f"START ETL FOR {len(item_list)} ITEMS, ITEM LIST: {item_list} AT {now()}")

    # MAJOR FOR LOOP
    for item in item_list:

        print(f"__________________________________________________________________________________________________")
        print(f"IMPORTING ITEM {item} AT {now()}, PROCESS: {item_list.index(item)+1} OUT OF {len(item_list)}")
        
        try:
            main(item)
        except:
            traceback.print_exc()
            continue

    print(f"IMPORT COMPLETED AT {now()}")

# df = impala_query("""
# select a.wk_idnt,
#         count(distinct loc_idnt) store_cnt
# from ods_sc.dim_pog_loc_hist a 
# join ods_sc. dim_pog_prod_hist b 
# on a.key_id = b.key_id and a.wk_idnt = b.wk_idnt
# where item_idnt = '100659714'
# group by 1
# order by 1
# """)

# print(df[df.wk_idnt > 202101].store_cnt.describe())