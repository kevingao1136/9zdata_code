import os
import time
import pandas as pd
import numpy as np
import traceback
import warnings
warnings.simplefilter("ignore")
from multiprocessing import Pool

def now(): 
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

def cartesian_product(left, right):
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

    return date_feature

def get_prominfo(path): 
    # Reduce prom_info
    prom_info = pd.read_pickle(path)

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

    for col in ['x件y折', 'x元y件', '加x元多y件', 
        '买x送y', '满x减y','x件减y', '第x件y折', 
        '换购', 'is_vip', 'free_gift', 'unit_price',
        'required_num', 'amount', 
        'prom_price', 
        'retail_price'
        ]:
        prom_info[col] = pd.to_numeric(prom_info[col])

    prom_info['offer_start_date'] = pd.to_datetime(prom_info['offer_start_date'], errors='coerce')
    prom_info['offer_end_date'] = pd.to_datetime(prom_info['offer_end_date'], errors='coerce')
    prom_info = prom_info.dropna(subset=['item', 'offer_end_date', 'offer_start_date'])
    prom_info['offer_code'] = prom_info['offer_code'].astype(int).astype(str)

    return prom_info

def gen_pog_and_date(item, pog_path, test_period):
    """
    Parameters:
        item: item code (string)
        date_feature: feature of calendar data
    Returns:
        data: POG data
    """
    pog_data = pd.read_pickle(pog_path + item + '.pk')
    data = pog_data.merge(date_feature, on='wk_idnt', how='left')
    data.rename(columns={'item_idnt': 'item_code', 'loc_idnt': 'store_id'}, inplace=True)

    ####################### TEST DATA #######################
    item_code_df = pd.DataFrame(data=[item], columns=['item_code'])
    test_sample = cartesian_product(date_feature.loc[date_feature.day_dt.between(*test_period)], store_info)
    test_sample = test_sample.loc[test_sample.day_dt > test_sample.loc_start_dt]
    test_sample = test_sample[['day_dt', 'store_id']].drop_duplicates()
    cross_join_prom = cartesian_product(item_code_df, test_sample)
    cross_join_prom['day_dt'] = pd.to_datetime(cross_join_prom['day_dt'])
    test_data = pd.merge(cross_join_prom, date_feature, on='day_dt', how='left')
    data = pd.concat([data, test_data])
    data.drop(columns=['wk_idnt'], inplace=True)
    data.drop_duplicates(inplace=True)

    return data

def get_store_info(path):
    '''
    Returns:
        store_info - store data
    '''
    #store info
    store_info = pd.read_pickle(path)
    store_info.loc_start_dt = pd.to_datetime(store_info.loc_start_dt)
    store_info = store_info[['store_id','area_idnt','loc_wh','loc_selling_area','pds_grace','store_type', 'loc_start_dt']]
    store_info = store_info.loc[~store_info.loc_wh.isnull()]
    store_info['store_id'] = store_info['store_id'].apply(pd.to_numeric)
    if store_info['store_id'].duplicated().any() == True:
        print(f"ALERT>>>>>>>>>>>>>>DUPLICATED STORE_ID")

    return store_info

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

    data = pd.read_pickle(path + item + '.pk')
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
    if item not in list(prom_info.item.unique()): print("ITEM NOT IN PROM_INFO DATA")
    prom_info = prom_info[prom_info['item'] == item]

    # CREATE TIME RANGE DATAFRAME FOR CROSS JOIN
    time_range = pd.date_range(*date_range)
    time_merge = pd.DataFrame()
    time_merge['day_dt'] = time_range
    cross_join_prom = cartesian_product(time_merge, prom_info)
    cross_join_prom = cross_join_prom[(cross_join_prom['day_dt'] >= cross_join_prom['offer_start_date']) & (
            cross_join_prom['day_dt'] <= cross_join_prom['offer_end_date'])]

    # REMOVE DUPLICATE UNIT PRICES - TAKE THE SMALLEST
    cross_join_prom['p_rate'] = 1 - cross_join_prom['unit_price'] / cross_join_prom['retail_price']
    cross_join_prom.loc[cross_join_prom['p_rate'] < 0, 'p_rate'] = 0
    cross_join_prom = cross_join_prom.sort_values(['item', 'day_dt', 'unit_price'])
    cross_join_prom = cross_join_prom.groupby(['day_dt', 'item']).first().reset_index()
    cross_join_prom['day_dt'] = pd.to_datetime(cross_join_prom['day_dt'], errors='coerce')
    
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
    if data.unit_price.isna().mean() == 1: print("UNIT PRICE IS NULL, GET PROM NOT MERGED CORRECTLY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if data.duplicated().any() == True:
        print(f"ALERT>>>>>>>>>>>>>>>> DUPLICATES!!!")

    return data

def encrypt_data(df):

    df['store_id'], df['item_code'] = df['store_id'].astype(str), df['item_code'].astype(str)

    for feat in ['store_id','item_code','activity_desc','store_type']:

        mapper = mapper_df[mapper_df.code_type == feat][['key_id','value_id']]
        mapper.rename({'key_id':feat+'_new','value_id':feat},axis=1,inplace=True)
        df.merge(mapper, how='left', on=feat)
        df.drop(feat,axis=1,inplace=True)

    return df

def main(item):

    # Load data
    print(f"IMPORTING ITEM {item} AT {now()}")
    print(f"Loading POG data...")
    pog_df = gen_pog_and_date(
        item=item,
        pog_path='/app2/0326weekly/data/pog/',
        test_period=(test_start, test_end)
        )

    print(f"Loading sales data...")
    sales_df = getsaletable(item=item, path='/app2/0326weekly/data/sales/')

    print(f"MERGING DATA...")
    df = pog_df.merge(store_info,on='store_id',how='inner')
    df.item_code, sales_df.item_code = df.item_code.astype(str), sales_df.item_code.astype(str)
    df = df.merge(sales_df,on=['item_code','store_id','day_dt'],how='left')
    res_df = get_prom(item=item, data=df, prom_info=prom_info, date_range=(test_start, test_end))
    res_df = res_df[res_df.day_dt >= '2019-01-01']

    res_df = encrypt_data(res_df)

    # Export data
    print(f"THE RESULT DATA HAS {len(res_df)} ROWS")
    print(f"Exporting {item} at {now()}...")
    print(f'FEATURES USED: {res_df.columns}')
    res_df.columns = range(len(res_df.columns))
    res_df.to_pickle(f"{export_path}{item}.pk.gz",compression='gzip')
    print(f"exported to {export_path} at {now()}")

#* DEFINE EXPORT PATH
export_path = '/app2/0326weekly/kevin_workspace/data_test10/'
log_path = '/app2/0326weekly/kevin_workspace/log_test10/'

# DEFINE TIME
test_start, test_end = '2021-04-01', '2021-05-12'

#* GET ITEM LIST
item_list = [i[:i.index('.')] for i in os.listdir('/app2/0326weekly/data/sales/')]

# LOAD FIXED DATA
print('Loading date_feature...')
date_feature = get_datefeature(path='/app2/0326weekly/data/date_feature.csv')
print('Loading prom_info...')
prom_info = get_prominfo(path='/app2/0326weekly/data/promotion_info.pk')
print('Loading store_df...')
store_info = get_store_info(path='/app2/0326weekly/data/store_info.pk')
print('Loading mapper_df...')
mapper_df = pd.read_pickle('/app2/0326weekly/data/feature_map.pk')
mapper_df.code_type.replace({'loc_idnt':'store_id','item_idnt':'item_code'},inplace=True)

# MAKE IT A LIST OF STRINGS
print(f"START ETL FOR {len(item_list)} ITEMS, ITEM LIST: {item_list} AT {now()}")

for item in item_list:
    # try:
    main(item)

    # except:
    #     traceback.print_exc()

print(f"IMPORT COMPLETED AT {now()}")