import pandas as pd
import time
import os
# import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
from catboost import CatBoostRegressor, Pool as catpool
from multiprocessing import Pool
import contextlib
import traceback

train_end = '2021-01-01'
v1_start = train_end
v1_end = '2021-02-01'
v2_start = v1_end
v2_end = '2021-03-01'

# GET ITEM LIST
item_list = list(pd.read_csv('/app/python-scripts/kevin_workspace/my_item.csv').item_code.astype(str))
exported = [f[:f.index('.csv')] for f in os.listdir("/app/python-scripts/kevin_workspace/outputs/lowsales_output_parallel/data/") if f.endswith('.csv')]
item_list = [i for i in item_list if i not in exported]
print(f"{len(item_list)} ITEMS, ITEM LIST: {item_list}")

log_path = f"/app/python-scripts/kevin_workspace/outputs/lowsales_output_parallel/log/"
export_path = f"/app/python-scripts/kevin_workspace/outputs/lowsales_output_parallel/data/"

def now():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

def impala_query(sql:str):
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

def get_sales_df(item:str):

    sales_df = impala_query(f"""
        select 
        fsc.prd_id as item_code
        ,to_date(transaction_date) as day_dt
        ,nvl(deliver_storeid,nvl(cast(order_store as string),cast(loc_idnt as string))) as store_id
        ,sum(quantity) as ttl_quantity
        ,sum(amountofsales)/sum(quantity) as avg_amountofsales
        ,sum(amt_payment) as ttl_amt_payment 
        from ods_sc.fct_sales_center fsc 
        where quantity < 50 and amt_payment < 3000 and quantity > 0 and channel in (select channel from scai.scai_fct_sales_center_channel)
        and nvl(deliver_storeid,nvl(cast(order_store as string),cast(loc_idnt as string))) in (select loc_idnt from ods_sc.dim_organization where area_idnt in ('1','2','3','4'))
        and prd_id = '{item}'
        group by 1,2,3
        """)

    sales_df = sales_df.groupby(['item_code','day_dt'],as_index=False).ttl_quantity.sum()
    sales_df['day_dt'] = pd.to_datetime(sales_df['day_dt'])

    train_start = max(sales_df.day_dt.min(), pd.to_datetime('2020-04-01'))
    train_range = pd.DataFrame({'day_dt':pd.date_range(train_start,v2_end)})
    train_range['item_code'] = item
    df = train_range.merge(sales_df,how='left',on=['item_code','day_dt'])
    df['ttl_quantity'] = df.ttl_quantity.fillna(0)

    return df, train_start

def get_unit_price(item:str):
    
    unit_price = impala_query(f"""
    with tmp as (
    select work_date as day_dt,prd_id as item_code,area_idnt,round(amt_payment/quantity,2) as unit_price2,sum(quantity) as ttl_qty
    from ods_sc.fct_sales_center fsc 
    join ods_sc.dim_organization dr 
        on cast(dr.loc_idnt as string) = cast(fsc.loc_idnt as string)
    where amt_payment < 3000 and quantity<50 and quantity>0 and area_idnt in ('1','2','3','4') 
    and prd_id = '{item}'
    group by 1,2,3,4  
    )
    ,tmp3 as (select day_dt
    ,item_code
    ,area_idnt
    ,unit_price2
    ,row_number() over(partition by day_dt,item_code,area_idnt order by ttl_qty desc) as ind from tmp 
    )
    select day_dt,item_code,area_idnt,unit_price2 from tmp3
    where ind = 1
    """)

    unit_price.rename({'unit_price2':'unit_price'},axis=1,inplace=True)
    unit_price['unit_price'] = unit_price['unit_price'].astype(float)
    unit_price['day_dt'] = pd.to_datetime(unit_price['day_dt'])
    unit_price = unit_price.groupby(['item_code','day_dt'],as_index=False).unit_price.mean()
    
    return unit_price

def train_model(X_train, X_test, y_train, y_test):

    train_pool = catpool(data=X_train, label=y_train)
    val_pool = catpool(data=X_test, label=y_test)

    # CatBoost model
    model = CatBoostRegressor(
        iterations=500,
        # learning_rate=0.1,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=False
        )

    model.fit(
        X=train_pool,
        eval_set=val_pool,
        early_stopping_rounds=20
    )

    return model

def main(item):

    try_again = True

    with open(f"{log_path}{item}.log", "w") as train_log, contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(train_log):
        while try_again:
            try:
                print(f"START TRAINING FOR ITEM {item} AT {now()} _______________________________________________________________________________")
                # GET SALES DF
                sales_df, train_start = get_sales_df(item)
                if sales_df.day_dt.min() > pd.to_datetime(train_end):
                    print(f"NO TRAINING DATA >>>>>>>>>>>>>>>>>>>>>>>>")
                
                print(f"TRAIN START: {train_start}, TRAIN END: {train_end}")

                # GET UNIT PRICE
                unit_price = get_unit_price(item)
                train_val_df = sales_df.merge(unit_price,how='left',on=['item_code','day_dt'])
                train_val_df['unit_price'] = train_val_df['unit_price'].fillna(train_val_df['unit_price'].max())
                train_val_df['weekday'] = train_val_df.day_dt.dt.dayofweek
                train_val_df = train_val_df[~train_val_df.day_dt.between('2020-01-01','2020-04-01')]

                # GET DATE FEATURE
                date_feature = pd.read_csv('../data/date_feature.csv',usecols=['day_dt','day_type','is_5th'],parse_dates=['day_dt'])
                train_val_df = train_val_df.merge(date_feature,how='left',on='day_dt')
                train_val_df.fillna(value={'day_type':0, 'is_5th':0},inplace=True)

                # DEFINE FEATURES
                X_features = ['unit_price','weekday','day_type','is_5th']
                y_feature = ['ttl_quantity']

                # DEFINE X AND y
                X_train, X_test, y_train, y_test = (train_val_df[train_val_df.day_dt.between(train_start,train_end)][X_features],
                                                    train_val_df[train_val_df.day_dt.between(v1_start,v2_end)][X_features],
                                                    train_val_df[train_val_df.day_dt.between(train_start,train_end)][y_feature],
                                                    train_val_df[train_val_df.day_dt.between(v1_start,v2_end)][y_feature]
                                                    )
                # PRINT OUT TARGET INFORMATION
                print('TTL_QUANTITY IN TRAIN:')
                print(y_train.ttl_quantity.unique())
                if y_train.ttl_quantity.nunique() <= 1:
                    print(f"TRAIN DATA HAS 1 UNIQUE QTY, SKIP ITEM >>>>>>>>>>>>>>>>")
                    return None
                print('TTL_QUANTITY IN VALIDATION:')
                print(y_test.ttl_quantity.unique())

                model = train_model(X_train, X_test, y_train, y_test)

                val_df = train_val_df[train_val_df.day_dt.between(v1_start,v2_end)].copy()
                val_df['y_pred'] = model.predict(val_df[X_features])

                v1_evaluate = val_df[val_df.day_dt.between(v1_start, v1_end)]
                v2_evaluate = val_df[val_df.day_dt.between(v2_start, v2_end)]

                # RESULT_DF TO STORE PREDICTION RESULTS
                item_result = pd.DataFrame({
                                        'item_code':[item],
                                        'v1_ttl_quantity':[v1_evaluate.ttl_quantity.sum()],
                                        'v1_y_pred':[v1_evaluate.y_pred.sum()],
                                        'v2_ttl_quantity':[v2_evaluate.ttl_quantity.sum()],
                                        'v2_y_pred':[v2_evaluate.y_pred.sum()],
                                        })

                print(f"PREDICTION RESULTS -")
                print(item_result)
                item_result.to_csv(f"{export_path}{item}.csv",index=False)
                print(f"ITEM RESULT EXPORTED TO {export_path} AT {now()}")

            except:
                error_message = traceback.format_exc()
                print(error_message)
                if 'Memory is likely oversubscribed' in error_message:
                    time.sleep(10)
                    print(f"TRY TO CONNECT TO IMPALA AGAIN IN 10 SECONDS...")
                    try_again=True
                else:
                    try_again=False
            else: 
                try_again=False # NO ERROR, STOP THE WHILE LOOP

# RUN MULTIPROCESSING
pool = Pool(4)
pool.map(main,item_list)
pool.close()
pool.join()