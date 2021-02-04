import os
import logging
from multiprocessing import Pool as multi_pool
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt

def now():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_data(df):
    """
    remove 1,2,3 months, clean feature values
    """

    # REMOVE months 1,2,3 in 2020
    df['year_month'] = df.day_dt.dt.strftime('%Y-%m')
    df = df.query("year_month not in ('2020-01','2020-02','2020-03')")
    df.drop(['year_month'],axis=1,inplace=True)

    # clean data
    df['pds_location_type_en'].replace({'Inmall':'inmall',
                                    'Inline+inmall':'inline+inmall',
                                    'Inmall+Inline':'inline+inmall',
                                    'Inmall+inline':'inline+inmall',
                                    'inmall+inline':'inline+inmall',
                                    'Inline':'inline',
                                    'Inline+Inmall':'inline+inmall',
                                    ' Inline+inmall':'inline+inmall'}, inplace=True)

    df.columns = pd.Series(df.columns).replace({'x件y折':'prom0',
                    'x元y件':'prom1',
                    '加x元多y件':'prom2',
                    '买x送y':'prom3',
                    '满x减y':'prom4',
                    'x件减y':'prom5',
                    '第x件y折':'prom6',
                    '换购':'prom7'}).values

    df.pds_floor_type.replace({
                'G/F+2/F':'G/F+1/F',
                'G/F+4/F':'G/F+1/F',
                'G/F+B/2':'B/1+G/F',
                '1/F+B/2': '1/F', 
                '2/F+B/3':'2/F',
                'B1/F':'B1',
                'G/F+B/1':'B/1+G/F',
                'B1':'B/1'
                },inplace=True)

    df['pds_grace'].replace({'高级':'Premium',
                            '标准':'Standard',
                            '经济':'Economy'
                            }, inplace=True)

    return df

def impute_data(df):
    """
    Impute missing values
    """
    # FILL NONE
    for col in ['activity_desc',
                'pds_grace',
                'pds_store_segmentation',
                'store_type',
                'city_tier',
                'pds_floor_type',
                'pds_location_type_en',
                'osd_type']:
        df[col].fillna('None',inplace=True)

    # FILL NULL VALUES
    df['ttl_quantity'].fillna(0,inplace=True)
    df['pds_mtg_type'].fillna('Others(其它',inplace=True)
    df['loc_selling_area'] = df.groupby(['pds_mtg_type','pds_location_type_en'])['loc_selling_area'].transform(lambda x: x.fillna(x.median()))
    df['p_rate'] = df.unit_price / df.retail_price # CORRECT P_RATE
    df['isweekend'] = df.weekday.apply(lambda x: 1 if x in (5,6) else 0)

    return df

def select_features(features, df):
    """
    features: list of features
    """
    return df[features]

def label_encoder(df):

    le = LabelEncoder()
    cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        df[col] = df[col].astype(str)
    # print(f"LABEL ENCODED COLS: {cat_cols}")

    return df

def agg_results(val_start, val_end, df):

    df = df[df.day_dt.between(val_start, val_end)]
    
    # agg by item, wh
    item_wh = df.groupby(['item_code','loc_wh'],as_index=False)['ttl_quantity','y_pred'].sum()
    item_wh['ratio'] = item_wh['ttl_quantity'] / item_wh['y_pred']
    item_wh['y_pred'] = item_wh['y_pred'].apply(lambda x: round(x,4))
    item_wh['ratio'] = item_wh['ratio'].apply(lambda x: round(x,4))
    item_wh['unit_price'] = df['unit_price'].mean()
    
    return item_wh

def load_data(item_path, train_start, train_end, test_start, test_end):
    """
    Parameters:
        item: item code
        train_start, train_end, test_start, test_end
    Returns:
        train_df: training data including validation data
        test_df: test data
    """
    df = pd.read_csv(item_path, parse_dates=['day_dt'])

    # GET TRAIN DATA AND IMPUTE UNIT PRICE
    train_df = df[df.day_dt.between(train_start, train_end)]
    train_df = train_df.merge(unit_price, on=['item_code','area_idnt','day_dt'], how='left')
    train_df['unit_price2'].fillna(train_df['unit_price'],inplace=True)
    train_df.drop('unit_price',axis=1,inplace=True)
    train_df.rename({'unit_price2':'unit_price'},axis=1,inplace=True)

    # GET TEST DATA
    test_df = df[df.day_dt.between(test_start, test_end)]

    return train_df, test_df

def train_model(X_train, X_test, y_train, y_test):

    cat_features = ['store_type','activity_desc']

    train_pool = Pool(data=X_train, 
                    label=y_train, 
                    # cat_features=cat_features, 
                    # weight=np.exp(np.linspace(0,10,len(X_train)))
                    )

    val_pool = Pool(data=X_test, 
                    label=y_test, 
                    # cat_features=cat_features
                    )

    # CatBoost model
    model = CatBoostRegressor(
                            iterations=1000,
                            loss_function='RMSE',
                            # loss_function='Tweedie:variance_power=1.5',
                            eval_metric='RMSE',
                            verbose=False
                            )

    model.fit(train_pool,
            eval_set=val_pool,
            # use_best_model=False
            )

    return model

def train_model_w_validation(X, y):

    cat_features = ['store_type','activity_desc']

    train_pool = Pool(data=X, 
                    label=y, 
                    # cat_features=cat_features, 
                    # weight=np.exp(np.linspace(0,10,len(X_train)))
                    )

    # CatBoost model
    model = CatBoostRegressor(
                            iterations=1000,
                            loss_function='RMSE',
                            # loss_function='Tweedie:variance_power=1.5',
                            eval_metric='RMSE',
                            verbose=False
                            )

    model.fit(train_pool)

    return model

def show_results(df, log_func):

    log = log_func()
    log(f"0-50%: {np.mean([1 if r<=0.5 else 0 for r in df.ratio])}")
    log(f"50-80%: {np.mean([1 if r>=0.5 and r<=0.8 else 0 for r in df.ratio])}")
    log(f"80-100%: {np.mean([1 if r>=0.8 and r<=1 else 0 for r in df.ratio])}")
    log(f"100-120%: {np.mean([1 if r>=1 and r<=1.2 else 0 for r in df.ratio])}")
    log(f"120-inf%: {np.mean([1 if r>=1.2 else 0 for r in df.ratio])}")
    log(f"65-120%: {np.mean([1 if r>=0.65 and r<=1.2 else 0 for r in df.ratio])}")

def plot_pred_results(item_code, train, val, test, img_path):
    '''
    Parameters:
        train - train dataset, required features: day_dt, ttl_quantity, y_pred, unit_price
        val - validation dataset (v1 + v2), required features: day_dt, ttl_quantity, y_pred, unit_price
        test - test dataset, required features: day_dt, y_pred, unit_price
        img_path - image export path
    '''
    # PLOT AX1 - TTL_QUANTITY & Y_PRED
    fig, ax1 = plt.subplots(figsize=(30,12))
    ax1.set_ylabel('ttl_quantitiy',fontsize=20)
    ax1.plot(train.groupby(['day_dt']).ttl_quantity.sum(),label='train_y_true')
    ax1.plot(train.groupby(['day_dt']).y_pred.sum(),label='train_y_pred')
    ax1.plot(val.groupby(['day_dt']).ttl_quantity.sum(),label='Validation_y_true',color='y')
    ax1.plot(val.groupby(['day_dt']).y_pred.sum(),label='Validation_y_pred',color='r')
    ax1.plot(test.groupby(['day_dt']).y_pred.sum(),label='test_y_pred')
    ax1.axvline(x=val.day_dt.min(),color='k',ls='--',linewidth=3)

    # ADD AX2 - UNIT PRICE
    ax2=ax1.twinx()
    ax2.set_ylabel('unit_price',fontsize=20)
    ax2.set_ylim([0,max(train.unit_price.max(),val.unit_price.max())+5])
    ax2.plot(train.groupby(['day_dt']).unit_price.mean(),'--',label='unit_price',color='k')
    ax2.plot(val.groupby(['day_dt']).unit_price.mean(),'--',color='k')
    ax2.plot(test.groupby(['day_dt']).unit_price.mean(),'--',color='k')
    fig.legend(fontsize=15)
    plt.title(f"PREDICTION RESULTS FOR ITEM {item_code}",size=20)
    plt.setp(ax1.get_xticklabels(),fontsize=15)
    plt.setp(ax1.get_yticklabels(),fontsize=15)
    fig.tight_layout()
    plt.grid()
    plt.savefig(img_path)

def export_results(item_code, train, val1, val2, test, img_path, result_data_path, meta_path, log_path, log_func):
    '''
    Parameters:
        item_code - item code
        train - train dataset
        val1 - v1 dataset
        val2 - v2 dataset
        test - test dataset
    
    Export to log_path, result_data_path, meta_path, img_path
    '''
    # CREATE RESULT DATA -----------------------------------------------------------------------------------------------------------------
    # 0 IF TEST PREDICTION < 0
    test['y_pred'] = test['y_pred'].apply(lambda x: 0 if x < 0 else x)

    # GET STORE COUNT
    test_store_cnt = test.store_id.nunique()

    # SET LOG FUNCTION
    log = log_func

    # IF DUPLICATE STORES IN TEST DATA, SKIP ITEM
    if test_store_cnt == test[['day_dt','item_code','store_id']].drop_duplicates().store_id.nunique(): # IF NO DUPLICATES
        log(f"THERE ARE {test_store_cnt} STORES IN TEST DATA") # LOG STORE COUNT
        if np.sum(test.groupby(['loc_wh']).y_pred.sum() < 100) > 0: # IF ANY DC IS LESS THAN 100, SKIP ITEM
            log(f"ALERT>>>>>>>>>>>>>>>>>> {np.sum(test.groupby(['loc_wh']).y_pred.sum() < 100)} DC HAS TTL_QUANTITY < 100!!! SKIP ITEM...")
        else:
            log(test.groupby(['loc_wh']).y_pred.sum())
    else: # IF DUPLICATE
        log(f"ALERT>>>>>>>>>>>>>>>>>> DUPLICATED STORES IN TEST DATA!!! SKIP ITEM...")

    # CREATE RESULT DATA
    train_val = pd.concat([train, val1, val2])
    train_val['istest'] = 0
    test['istest'] = 1
    result_data = pd.concat([train_val, test])
    result_data['model_id'] = "SC_A5_LGBM_V1"
    result_data = result_data[['model_id','item_code','store_id','day_dt','ttl_quantity','y_pred','istest']]
    result_data = result_data[result_data.istest==1]
    assert result_data.istest.sum() > 0, "NO TEST RESULT DATA!"
    result_data.to_csv(result_data_path,index=False)

    # CREATE META DATA --------------------------------------------------------------------------------------------------------------------
    # CREATE META DATA DICT
    meta_dict = {'model_id':"SC_A5_LGBM_V1",
                'scenario':'TopSales',
                'aid':'A5',
                'model_time':now(),
                'log_path':log_path,
                'result_path':result_data_path,
                'img_path':img_path,
                'v1_start_time':val1.day_dt.min().strftime("%Y-%m-%d"),
                'v1_end_time':val1.day_dt.max().strftime("%Y-%m-%d"),
                'v2_start_time':val2.day_dt.min().strftime("%Y-%m-%d"),
                'v2_end_time':val2.day_dt.max().strftime("%Y-%m-%d"),
                'package':np.nan,
                'test_store_cnt':test_store_cnt}


    # GET VALIDATION ACCURACY
    accuracy_list = []
    for data in (val1, val2):
        store_avg_df = data.groupby('store_id').ttl_quantity.sum().reset_index()
        store_avg_df['avg_qty'] = store_avg_df['ttl_quantity'] / 4
        store_avg_df = store_avg_df.drop('ttl_quantity',axis=1)
        data = data.merge(store_avg_df, how='left', on='store_id')
        data['ttl_quantity'] = data['ttl_quantity'].fillna(0)
        data['y_diff'] = data['ttl_quantity'] - data['y_pred']
        data['isR'] = data.apply(lambda x: 1 if min([-1,x.avg_qty*-0.5]) <= x.y_diff <= max([1,x.avg_qty]) else 0,axis=1)
        accuracy = data.isR.mean()
        accuracy_list.append(accuracy)
    meta_dict['v1_accuracy'] = round(accuracy_list[0],4)
    meta_dict['v2_accuracy'] = round(accuracy_list[1],4)


    # GET VALIDATION SELL THRU
    sell_thru_list = []
    for data in (val1, val2):
        sell_thru = data.ttl_quantity.sum() / data.y_pred.sum()
        sell_thru_list.append(sell_thru)
    meta_dict['v1_sell_thru'] = round(sell_thru_list[0],4)
    meta_dict['v2_sell_thru'] = round(sell_thru_list[1],4)

    # MERGE VAL1 AND VAL2 & PLOT RESULTS
    all_val = pd.concat([val1, val2])
    plot_pred_results(item_code=item_code, train=train, val=all_val, test=test, img_path=img_path)

    # CONVERT DICT TO DATAFRAME AND EXPORT METADATA
    for k, v in meta_dict.items():
        meta_dict[k] = [v]

    meta_data = pd.DataFrame(meta_dict)
    meta_cols = ['model_id','scenario','aid','model_time','log_path','result_path',
                'img_path','v1_start_time','v1_end_time','v1_accuracy','v1_sell_thru',
                'v2_start_time','v2_end_time','v2_accuracy','v2_sell_thru','package','test_store_cnt']
    meta_data = meta_data[meta_cols]
    meta_data.to_csv(meta_path,index=False)

def main(item):

    #* DEFINE PATHS
    data_path = f"/app2/kevin_workspace/data03180414/"
    img_path = f"/app/python-scripts/kevin_workspace/output_0304/train_img/{item}.png"
    log_path = f"/app/python-scripts/kevin_workspace/output_0304/log/train_{item}.log"
    result_data_path = f"/app/python-scripts/kevin_workspace/output_0304/result_data/{item}_A5_result_data.csv"
    meta_path = f"/app/python-scripts/kevin_workspace/output_0304/meta_data/{item}_A5_meta_data.csv"

    #* SET LOG OPTIONS
    logging.basicConfig(filename=log_path, format='%(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log = logger.info

    #* SET DATES
    train_start, train_end ='2019-01-01', '2021-03-18'
    test_start, test_end ='2021-03-18', '2021-04-14'
    v1_start, v1_end = '2020-12-16', '2021-01-07'
    v2_start, v2_end = '2021-01-08', '2021-01-27'

    try:
        log(f"_________________________________________________________________________________________________________________________________________________")
        log(f"STARTING TRAINING ITEM {item} AT {now()}")
        log(f"Loading data...")
        raw_train_df, raw_test_df = load_data(f"{data_path}{item}.csv", 
                    train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)

        # LOG TIME RANGE
        log(f"train_df: {raw_train_df.day_dt.min()} to {raw_train_df.day_dt.max()}")
        log(f"test_df: {raw_test_df.day_dt.min()} to {raw_test_df.day_dt.max()}")

        # IF EMPTY, CONTINUE WITH NEXT ITEM
        try:
            assert len(raw_train_df) > 0, "EMPTY TRAIN DATA"
            assert len(raw_test_df) > 0, "EMPTY TEST DATA"
        except AssertionError as error:
            log(f"item {item} has error: {error}")
            return # STOP FUNCTION
        
        # MAKE NEW COPIES OF DATAFRAMES
        train_df, test_df = raw_train_df.copy(), raw_test_df.copy()

        # ADD LUNAR NEW YEAR FEATURE
        train_df = train_df.merge(lunar, how='left', on='day_dt')
        test_df = test_df.merge(lunar, how='left', on='day_dt')

        # ADD PROMOTION SCHEDULE FEATURE
        train_df = train_df.merge(prom_schedule, how='left', on='day_dt')
        test_df = test_df.merge(prom_schedule, how='left', on='day_dt')

        # DEFINE FEATURES TO SELECT
        index_features = ['day_dt','store_id','item_code','loc_wh']
        
        # X_features = [
        #             'activity_desc', 
        #             'day_type', 'last_day_type','month', 'mday', 'weekday', 'is_5th', 'dis_spring', 'schedule_ix',
        #             'distt_idnt', 'regn_idnt', 'area_idnt', 'mkt_idnt', 'store_type', 'loc_selling_area', 
        #             'pds_location_type_en','pds_mtg_type', 'pds_store_segmentation', 'pds_grace', 
        #             'pds_floor_type', 'city_tier', 'is_intracity_dlvr_store', 'is_central_store',
        #             'p_rate', 'unit_price', 'osd_type'
        #             ]

        X_features = ['day_type','month','weekday','isweekend','store_type','activity_desc','year','is_5th',
                    'prom0','prom1','prom2','prom3','prom4','prom5','prom6','prom7', 'is_vip','free_gift',
                    'required_num','unit_price','p_rate','dis_spring','schedule_ix']

        y_feature = ['ttl_quantity']

        log(f"SELECTED FEATURES: {X_features}")

        log(f"Cleaning train data...")
        train_df = clean_data(train_df)
        train_df = impute_data(train_df)
        train_df = select_features(index_features+X_features+y_feature, train_df)
        train_df = label_encoder(train_df)

        log(f"Cleaning test data...")
        test_df = clean_data(test_df)
        test_df = impute_data(test_df)
        test_df = select_features(index_features+X_features+y_feature, test_df)
        test_df = label_encoder(test_df)

        # SPLIT TRAIN DATA INTO TRAIN AND VALIDATION
        train_start, train_end, val_start, val_end = train_start, v1_start, v1_start, v2_end
        train_index = train_df[train_df.day_dt.between(train_start, train_end)].index
        val_index = train_df[train_df.day_dt.between(val_start, val_end)].index

        # Define X and y
        X = train_df[X_features]
        X['year'] = X['year'].apply(lambda x: 2020 if x == 2021 else x) #* CHANGE 2021 TO 2020
        y = train_df[y_feature[0]]
        X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[val_index], y.loc[train_index], y.loc[val_index]

        log(f"Training model...")
        model = train_model(X_train, X_test, y_train, y_test)

        log(f"Evaluating model...")
        y_pred_val = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # GET EVALUATE RESULTS
        train_evaluate, val_evaluate = train_df.loc[train_index], train_df.loc[val_index]
        train_evaluate['y_pred'], val_evaluate['y_pred']  = y_pred_train, y_pred_val
        train_evaluate['residual'], val_evaluate['residual'] = train_evaluate.ttl_quantity - train_evaluate.y_pred, val_evaluate.ttl_quantity - val_evaluate.y_pred
        
        # VALIDATION RESULTS
        item_wh = agg_results(val_start=val_start, val_end=val_end, df=val_evaluate)
        val_passed = np.mean([1 if r>=0.65 and r<=1.2 else 0 for r in item_wh.ratio])
        log(f"ALL VALIDATION {val_start, val_end} RESULTS________________________________________________________________________")
        show_results(df=item_wh, log_func=log)
        log(item_wh)

        if val_passed >= 0.75: # IF 75% SELL THRU BETWEEN 65% AND 120%

            log(f"ITEM PASSED!!!")
            if test_df.duplicated().any() == True:
                log("DUPLICATED DATA IN TEST DATA...REMOVED")
            test_df = test_df.drop_duplicates()
            
            # ADD VALIDATION DATASET INTO TRAINING
            log(f"TRAINING NEW MODEL WITH ADDED VALIDATION DATA...")
            model = train_model_w_validation(X=X, y=y)

            # PREDICT ON TEST DATASET
            log(f"PREDICTING ON TEST DATA...")
            test_X = test_df[X_features]
            test_X['year'] = test_X.year.apply(lambda x: 2020 if x == 2021 else x) #* CHANGE 2021 TO 2020
            test_df['y_pred'] = [round(x,4) for x in model.predict(test_X)]

            log(f"PREDICTION RESULTS:")
            pred_results = test_df.groupby('day_dt').agg({'ttl_quantity':'sum', 'y_pred':'sum', 'unit_price':'mean'})
            pred_results['ratio'] = pred_results.ttl_quantity / pred_results.y_pred
            log(pred_results)

            log(f"EXPORTING RESULTS - RESULT DATA, META DATA, IMG, LOG...")
            export_results(item_code=item,
                        train=train_evaluate,
                        val1=val_evaluate[val_evaluate.day_dt.between(v1_start, v1_end)],
                        val2=val_evaluate[val_evaluate.day_dt.between(v2_start, v2_end)],
                        test=test_df,
                        img_path=img_path,
                        result_data_path=result_data_path,
                        meta_path=meta_path,
                        log_path=log_path
                        )
            log(f"EXPORTED RESULTS AT {now()}:")

        log(f"ITEM TRAINING FINISHED AT {now()}")    

    except Exception as error:
        log(f"ERROR: {error}")
        log(f"ITEM TRAINING FINISHED AT {now()}")

if __name__ == '__main__': #* RUN CODE AND OUTPUT TO meta.log - WHICH INCLUDE ONLY TWO PRINT STATEMENTS (MAIN LOGS EXPORTED TO LOG_PATH)

    item_list = [i[:-4] for i in os.listdir(f"/app2/kevin_workspace/data03180414/")]
    print(f"TRAIN MODEL FOR {len(item_list)} ITEMS AT {now()}, ITEM LIST: {item_list}")

    # Load needed data to add new features
    unit_price = pd.read_csv('../data/scai_unit_pricefromsales.csv', sep = '|', parse_dates=['day_dt'])
    lunar = pd.read_csv('lunar_days.csv', parse_dates=['day_dt'], usecols=['day_dt','dis_spring'])
    prom_schedule = pd.read_csv('schedule.csv',parse_dates=['day_dt'])
    prom_schedule.fillna(0, inplace=True)

    # Multi processing
    pool = multi_pool(16)
    pool.map(main, item_list)
    pool.close()
    pool.join()

print(f"ALL ITEMS TRAINING COMPLETED AT {now()}")