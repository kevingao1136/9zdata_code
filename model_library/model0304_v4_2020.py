import os
import functools
import contextlib
from multiprocessing import Pool
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool as catpool
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import traceback
import warnings
warnings.simplefilter("ignore")

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
    df['loc_selling_area'] = df.groupby(['store_type'])['loc_selling_area'].transform(lambda x: x.fillna(x.median()))
    df['p_rate'] = df.unit_price / df.retail_price #* CORRECT P_RATE AFTER UNIT PRICE UPDATE
    df['isweekend'] = df.weekday.apply(lambda x: 1 if x in (5,6) else 0)

    return df

def select_features(features, df):
    """
    features: list of features
    """
    return df[features]

def label_encoder(df, X_features):

    le = LabelEncoder()
    cat_cols = list(df[X_features].dtypes[df[X_features].dtypes == 'object'].index)
    try:
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
        # print(f"LABEL ENCODED COLS: {cat_cols}")

    except Exception as error:
        print(f"LABEL ENCODING ERROR: {error}")

    return df

def get_prom_schedule(item, prom_info):

    offer_schedule = pd.DataFrame()

    for start_date, end_date in zip(prom_info[prom_info.item == int(item)].offer_start_date, prom_info[prom_info.item == int(item)].offer_end_date):
        date_range = pd.date_range(start_date, end_date)
        offer_schedule = pd.concat([offer_schedule, pd.DataFrame({'day_dt':date_range, 'day_of_prom':range(len(date_range))})])

    return offer_schedule

def agg_results(df, v1_start, v1_end, v2_start, v2_end):
    
    # agg by item, wh for v1
    df_v1 = df[df.day_dt.between(v1_start, v1_end)]
    item_wh_v1 = df_v1.groupby(['item_code','loc_wh'],as_index=False)['ttl_quantity','y_pred'].sum()
    item_wh_v1['ratio'] = item_wh_v1['ttl_quantity'] / item_wh_v1['y_pred']
    item_wh_v1['y_pred'] = item_wh_v1['y_pred'].apply(lambda x: round(x,4))
    item_wh_v1['ratio'] = item_wh_v1['ratio'].apply(lambda x: round(x,4))
    item_wh_v1['unit_price'] = df_v1['unit_price'].mean()
    
    # agg by item, wh for v2
    df_v2 = df[df.day_dt.between(v2_start, v2_end)]
    item_wh_v2 = df_v2.groupby(['item_code','loc_wh'],as_index=False)['ttl_quantity','y_pred'].sum()
    item_wh_v2['ratio'] = item_wh_v2['ttl_quantity'] / item_wh_v2['y_pred']
    item_wh_v2['y_pred'] = item_wh_v2['y_pred'].apply(lambda x: round(x,4))
    item_wh_v2['ratio'] = item_wh_v2['ratio'].apply(lambda x: round(x,4))
    item_wh_v2['unit_price'] = df_v2['unit_price'].mean()
    
    item_wh = pd.concat([item_wh_v1,item_wh_v2])
    
    return item_wh

def load_data(item_path, train_start, val_end, test_start, test_end):
    """
    Parameters:
        item: item code
        train_start, train_end, test_start, test_end
    Returns:
        train_val_df: training data including validation data
        test_df: test data
    """
    df = pd.read_pickle(item_path)
    

    # GET TRAIN DATA AND IMPUTE UNIT PRICE
    train_val_df = df[df.day_dt.between(train_start, val_end)]
    train_val_df.area_idnt = train_val_df.area_idnt.astype(int)
    train_val_df = train_val_df.merge(unit_price, on=['item_code','area_idnt','day_dt'], how='left')
    train_val_df['unit_price2'].fillna(train_val_df['unit_price'],inplace=True)
    train_val_df.drop('unit_price',axis=1,inplace=True)
    train_val_df.rename({'unit_price2':'unit_price'},axis=1,inplace=True)

    # GET TEST DATA
    test_df = df[df.day_dt.between(test_start, test_end)]

    return train_val_df, test_df

def train_model(X_train, X_test, y_train, y_test, sample_weight):

    cat_features = ['year','month','store_type','activity_desc']

    train_pool = catpool(data=X_train, label=y_train, cat_features=cat_features, weight=sample_weight)
    val_pool = catpool(data=X_test, label=y_test, cat_features=cat_features)

    # CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        loss_function='RMSE',
        one_hot_max_size=50,
        # loss_function='Tweedie:variance_power=1.9',
        eval_metric='RMSE',
        verbose=200
        )

    model.fit(
        X=train_pool,
        eval_set=val_pool,
        use_best_model=False,
        early_stopping_rounds=50
    )

    return model

def retrain_model(X, y, trees, learning_rate, sample_weight):

    cat_features = ['year','month','store_type','activity_desc']

    train_pool = catpool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

    # CatBoost model
    #* PARAMETERS DEPEND ON PRIOR MODEL
    model = CatBoostRegressor(
        iterations=trees,
        learning_rate=learning_rate,
        loss_function='RMSE',
        one_hot_max_size=50,
        # loss_function='Tweedie:variance_power=1.5',
        eval_metric='RMSE',
        verbose=200
        )

    model.fit(X=train_pool)

    return model

def show_results(df):
    print(f"0-50%: {np.mean([1 if r<=0.5 else 0 for r in df.ratio])}")
    print(f"50-80%: {np.mean([1 if r>=0.5 and r<=0.8 else 0 for r in df.ratio])}")
    print(f"80-100%: {np.mean([1 if r>=0.8 and r<=1 else 0 for r in df.ratio])}")
    print(f"100-120%: {np.mean([1 if r>=1 and r<=1.2 else 0 for r in df.ratio])}")
    print(f"120-inf%: {np.mean([1 if r>=1.2 else 0 for r in df.ratio])}")
    print(f"65-120%: {np.mean([1 if r>=0.65 and r<=1.2 else 0 for r in df.ratio])}")

def plot_pred_results(item_code, train, val, test, img_path, export_img):
    '''
    Parameters:
        item_code - item code
        train - train dataset, required features: day_dt, ttl_quantity, y_pred, unit_price
        val - validation dataset (v1 + v2), required features: day_dt, ttl_quantity, y_pred, unit_price
        test - test dataset, required features: day_dt, y_pred, ttl_quantity, unit_price
        img_path - image export path
    '''
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(30,12))
    fig.subplots_adjust(right=0.8)
    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.1))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    # PLOT TRAIN, VAL, TEST
    y_true_train, = host.plot(train.groupby(['day_dt']).ttl_quantity.sum(),label='y_true_train',color='tab:blue')
    y_pred_train, = host.plot(train.groupby(['day_dt']).y_pred.sum(),label='y_pred_train',color='tab:orange')
    y_true_val, = host.plot(val.groupby(['day_dt']).ttl_quantity.sum(),label='y_true_val',color='tab:olive')
    y_pred_val, = host.plot(val.groupby(['day_dt']).y_pred.sum(),label='y_pred_val',color='tab:red')
    y_true_test, = host.plot(test.groupby(['day_dt']).ttl_quantity.sum(),label='y_true_test',color='k')
    y_pred_test, = host.plot(test.groupby(['day_dt']).y_pred.sum(),label='y_pred_test',color='tab:green')
    train_unit_price, = par1.plot(train.groupby(['day_dt']).unit_price.mean(),'--',label='unit_price',color='k')
    val_unit_price, = par1.plot(val.groupby(['day_dt']).unit_price.mean(),'--', color='k')
    test_unit_price, = par1.plot(test.groupby(['day_dt']).unit_price.mean(),'--',color='k')
    train_store_cnt, = par2.plot(train.groupby(['day_dt']).store_id.nunique(),'--',label='store_cnt',color='m')
    val_store_cnt, = par2.plot(val.groupby(['day_dt']).store_id.nunique(),'--',color='m')
    test_store_cnt, = par2.plot(test.groupby(['day_dt']).store_id.nunique(),'--',color='m')
    host.axvline(x=val.day_dt.min(),color='k',ls='--',linewidth=3)
    x_ticks = pd.to_datetime(pd.date_range(train.day_dt.min(), test.day_dt.max()).map(lambda x: x.strftime('%Y-%m')).unique())
    host.tick_params(axis='x', rotation=45)
    host.set_xticks(x_ticks)
    host.grid()

    # SET Y LIMITS
    par1.set_ylim([0,max(train.unit_price.max(),val.unit_price.max(),test.unit_price.max())+5])
    par2.set_ylim(0, 4500)

    # SET LABELS
    host.set_title(f"PREDICTION RESULTS FOR ITEM {item_code}", fontsize=20)
    host.set_xlabel("day_dt",fontsize=20)
    host.set_ylabel("ttl_quantity",fontsize=20)
    par1.set_ylabel("unit_price",fontsize=20)
    par2.set_ylabel("store_cnt",fontsize=20)

    # SET LEGEND NAMES
    lines = [y_true_train, y_pred_train, y_true_val, y_pred_val, y_true_test, y_pred_test, train_unit_price, val_unit_price, test_unit_price, train_store_cnt, val_store_cnt, test_store_cnt]
    host.legend(lines, [l.get_label() for l in lines])

    # EXPORT IMAGE
    if export_img:
        plt.savefig(f"{img_path}{item_code}.png")

def export_results(item_code, train, val1, val2, test, export_result_data, export_meta_data, export_img):
    '''
    Parameters:
        item_code - item code
        train - train dataset
        val1 - v1 dataset
        val2 - v2 dataset
        test - test dataset
    
    Export to log_path, result_data_path, meta_path, img_path
    '''
    if export_result_data:
        # CREATE RESULT DATA -----------------------------------------------------------------------------------------------------------------
        # 0 IF TEST PREDICTION < 0
        test['y_pred'] = test['y_pred'].apply(lambda x: 0 if x < 0 else x)

        # GET STORE COUNT
        test_store_cnt = test.store_id.nunique()

        # IF DUPLICATE STORES IN TEST DATA, SKIP ITEM
        if test_store_cnt == test[['day_dt','item_code','store_id']].drop_duplicates().store_id.nunique(): # IF NO DUPLICATES
            print(f"THERE ARE {test_store_cnt} STORES IN TEST DATA") # print STORE COUNT
            if np.sum(test.groupby(['loc_wh']).y_pred.sum() < 100) > 0: # IF ANY DC IS LESS THAN 100, SKIP ITEM
                print(f"ALERT>>>>>>>>>>>>>>>>>> {np.sum(test.groupby(['loc_wh']).y_pred.sum() < 100)} DC HAS TTL_QUANTITY < 100!!! SKIP ITEM...")
            # else:
            #     print(test.groupby(['loc_wh']).y_pred.sum())
        else: # IF DUPLICATE
            print(f"ALERT>>>>>>>>>>>>>>>>>> DUPLICATED STORES IN TEST DATA!!! SKIP ITEM...")

        # CREATE RESULT DATA
        train_val = pd.concat([train, val1, val2])
        train_val['istest'] = 0
        test['istest'] = 1
        result_data = pd.concat([train_val, test])
        result_data['model_id'] = "SC_A5_LGBM_V1"
        result_data = result_data[['model_id','item_code','store_id','day_dt','ttl_quantity','y_pred','istest']]
        result_data = result_data[result_data.istest==1]
        assert result_data.istest.sum() > 0, "NO TEST RESULT DATA!"
        result_data.to_csv(f"{result_data_path}{item_code}_A5_result_data.csv",index=False)

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
                'test_store_cnt':test_store_cnt,
                'item_code':item_code}


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
    plot_pred_results(item_code=item_code, train=train, val=all_val, test=test, img_path=img_path, export_img=export_img)

    # CONVERT DICT TO DATAFRAME AND EXPORT METADATA
    for k, v in meta_dict.items():
        meta_dict[k] = [v]

    meta_data = pd.DataFrame(meta_dict)
    meta_cols = ['model_id','scenario','aid','model_time','log_path','result_path',
                'img_path','v1_start_time','v1_end_time','v1_accuracy','v1_sell_thru',
                'v2_start_time','v2_end_time','v2_accuracy','v2_sell_thru','package','test_store_cnt']
    meta_data = meta_data[meta_cols]
    meta_data = pd.concat([meta_data]*4, ignore_index=True)
    meta_data['test_start_time'] = ['2021-03-18','2021-04-15','2021-05-13','2021-06-03']
    meta_data['test_end_time'] = ['2021-04-14','2021-05-12','2021-06-02','2021-06-30']
    if export_meta_data:
        meta_data.to_csv(f"{meta_path}{item_code}_A5_meta_data.csv",index=False)

def main(item, threshold):

    # REDIRECT STDOUT AND STDERR TO LOG FILE
    with open(f"{log_path}{item}.log", "w") as train_log, contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(train_log):
        try:
            print(f"ITEM {item} STARTING AT {now()}, PROCESS: {item_list.index(item)+1} OUT OF {len(item_list)}")
            print(f"Loading data from {data_path}")
            raw_train_val_df, raw_test_df = load_data(f"{data_path}{item}.pk.gz", 
                train_start=train_val_start, val_end=train_val_end, test_start=test_start, test_end=test_end)

            if raw_train_val_df.day_dt.min() >= pd.to_datetime('2020-01-01'):
                print(f"TRAINING DATA TOO SMALL, SKIP ITEM")
                return None

            # print TIME RANGE
            print(f"train_val_df: {raw_train_val_df.day_dt.min()} to {raw_train_val_df.day_dt.max()}")
            print(f"test_df: {raw_test_df.day_dt.min()} to {raw_test_df.day_dt.max()}")

            # IF EMPTY, CONTINUE WITH NEXT ITEM
            if len(raw_train_val_df[raw_train_val_df.day_dt < v1_start]) == 0:
                print(f"EMTPY TRAIN AND VAL DATA >>>")
                return None
            if len(raw_test_df) == 0:
                print(f"EMPTY TEST DATA >>>")
                return None
            
            # PRINT OFFER DESC FROM TRAIN, VAL AND TEST
            try:                
                print("\nOFFERS IN TRAIN AND VAL DATA:")
                for offer in raw_train_val_df.offer_code.dropna().unique():
                    tmp = prom_info[(prom_info.offer_code == int(offer)) & (prom_info.item == int(item))][['item_desc','offer_start_date','offer_end_date','promotion_merch','buyer_note','retail_price', 'prom_price','unit_price']]
                    print(tmp.to_markdown(showindex=False))
                    print(prom_info[prom_info.offer_code == int(offer)].item_desc)

                print("\nOFFERS IN TEST DATA:")
                for offer in raw_test_df.offer_code.dropna().unique():
                    tmp = prom_info[(prom_info.offer_code == int(offer)) & (prom_info.item == int(item))][['item_desc','offer_start_date','offer_end_date','promotion_merch','buyer_note','retail_price', 'prom_price','unit_price']]
                    print(tmp.to_markdown(showindex=False))
                    print(prom_info[prom_info.offer_code == int(offer)].item_desc)

            except Exception as e:
                print(f"Error when printing offer desc: {e}")

            # MAKE NEW COPIES OF DATAFRAMES
            train_val_df, test_df = raw_train_val_df.copy(), raw_test_df.copy()
            del raw_train_val_df, raw_test_df

            # ADD NUM ITEMS TO TRAIN AND TEST
            num_items_dict = prom_info.groupby(['offer_code']).item.count().to_frame().reset_index().rename({'item':'num_items_in_offer'},axis=1)
            train_val_df.offer_code.fillna('0',inplace=True)
            train_val_df.offer_code = train_val_df.offer_code.astype(int)
            train_val_df = train_val_df.merge(num_items_dict, how='left', on='offer_code')
            train_val_df.num_items_in_offer.fillna(100,inplace=True)

            test_df.offer_code.fillna('0',inplace=True)
            test_df.offer_code = test_df.offer_code.astype(int)
            test_df = test_df.merge(num_items_dict, how='left', on='offer_code')
            test_df.num_items_in_offer.fillna(100,inplace=True)

            # ADD LUNAR NEW YEAR FEATURE
            train_val_df = train_val_df.merge(lunar, how='left', on='day_dt')
            test_df = test_df.merge(lunar, how='left', on='day_dt')

            # ADD PROMOTION SCHEDULE FEATURE
            prom_schedule = get_prom_schedule(item=item, prom_info=prom_info) #* GET PROMOTION SCHEDULE FOR CURRENT ITEM
            train_val_df = train_val_df.merge(prom_schedule, how='left', on='day_dt')
            test_df = test_df.merge(prom_schedule, how='left', on='day_dt')
            train_val_df.day_of_prom.fillna(-1,inplace=True)
            test_df.day_of_prom.fillna(-1,inplace=True)

            # DEFINE FEATURES TO SELECT
            index_features = ['day_dt','store_id','item_code','loc_wh']

            X_features = ['year','month','day_of_prom','weekday','day_type','last_day_type','is_5th','store_type','activity_desc',
                        'prom0','prom1','prom2','prom3','prom4','prom5','prom6','prom7', 'is_vip','free_gift',
                        'required_num','amount','prom_price','unit_price','p_rate','dis_spring','num_items_in_offer']

            y_feature = ['ttl_quantity']

            print(f"\nSELECTED FEATURES: {X_features}")

            print(f"Cleaning train data...")
            train_val_df = clean_data(train_val_df)
            train_val_df = impute_data(train_val_df)
            train_val_df = select_features(index_features+X_features+y_feature, train_val_df)
            train_val_df = label_encoder(train_val_df,X_features)

            print(f"Cleaning test data...")
            test_df = clean_data(test_df)
            test_df = impute_data(test_df)
            test_df = select_features(index_features+X_features+y_feature, test_df)
            test_df = label_encoder(test_df,X_features)

            #* SORT train_val_df FOR WEIGHT ASSIGNMENT
            train_val_df = train_val_df.sort_values('day_dt')

            # SPLIT TRAIN DATA INTO TRAIN AND VALIDATION
            train_start, train_end, val_start, val_end = train_val_start, v1_start, v1_start, v2_end
            train_index = train_val_df[train_val_df.day_dt.between(train_start, train_end)].index
            val_index = train_val_df[train_val_df.day_dt.between(val_start, val_end)].index

            #* ASSIGN SAMPLE WEIGHT BY DAY_DT BEFORE VALIDATION SET
            print(f"ASSIGNING SAMPLE WEIGHT FOR TRAIN DATA FROM {train_start} TO {train_end}")
            date_range = pd.date_range(train_start, train_end)
            weight_dict = dict(zip(date_range, np.linspace(0,1,len(date_range)) ** 4))
            sample_weight_train = [weight_dict[day_dt] for day_dt in train_val_df.loc[train_index].day_dt]

            # plt.figure(figsize=(12,8))
            # plt.plot(train_val_df.loc[train_index].day_dt, sample_weight_train)
            # plt.savefig("./test/sample_weight_train.png")

            #* ASSIGN SAMPLE WEIGHT BY DAY_DT BEFORE TEST SET
            print(f"ASSIGNING SAMPLE WEIGHT FOR TRAIN AND VAL FROM {train_start} TO {val_end}")
            date_range = pd.date_range(train_start, val_end)
            weight_dict = dict(zip(date_range, np.linspace(0,1,len(date_range)) ** 4))
            sample_weight_train_val = [weight_dict[day_dt] for day_dt in train_val_df.day_dt]

            # plt.figure(figsize=(12,8))
            # plt.plot(train_val_df.day_dt, sample_weight_train_val)
            # plt.savefig("./test/sample_weight_train_val.png")

            # Define X and y
            X = train_val_df[X_features]
            X['year'] = X['year'].apply(lambda x: 2020 if int(x) == 2021 else x)
            y = train_val_df[y_feature[0]]
            X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[val_index], y.loc[train_index], y.loc[val_index]
            print(f"Training model...")
            model = train_model(X_train, X_test, y_train, y_test, sample_weight=sample_weight_train)
            # print(model.get_feature_importance(type='ShapValues', data=catpool(X_train, y_train, cat_features=['year','month','store_type','activity_desc'])))
            print("FEATURE IMPORTANCE - PREDICTION VALUES CHANGE:")
            print(pd.Series(index=X_train.columns, 
                        data=model.get_feature_importance(type='PredictionValuesChange')).sort_values(ascending=False))
            print("FEATURE IMPORTANCE - LOSS FUNCTION CHANGE:")
            print(pd.Series(index=X_train.columns, 
                        data=model.get_feature_importance(type='LossFunctionChange', 
                        data=catpool(X_train, y_train, cat_features=['year','month','store_type','activity_desc']))).sort_values(ascending=False))

            print(f"Evaluating model with {model.best_iteration_+50} trees and {round(model.learning_rate_, 4)} learning rate...")
            y_pred_val = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            # GET EVALUATE RESULTS
            train_evaluate, val_evaluate = train_val_df.loc[train_index], train_val_df.loc[val_index]
            train_evaluate['y_pred'], val_evaluate['y_pred']  = y_pred_train, y_pred_val

            # VALIDATION RESULTS
            item_wh = agg_results(df=val_evaluate, v1_start=v1_start, v1_end=v1_end, v2_start=v2_start, v2_end=v2_end)
            val_passed = np.mean([1 if r>=0.65 and r<=1.2 else 0 for r in item_wh.ratio])
            print(f"ALL VALIDATION {val_start, val_end} RESULTS________________________________________________________________________")
            show_results(item_wh)
            print(item_wh)

            if val_passed >= threshold: #* IF 75% SELL THRU BETWEEN 65% AND 120%

                print(f"ITEM PASSED!!!")
                if test_df.duplicated().any(): # CHECK DUPLICATES IN TEST DATA
                    print("DUPLICATED DATA IN TEST DATA...REMOVED")
                test_df = test_df.drop_duplicates()

                # ADD VALIDATION DATASET INTO TRAINING
                print(f"TRAINING NEW MODEL WITH ADDED VALIDATION DATA USING {model.best_iteration_+50} TREES AND {round(model.learning_rate_,4)} LEARNING RATE...")
                retrained_model = retrain_model(X=X, y=y, trees=model.best_iteration_+50, learning_rate=model.learning_rate_, sample_weight=sample_weight_train_val)

                # PREDICT ON TEST DATASET
                print(f"PREDICTING ON TEST DATA...")
                test_X = test_df[X_features]
                test_X['year'] = test_X.year.apply(lambda x: 2020 if int(x) == 2021 else x)
                test_df['y_pred'] = [round(x,4) for x in retrained_model.predict(test_X)]

                print(f"PREDICTION RESULTS:")
                pred_results = test_df.groupby('day_dt').agg({'ttl_quantity':'sum', 'y_pred':'sum', 'unit_price':'mean'})
                pred_results['ratio'] = pred_results.ttl_quantity / pred_results.y_pred
                print(pred_results)

                print(f"EXPORTING RESULT DATA, META DATA, IMG, LOG...")
                export_results(item_code=item,
                            train=train_evaluate,
                            val1=val_evaluate[val_evaluate.day_dt.between(v1_start, v1_end)],
                            val2=val_evaluate[val_evaluate.day_dt.between(v2_start, v2_end)],
                            test=test_df,
                            export_result_data=export_result_data,
                            export_meta_data=export_meta_data,
                            export_img=export_img)
                print(f"EXPORTED RESULTS TO {img_path} AT {now()}")
                
            print(f"ITEM TRAINING FINISHED AT {now()}")
            
        except:
            traceback.print_exc()      

if __name__ == '__main__':
    
    #* DEFINE PATHS
    data_path = f"/app2/kevin_workspace/data0304/"
    output_path = f"/app/python-scripts/kevin_workspace/outputs/output0304_v5_2020/"
    img_path, log_path, result_data_path, meta_path = output_path+'train_img/', output_path+'log/', output_path+'result_data/', output_path+'meta_data/'
    export_result_data = False
    export_meta_data = False
    export_img = True
    threshold = 0
    num_cores = 1
    # img_path = log_path = result_data_path = meta_path = './test/'

    #* SET DATES
    train_val_start, train_val_end ='2020-01-01', '2021-01-27'
    test_start, test_end = '2021-03-18', '2021-06-30'
    v1_start, v1_end = '2020-12-16', '2021-01-07'
    v2_start, v2_end = '2021-01-08', '2021-01-27'
    assert train_val_end == v2_end, "train_val_end should = v2_end"

    #* DEFINE ITEM LIST
    # item_list = [i[:i.index(".pk")] for i in os.listdir(data_path)]
    item_list = [i[:-4] for i in os.listdir('/app/python-scripts/kevin_workspace/outputs/output0304_v1_submit/train_img/') if i.endswith('.png')]
    # exported = os.listdir()
    # item_list = item_list[:2]
    # item_list = ['100014272']
    
    print(f"TRAINING MODEL FOR {len(item_list)} ITEMS AT {now()}, ITEM LIST: {item_list}")

    # Load needed data to add new features
    unit_price = pd.read_csv('/app/python-scripts/kevin_workspace/assist_data/unit_price0304.csv.gz',parse_dates=['day_dt'],compression='gzip')
    lunar = pd.read_csv('/app/python-scripts/kevin_workspace/assist_data/lunar_days.csv.gz', parse_dates=['day_dt'], usecols=['day_dt','dis_spring'],compression='gzip')
    prom_info = pd.read_csv('/app/python-scripts/kevin_workspace/assist_data/prom_info.csv.gz',compression='gzip',parse_dates=['offer_start_date','offer_end_date'])

    # RUN MULTIPROCESSING
    pool = Pool(num_cores)
    pool.map(functools.partial(main, threshold=threshold), item_list)
    pool.close()
    pool.join()

    print(f"TRAINING COMPLETED AT {now()}")