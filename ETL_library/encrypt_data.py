import pandas as pd
import numpy as np
from multiprocessing import Pool
import os

data_path = '/app2/kevin_workspace/data0304/'
export_path = '/app2/kevin_workspace/data0304_encrypt/'
item_list = [i[:i.index('.pk.gz')] for i in os.listdir(data_path)]

mapper_df = pd.read_pickle('/app/python-scripts/kevin_workspace/assist_data/fea_mapper.pk.gz',compression='gzip')
mapper_df.code_type.replace({'loc_idnt':'store_id','item_idnt':'item_code'},inplace=True)
features = list(mapper_df.code_type.unique())

def get_mapper(feat):

    mapper = mapper_df[mapper_df.code_type == feat][['key_id','value_id']]
    mapper.rename({'key_id':feat+'_new','value_id':feat},axis=1,inplace=True)

    return mapper

def main(item):

    print("ENCRYPTING ITEM: " + item)
    df = pd.read_pickle(data_path + item + '.pk.gz',compression='gzip')
    df['store_id'], df['item_code'] = df['store_id'].astype(str), df['item_code'].astype(str)
    common_feat = list(set(df.columns) & set(features))
    print("ENCRYPT FEATURES: " + str(common_feat))

    for feat in common_feat:
        print('ENCRYPTING ' + feat)
        feat_mapper = get_mapper(feat)
        df = df.merge(feat_mapper, how='left', on=feat, indicator=True)
        if 'both' not in df._merge.astype(str).unique(): print("NO DATA WAS ENCRYPTED >>>>>>>>>>>>>>>>>")
        df.drop(['_merge'],axis=1,inplace=True)

    # DROP COLUMNS
    df.drop(common_feat, axis=1, inplace=True)
    print("FEATURES: " + str(list(df.columns)))
    df.to_csv(export_path + item + '.csv.gz', header=False, index=False, compression='gzip')
    print(f"ITEM {item} EXPORTED TO {export_path}")

# RUN MULTIPROCESSING
pool = Pool(8)
pool.map(main, item_list)
pool.close()
pool.join()

print(f"TRAINING COMPLETED AT {pd.Timestamp().now()}")