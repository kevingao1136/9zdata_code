#%%
import pandas as pd
import time
import os
import tarfile
import pexpect

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def get_tar():

    if os.path.exists(new_itemlist_path):

        new_itemlist = list(pd.read_pickle(new_itemlist_path).item_code)
        tar_file = tarfile.open(tar_file_path, mode='w')
        cnt = 0

        tar_file.add(new_itemlist_path)
        print(f'ADDED {new_itemlist_path}')

        for item in new_itemlist:

            item_path = f'{feature_path}/feature_{item}.csv.gz'
            if os.path.exists(item_path):
                tar_file.add(item_path)
                cnt += 1
                print(f'ADDED {item}')
            else:
                print(f'{item} DOES NOT EXIST IN {feature_path}')
        tar_file.close()
        print(f'TAR FILE CREATED AT {tar_file_path}')
        print(f'{cnt} ITEMS ADDED TO TAR FILE.')

def scp_to38():

    child = pexpect.spawn(f"scp {tar_file_path} bbmprd@10.82.20.38:/app2/upload/")
    child.expect(f"bbmprd@10.82.20.38's password:")
    child.sendline("CYB4QBOgX5")
    child.expect(pexpect.EOF, timeout=None)
    os.system(f"mv {tar_file_path} /app2/upload/processed")

print('WAITING FOR 23:00...')
while True:

    feature_path = '/app2/feature_pipeline/feature_classified'
    new_itemlist_path = f'/app2/feature_pipeline/data/item/items_classified{today()}.pk'
    tar_file_path = f'/app2/upload/A5_data_{today()}.tar'

    now = pd.Timestamp.now().strftime('%H:%M')

    if now == '23:30':
        
        print(f"ZIPPING FEATURES AT {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        get_tar()

        print('START TO SCP TO 38...')
        scp_to38()

        print('START SCANNING AGAIN IN 23 HOURS...')
        time.sleep(82800) #SLEEP 23 HOURS

    time.sleep(1)

# %%
