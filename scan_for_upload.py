#%%
import pandas as pd
import time
import os
import tarfile
import pexpect

today = pd.Timestamp.now().strftime('%Y%m%d')
feature_path = '/app2/feature_pipeline/feature_classified'
# new_itemlist_path = f'/app2/feature_pipeline/data/item/items_classified{today}.pk'
new_itemlist_path = '/app2/feature_pipeline/data/item/items_classified20210421.pk'
tar_file_path = f'/app2/upload/A5_data_{today}.tar'

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

while True:

    print(f"CHECKING CURRENT TIME AT {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    now = pd.Timestamp.now().strftime('%H:%M:%S')
    time.sleep(10)

    if now[:5] == '23:30':
        print(f"ZIPPING FEATURES AT {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        get_tar()
        scp_to38()
        print('SCP COMPLETE, SCAN AGAIN IN 23 HOURS...')
        time.sleep(82800)

# %%
