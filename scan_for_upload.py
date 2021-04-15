#%%
import shutil
import time
import os
# import shutil
from datetime import datetime

src_path = '/app/python-scripts/kevin_workspace/ali_upload/tar_upload/'
des_path = '/app/python-scripts/kevin_workspace/tar_upload/'

find_whole_hour = True
keep_scanning = True

def move_files():
    if os.listdir(src_path):
        print(f'MOVING {os.listdir(src_path)} TO NEW PATH...')
        for f in os.listdir(src_path):
            shutil.move(src_path + f, des_path)
        
    else:
        print(f'NO NEW DATA IN {src_path}.')

print('WAITING FOR THE NEXT WHOLE HOUR...')
while find_whole_hour:

    whole_hour = datetime.now().timestamp() % 3600
    whole_hour = 0
    if whole_hour == 0:

        find_whole_hour = False

        while keep_scanning:
            print('STARTING SCAN AT', datetime.now())
            move_files()
            time.sleep(3)

# %%
