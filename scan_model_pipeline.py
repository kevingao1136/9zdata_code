import os
import time
import pandas as pd

def now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#%%
#* UNTAR FILE FROM /APP/DOWNLOAD TO DATA, TRAIN MODEL AND OUTPUT RESULT AND META DATA TO /APP/UPLOAD
tar_path = '/app/download'
upload_path = '/app/upload'
result_data_path = '/app/auto_pipeline_main/output/result_data'
meta_data_path = '/app/auto_pipeline_main/output/meta_data'
work_space_path = '/app/auto_pipeline_main'
data_path_to_remove = '/app/auto_pipeline_main/app'
completed_path = '/app/auto_pipeline_main/output'

def unzip_tar():
    '''
    1. MOVE ALL THE DATA IN TAR FILE INTO DATA PATH
    2. MOVE TAR FILE TO PROCESSED
    '''

    print(f"SCANNING FOR NEW TAR FILES AT {now()} IN {tar_path}")
    tar_path_files = [f for f in os.listdir(tar_path) if not f.startswith('.')]
    if len(tar_path_files) == 2 and 'processed' in tar_path_files:

        print("UNTARRING TAR FILE...")
        tar_filename = [f for f in os.listdir(tar_path) if '.tar' in f][0]

        if tar_filename:
            os.system(f"tar -xvf {tar_path}/{tar_filename} -C {work_space_path}")
            os.system(f"mv {tar_path}/{tar_filename} {tar_path}/processed")

        else:
            print('TAR FILE NOT EXIST')

        print(f'DECOMPRESSED TAR FILE {tar_filename} TO {work_space_path}')

    else:
        raise "WRONG NUMBER OF ITEMS IN TAR PATH"

print('SCANNING FOR TAR FILE EVERY SECOND...')
while True:

    if [f for f in os.listdir(tar_path) if f.startswith('10') and f.endswith('.tar')]:

        print(f'TAR FILE FOUND, WAITING FOR IT TO LOAD AT {now()}')
        tar_filename = [f for f in os.listdir(tar_path) if f.startswith('10') and f.endswith('.tar')][0]

        # WAITING UNTIL LAST MODIFIED TIME IS 5 MINUTES AWAY FROM NOW
        while True:

            raw_time = time.localtime(os.path.getmtime(f'{tar_path}/{tar_filename}'))
            file_modified_time = time.strftime('%Y-%m-%d %H:%M:%S', raw_time)
            sec_from_modified = (pd.to_datetime(now()) - pd.to_datetime(file_modified_time)).seconds
            if sec_from_modified > 300:
                break

        print(f'MODEL PIPELINE STARTED AT {now()}')
        print('GENERATING TRAINING DATA FROM TAR...')
        unzip_tar()
        #--------------------------------------------------------------------------------------------------------------------------------
        print('START TRAINING MODEL...')
        os.system('nohup python -u /app/auto_pipeline_main/train_model.py > /app/auto_pipeline_main/output/log/meta_train.log 2>&1&')

        print('KEEP SCANNING FOR TAR FILE EVERY SECOND...')

    time.sleep(1)
