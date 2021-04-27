#%%
import tarfile
import pandas as pd
import socket
import os
import time

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

upload_path = '/app/upload'
result_data_path = '/app/auto_pipeline_main/output/result_data'
meta_data_path = '/app/auto_pipeline_main/output/meta_data'
data_path_to_remove = '/app/auto_pipeline_main/app'
completed_path = '/app/auto_pipeline_main/output/completed.csv'
server_name = socket.gethostbyname(socket.gethostname())

#%%
print('START SCANNING FOR COMPLETED CSV EVERY SECOND...')
while True:

    if os.path.exists(completed_path):

        result_data_tar_path = f"{upload_path}/{server_name}_result_data_{today()}.tar"
        meta_data_tar_path = f"{upload_path}/{server_name}_meta_data_{today()}.tar"
        result_tarfile = tarfile.open(result_data_tar_path, mode="w")
        meta_tarfile = tarfile.open(meta_data_tar_path, mode="w")

        print('TRAINING COMPLETED, TARRING RESULT AND META DATA...')
        os.system(f'rm {completed_path}')

        for result_data, meta_data in zip(os.listdir(result_data_path), os.listdir(meta_data_path)):
            result_tarfile.add(f"{result_data_path}/{result_data}")
            meta_tarfile.add(f"{meta_data_path}/{meta_data}")
        print(f'RESULT AND META DATA EXPORTED TO {upload_path}')

        print('REMOVING ALL TMP FILES...')
        os.system(f'rm -r {data_path_to_remove}')
        os.system('mv /app/auto_pipeline_main/output/result_data/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/meta_data/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/model_img/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/log/* /app/auto_pipeline_main/output/processed')

        print(f'RESULT META SCAN ENDED AT {now()}')
        print('STILL SCANNING EVERY MINUTE...')

    time.sleep(1)