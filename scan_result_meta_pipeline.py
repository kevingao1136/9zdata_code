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
corr_path = '/app/auto_pipeline_main/output/corr_matrix'
data_path_to_remove = '/app/auto_pipeline_main/app'
completed_path = '/app/auto_pipeline_main/output'
server_name = socket.gethostbyname(socket.gethostname())

#%%
print('START SCANNING FOR COMPLETED CSV EVERY SECOND...')
while True:

    if 'A5_completed.csv' in os.listdir(completed_path) and 'A6_completed.csv' in os.listdir(completed_path):

        print('TRAINING COMPLETED, TARRING RESULT AND META DATA...')
        os.system(f'rm {completed_path}/A5_completed.csv {completed_path}/A6_completed.csv')

        result_data_tar_path = f"{upload_path}/{server_name}_result_data_{today()}.tar"
        meta_data_tar_path = f"{upload_path}/{server_name}_meta_data_{today()}.tar"
        corr_tar_path = f"{upload_path}/{server_name}_corr_matrix_{today()}.tar"

        os.system(f"tar -cvf {result_data_tar_path} {result_data_path}/*")
        os.system(f"tar -cvf {meta_data_tar_path} {meta_data_path}/*")
        os.system(f"tar -cvf {corr_tar_path} {corr_path}/*")
        
        print(f'RESULT AND META DATA AND CORR MATRIX EXPORTED TO {upload_path}')

        print('REMOVING ALL TMP FILES...')
        os.system(f'rm -r {data_path_to_remove}')
        os.system('mv /app/auto_pipeline_main/output/result_data/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/meta_data/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/model_img/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/log/* /app/auto_pipeline_main/output/processed')
        os.system('mv /app/auto_pipeline_main/output/corr_matrix/* /app/auto_pipeline_main/output/processed')

        print(f'RESULT META SCAN ENDED AT {now()}')
        print('STILL SCANNING EVERY MINUTE...')

    time.sleep(1)