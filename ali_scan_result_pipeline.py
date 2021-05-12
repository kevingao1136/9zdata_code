import os
import tarfile
import socket
import pandas as pd
import pexpect
import time

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

result_data_path = '/app/auto_pipeline/output/result_data'
meta_data_path = '/app/auto_pipeline/output/meta_data'
corr_path = '/app/auto_pipeline/output/corr_matrix'
upload_path = '/app/auto_pipeline/upload'
completed_path = '/app/auto_pipeline/output'
server_name = socket.gethostbyname(socket.gethostname())

print('SCANNING FOR COMPLETED CSV EVERY SECOND...')
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

        print('SCP FILES TO MAIN SERVER...')
        child = pexpect.spawn(f"scp {result_data_tar_path} {meta_data_tar_path} {corr_tar_path} gaiprd@10.44.2.10:/app/upload")
        child.expect(f"gaiprd@10.44.2.10's password:")
        child.sendline("nL1C39aGSC")
        child.expect(pexpect.EOF, timeout=None)
        print(f'RESULT DATA AND META DATA SENT TO 10.44.2.10:/app/upload AT {now()}')

        print('REMOVING ALL TMP FILES...')
        os.system(f'rm -r /app/auto_pipeline/app')
        os.system('mv /app/auto_pipeline/output/log/* /app/auto_pipeline/output/processed')
        os.system('mv /app/auto_pipeline/output/meta_data/* /app/auto_pipeline/output/processed')
        os.system('mv /app/auto_pipeline/output/result_data/* /app/auto_pipeline/output/processed')
        os.system('mv /app/auto_pipeline/output/model_img/* /app/auto_pipeline/output/processed')
        os.system('mv /app/auto_pipeline/output/corr_matrix/* /app/auto_pipeline/output/processed')
        os.system(f'mv {upload_path}/*.tar {upload_path}/processed')
        print(f'MODEL PIPELINE ENDED AT {now()}')
        print('KEEP SCANNING EVERY SECOND...')
    
    time.sleep(1)