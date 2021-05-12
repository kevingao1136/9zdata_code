import os
import pandas as pd
import time

feature_full_log_path = '/app2/feature_pipeline/logs/feature_full'
work_table_log_path = '/app2/feature_pipeline/logs/work_table'
feature_delta1_log_path = '/app2/feature_pipeline/logs/feature_delta1'
feature_delta2_log_path = '/app2/feature_pipeline/logs/feature_delta2'

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def right_now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

print(f'WAITING FOR 10AM...')
while True:

    now = pd.Timestamp.now().strftime('%H:%M')

    if now == '10:35':
        print(f'STARTING WORK TABLE FULL AT {right_now()}...')
        os.system(f'nohup python -u work_table_full.py > {work_table_log_path}/{today()}.log 2>&1&')
        time.sleep(61)

    if 'work_table_complete.csv'  in os.listdir('/app2/feature_pipeline'):
        print(f'STARTING FEATURE FULL AT {right_now()}...')
        os.system('rm /app2/feature_pipeline/work_table_complete.csv')
        os.system(f'nohup python -u feature_full.py > {feature_full_log_path}/{today()}.log 2>&1&')
        print(f'FEATURE PIPELINE FOR {today()} COMPLETE...')

    if now == '13:00':
        print(f'STARTING FEATURE DELTA1 AT {right_now()}...')
        os.system(f'nohup python -u feature_delta1.py > {feature_delta1_log_path}/{today()}.log 2>&1&')
        time.sleep(61)
    
    if now == '18:00':
        print(f'STARTING FEATURE DELTA1 AT {right_now()}...')
        os.system(f'nohup python -u feature_delta2.py > {feature_delta2_log_path}/{today()}.log 2>&1&')
        time.sleep(61)

    time.sleep(1)