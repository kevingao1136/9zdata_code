import os
import pandas as pd
import time

feature_full_log_path = '/app2/feature_pipeline/logs/feature_full'
work_table_log_path = '/app2/feature_pipeline/logs/work_table'

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def right_now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

print(f'WAITING FOR 10AM...')
while True:

    now = pd.Timestamp.now().strftime('%H:%M')

    if now == '10:00':
        print(f'STARTING WORK TABLE FULL AT {right_now()}...')
        os.system(f'nohup python -u work_table_full.py > {work_table_log_path}/{today()}.log 2>&1&')
        time.sleep(61)

    if 'complete.csv'  in os.listdir('/app2/feature_pipeline'):
        print(f'STARTING FEATURE FULL AT {right_now()}...')
        os.system('rm /app2/feature_pipeline/complete.csv')
        os.system(f'nohup python -u feature_full.py > {feature_full_log_path}/{today()}.log 2>&1&')
        print(f'FEATURE PIPELINE FOR {today()} COMPLETE...')

    time.sleep(1)