import os
import time
import pandas as pd

download_path = '/app/auto_pipeline/download/processed'
upload_path = '/app/auto_pipeline/upload/processed'
output_path = '/app/auto_pipeline/output/processed'

def today():
    return pd.to_datetime(pd.Timestamp.now().strftime("%Y-%m-%d"))

while True:

    print('SCANNING FOR FILES 3 DAYS AGO AND DELETE...')
    print(f'TODAY IS {today()}')
    for path in [download_path, upload_path]:

        for file in os.listdir(path):

            file_date = pd.to_datetime(file[-12:file.index('.tar')])
            date_diff = (today() - file_date) > pd.Timedelta(days=3)

            if date_diff:
                os.system(f"rm {path}/{file}")
                print(f'DELETED {path}/{file} ON {today()}')

    # for file in os.listdir(output_path):

    #     if file.endswith('.csv'):
    #         os.system(f"rm {output_path}/{file}")
    #         print(f'DELETED {output_path}/{file} ON {today()}')
    
    time.sleep(86400)