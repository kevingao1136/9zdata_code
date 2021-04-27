import os
import tarfile
import pandas as pd
import numpy as np
import pexpect
import time

server_list = [
    '10.44.2.7',
    '10.44.2.10',
    '10.44.2.11',
    '10.44.2.9',
    '10.44.2.8',
    '10.44.2.5',
    '10.44.2.4',
    '10.44.2.6',
    '10.44.2.13',
    '10.44.2.12',
    '10.44.0.212',
    '10.44.0.130',
    '10.44.0.129',
    '10.44.0.131',
    '10.44.0.203',
    '10.44.0.204',
    '10.44.0.141',
    '10.44.0.181',
]

def today():
    return pd.Timestamp.now().strftime('%Y%m%d')

def now():
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

def unzip_tar():
    '''
    1. MOVE ALL THE DATA IN TAR FILE INTO DATA PATH
    2. MOVE TAR FILE TO PROCESSED
    '''

    print('MOVE THE DOWNLOADED TAR FILE TO ALI_DOWNLOAD')
    os.system(f"mv /app/download/A5_data*.tar /app/auto_pipeline_main/ali_download")

    print(f"SCANNING FOR NEW TAR FILES AT {now()} IN {tar_path}")
    if len(os.listdir(tar_path)) == 2 and 'processed' in os.listdir(tar_path):
        
        print("UNTARRING TAR FILE...")
        tar_filename = [f for f in os.listdir(tar_path) if '.tar' in f][0]
        os.system(f"tar -xvf {tar_path}/{tar_filename} -C {tar_path}")
        os.system(f"mv {tar_path}/{tar_filename} {tar_path}/processed")

        print(f'DECOMPRESSED TAR FILE {tar_filename}')

    else:
        raise "WRONG NUMBER OF ITEMS IN TAR PATH"

def create_tar(server_mapper):
    '''
    HELPER FUNCTION FOR CREATING TAR FILE BY ITEM LIST AND SERVER MAPPER
    '''
    print('CREATING TAR FILES FOR TODAYS ITEMS...')
    for server in server_mapper.server.unique():

        # CREATE TAR FILE FOR EACH SERVER NAME
        tar_file_path = f"{tar_upload_path}/{server}_{today()}.tar"
        tar_file = tarfile.open(tar_file_path, mode="w")

        itemlist_filepath = os.listdir(itemlist_path)[0]
        tar_file.add(f"{itemlist_path}/{itemlist_filepath}")
        print(f"ADDED TO {itemlist_filepath} SERVER")

        # ADD CORRESPONDING DATA FILES TO EACH TAR FILE
        for item in server_mapper[server_mapper.server == server].item.unique():
            tar_file.add(f"{data_path}/{str(item)}{data_format}")
            print(f"{item} ADDED TO {server}")

        tar_file.close()

    print('JOB COMPLETED.')

def generate_server_tar():
    '''
    GET SERVER MAPPER, CREATE TAR FILES, AND MOVE TAR FILES READY FOR UPLOAD
    '''
    # GET ITEM LIST
    item_list = [i[:i.index('.')] for i in os.listdir(data_path)]

    print('CREATING NEW MAPPER...')
    server_mapper = pd.DataFrame({
                    'item':item_list,
                    'server':[np.random.choice(server_list) for _ in item_list]
                    })

    print('CREATING TAR FILES FOR ALL ITEMS...')
    create_tar(server_mapper=server_mapper)

    server_mapper.to_csv(f"{mapper_path}/server_mapper_{today()}.csv",index=False)
    print('SERVER MAPPER EXPORTED.')
    print('SERVER DISTRIBUTION: ')
    print(server_mapper.server.value_counts())

def scp_to_server():

    for filename in [tar for tar in os.listdir(tar_upload_path) if tar != 'processed']:

        server = filename[:filename.index('_')]

        if server != '10.44.2.10':
            child = pexpect.spawn(f"scp {tar_upload_path}/{filename} gaiprd@{server}:/app/auto_pipeline/download/")
            child.expect(f"gaiprd@{server}'s password:")
            child.sendline("nL1C39aGSC")
            child.expect(pexpect.EOF, timeout=None)
            print(f'{filename} SENT TO {server}:/app/auto_pipeline/download AT {now()}')
            os.system(f"mv {tar_upload_path}/{filename} {tar_upload_path}/processed")

        else:
            os.system(f"mv {tar_upload_path}/{filename} /app/download")
            print(f"{filename} MOVED TO /app/download")

#* TAKE UPLOADED TAR FILE, UNZIP, SEND TO 18 ALI CLOUD AT /app/download/
download_path = '/app/download'
tar_path = '/app/auto_pipeline_main/ali_download'
tar_upload_path = '/app/auto_pipeline_main/ali_upload'
data_path = '/app/auto_pipeline_main/ali_download/app2/feature_pipeline/feature_classified'
itemlist_path = '/app/auto_pipeline_main/ali_download/app2/feature_pipeline/data/item'
mapper_path = f"/app/auto_pipeline_main/server_mapper"
data_format = '.csv.gz'

print('START SCANNING FOR INCOMING TAR FILE FROM 38 EVERY SECOND...')
while True:

    if len(os.listdir(download_path)) == 2 and len([f for f in os.listdir(download_path) if f.startswith('A5') and f.endswith('.tar')]) == 1:

        print(f'FILE FOUND, WAITING FOR IT TO UPLOAD AT {now()}')
        tar_filename = [f for f in os.listdir(download_path) if '.tar' in f][0]
        
        # WAITING UNTIL LAST MODIFIED TIME IS 5 MINUTES AWAY FROM NOW
        while True:
            raw_time = time.localtime(os.path.getmtime(f'{download_path}/{tar_filename}'))
            file_modified_time = time.strftime('%Y-%m-%d %H:%M:%S', raw_time)
            sec_from_modified = (pd.to_datetime(now()) - pd.to_datetime(file_modified_time)).seconds
            if sec_from_modified > 300:
                break
        
        print(f'TAR PIPELINE STARTING AT {now()}')
        unzip_tar()

        print('GENERATE TAR FILES FOR DIFFERENT SERVERS...')
        generate_server_tar()

        print('SCP TO SERVERS...')
        scp_to_server()

        print('REMOVING ALL UNZIPPED DATA...')
        os.system('rm -r /app/auto_pipeline_main/ali_download/app2')

        print(f'UNTAR PIPELINE COMPLETED AT {now()}')
        
    time.sleep(1)