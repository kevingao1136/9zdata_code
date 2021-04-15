#%%
import os
import tarfile
import pandas as pd
import numpy as np

def today():
    return pd.Timestamp.now().strftime("%Y-%m-%d")

def create_tar(item_list, server_mapper):

    print('CREATING TAR FILES FOR ITEM LIST: ' + str(item_list))
    for server in server_mapper[server_mapper.item.isin(item_list)].server.unique():

        # CREATE TAR FILE FOR EACH SERVER NAME
        tar_file_path = f"{tar_path}{server}_{today()}.tar"
        if os.path.exists(tar_file_path):
            tar_file_path = tar_file_path[:tar_file_path.index('.tar')] + '_new' + '.tar'

        tar_file = tarfile.open(tar_file_path, mode="w")

        # ADD CORRESPONDING DATA FILES TO EACH TAR FILE
        for item in server_mapper[server_mapper.server == server].item.unique():
            tar_file.add(f"{data_path}{str(item)}.csv")
            print(f"{item} ADDED TO {server}")

        tar_file.close()
    print('JOB COMPLETED.')

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

# DEFINE PATHS
data_path = '/app/python-scripts/kevin_workspace/all_data/'
mapper_path = '/app/python-scripts/kevin_workspace/ali_upload/server_mapper.csv'
tar_path = f'/app/python-scripts/kevin_workspace/ali_upload/tar_upload/'

# GET ITEM LIST
item_list = [i[:i.index('.')] for i in os.listdir(data_path)]
if not os.path.exists(mapper_path):

    print('CREATING NEW MAPPER...')
    server_mapper = pd.DataFrame({
                    'item':item_list,
                    'server':[np.random.choice(server_list) for _ in item_list]
                    })

    server_mapper.to_csv(mapper_path,index=False)
    print('SERVER MAPPER EXPORTED.')
    print('SERVER DISTRIBUTION: ')
    print(server_mapper.server.value_counts())

    print('CREATING TAR FILES FOR ALL ITEMS...')
    create_tar(item_list=item_list, server_mapper=server_mapper) # ALL ITEMS

else:
    print('LOADING EXISTING MAPPER...')
    server_mapper = pd.read_csv(mapper_path)
    new_item_list = [i[:i.index('.')] for i in os.listdir(data_path)]
    old_item_list = [str(i) for i in list(server_mapper.item.unique())]
    add_item_list = [i for i in new_item_list if i not in old_item_list]
    remove_item_list = [i for i in old_item_list if i not in new_item_list]

    if add_item_list:
        print('ADDED ITEMS: ' + str(add_item_list) + ' ------> ADDING TO EXISITING MAPPER...')
        new_server_mapper = pd.DataFrame({
                                'item':add_item_list,
                                'server':[np.random.choice(server_list) for _ in add_item_list]
                                })

        updated_server_mapper = pd.concat([new_server_mapper, server_mapper])
        updated_server_mapper.to_csv(mapper_path,index=False)
        print('UPDATED MAPPER EXPORTED.')
        print('UPDATED SERVER DISTRIBUTION:')
        print(updated_server_mapper.server.value_counts())

        print('CREATING TAR FILES FOR ADDED ITEMS...')
        create_tar(item_list=add_item_list, server_mapper=new_server_mapper)
    
    else:
        print('NO ADDED ITEMS.')

    if remove_item_list:
        print('REMOVED ITEMS: ' + str(remove_item_list) + ' ------> REMOVING FROM EXISITING MAPPER...')
        new_server_mapper = server_mapper[~server_mapper.item.isin(remove_item_list)]
        new_server_mapper.to_csv(mapper_path,index=False)
        print('UPDATED MAPPER EXPORTED.')
        print('UPDATED SERVER DISTRIBUTION:')
        print(new_server_mapper.server.value_counts())

# %%