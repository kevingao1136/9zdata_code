# nohup
nohup python -u ETL.py > logs/ETL.log  2>&1&
nohup python -u ETL.py >> logs/ETL.log  2>&1& # APPEND

ps -o cmd fp <PID> # FIND COMMAND BY PID

# scp
scp lunar_days.csv gaiprd@10.44.2.10:/app/kevin_workspace/
scp *.csv dmpprd@10.82.28.136:/app/scripts/scai/ai_data/data/scai_ai_forecast_prmt/20210129
scp *.csv dmpprd@10.82.28.136:/app/scripts/scai/ai_data/data/scai_ai_mode_info/20210129

# check server size
df -h

# linux count files
find -name "*.csv" |wc -l
ls | wc -l

# ALI CLOUD
WTCCN-SA-GAIMC2
10.44.2.10
gaiprd
nL1C39aGSC

# check dir size
du -h

# change pip url
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# kill multiple processes by file name
kill -9 `ps -ef | grep train_model_parallel.py | grep -v grep | awk '{print $2}'`

# FIND PID BY FILE NAME
ps aux | grep -i 'search-term'

find output0304/ -name 100009992* -exec cp {} chosen_output0304/ \;
