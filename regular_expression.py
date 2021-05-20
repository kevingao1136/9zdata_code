import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
pd.options.display.max_colwidth = None
pd.options.display.max_columns = None

def convert_string(string, year, verbose=False):
    '''
    Parameters:
        string - string with format:
        #* mmdd-mmdd, yyyymmdd-yyyymmdd, mdd-mdd, mdd-mmdd, mmd-mdd
        #* mm.dd-mm.dd, m.dd-m.dd, m.dd-mm.d, mm.d-m.dd
    Returns:
        start_date, end_date as tuple of timestamps (if string passes condition), ELSE raw string
    '''
    
    if re.match("\d{4}[-]{1}\d{4}", string): # mmdd-mmdd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=year, month=int(string[:2]), day=int(string[2:4])))
            end_date = pd.to_datetime(datetime.datetime(year=year, month=int(string[5:7]), day=int(string[7:9])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string

    elif re.match("\d{3}[-]\d{3}", string): # mdd-mdd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=year, month=int(string[0]), day=int(string[1:3])))
            end_date = pd.to_datetime(datetime.datetime(year=year, month=int(string[4:5]), day=int(string[5:7])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string


    elif re.match("\d{8}[-]{1}\d{8}", string): # yyyymmdd-yyyymmdd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[4:6]), day=int(string[6:8])))
            end_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[13:15]), day=int(string[15:17])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string


    elif re.match("\d{2}[./]{1}\d{2}[-]{1}\d{2}[./]{1}\d{2}", string): # mm.dd-mm.dd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[:2]), day=int(string[3:5])))
            end_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[6:8]), day=int(string[9:11])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string


    elif re.match("\d{1}[.]{1}\d{2}[-]{1}\d{1}[.]{1}\d{2}",string): # m.dd-m.dd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[0]), day=int(string[2:4])))
            end_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[5]), day=int(string[7:9])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string


    elif re.match("\d{1}[.]{1}\d{2}[-]{1}\d{2}[.]{1}\d{1}",string): # m.dd-mm.d
        try:
            start_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[0]), day=int(string[2:4])))
            end_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[5:7]), day=int(string[8])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string

    elif re.match("\d{2}[.]{1}\d{1}[-]{1}\d{1}[.]{1}\d{2}",string): # mm.d-m.dd
        try:
            start_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[:2]), day=int(string[3])))
            end_date = pd.to_datetime(datetime.datetime(year=2019, month=int(string[5]), day=int(string[7:])))
        except Exception as error:
            if verbose: print(f"{string} HAS ERROR: {error}")
            return string
        
    else: # UNSUPPORTED FORMAT
        if verbose: print(f"INVALID FORMAT: {string}")
        return string

    return start_date, end_date

# TEST CASES
for s in ['0101-0228','02.02-03.30','20190102-20190807','228-528','2.28-3.12','2.28-12.1','12.1-1.12']:
    print(convert_string(s, year=2019, verbose=True))