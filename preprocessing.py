import numpy as np
import glob
import pandas as pd
import time

path = 'C:\\Users\Kovid\COMP9417\StudentLife_Dataset'
    
folders = []
csvfiles = []


# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for folder in d:
#         folders.append(os.path.join(r, folder))
        p = os.path.join(r, folder)
        os.chdir(os.path.join(r, folder))
        file = (glob.glob('*.{}'.format('csv')))
        for f in file:
            csvfiles.append(p+'\\'+f)
               
df_fs = pd.read_csv(csvfiles[0])
df_panas = pd.read_csv(csvfiles[1])
inputcsv = csvfiles[2:]

df_activity = [_ for _ in range(49)]   
df_audio = [_ for _ in range(49)] 
df_bluetooth = [_ for _ in range(49)] 
df_conversation = [_ for _ in range(49)] 
df_dark = [_ for _ in range(49)]   
df_gps = [_ for _ in range(49)] 
df_phonecharge = [_ for _ in range(49)] 
df_phonelock = [_ for _ in range(49)] 
df_wifi = [_ for _ in range(49)]   
df_wifi_location = [_ for _ in range(49)] 

def clean_time(df):
    try:
        df['time'] = df['timestamp'].apply(lambda x: time.strftime("%H:%M:%S", time.localtime(x)))
        df['date'] = df['timestamp'].apply(lambda x: time.strftime("%d-%m-%Y ", time.localtime(x)))
        return df
    except KeyError:
        pass
    try:
        df['time'] = df['time'].apply(lambda x: time.strftime("%H:%M:%S", time.localtime(x)))
        df['date'] = df['time'].apply(lambda x: time.strftime("%d-%m-%Y ", time.localtime(x)))
        return df
    except KeyError:
        pass
    try:
        df['time'] = df['start_timestamp'].apply(lambda x: time.strftime("%H:%M:%S", time.localtime(x)))
        df['date'] = df['start_timestamp'].apply(lambda x: time.strftime("%d-%m-%Y ", time.localtime(x)))
        return df
    except KeyError:
        pass
    try:
        df['time'] = df['end_timestamp'].apply(lambda x: time.strftime("%H:%M:%S", time.localtime(x)))
        df['date'] = df['end_timestamp'].apply(lambda x: time.strftime("%d-%m-%Y ", time.localtime(x)))
        return df
    except KeyError:
        pass
    return df

def clean():
    ctr = 0
    for i in range(len(inputcsv)):
        if ctr >= 49:
            ctr = 0
        if 'activity' in inputcsv[i]:
            df_activity[ctr] = pd.read_csv(inputcsv[i])
            df_activity[ctr].replace('', np.nan, inplace=True)
            df_activity[ctr] = clean_time(df_activity[ctr])
            print(inputcsv[i])
            df_activity[ctr].head()
        if 'audio' in inputcsv[i]:
            df_audio[ctr] = pd.read_csv(inputcsv[i])
            df_audio[ctr].replace('', np.nan, inplace=True)
            df_audio[ctr] = clean_time(df_audio[ctr])
            print(inputcsv[i])
            df_audio[ctr].head()
        if 'bluetooth' in inputcsv[i]:
            df_bluetooth[ctr] = pd.read_csv(inputcsv[i])
            df_bluetooth[ctr].replace('', np.nan, inplace=True)
            df_bluetooth[ctr] = clean_time(df_bluetooth[ctr])
            print(inputcsv[i])
            df_bluetooth[ctr].head()
        if 'conversation' in inputcsv[i]:
            df_conversation[ctr] = pd.read_csv(inputcsv[i])
            df_conversation[ctr].replace('', np.nan, inplace=True)
            df_conversation[ctr] = clean_time(df_conversation[ctr])
            print(inputcsv[i])
            df_conversation[ctr].head()
        if 'dark' in inputcsv[i]:
            df_dark[ctr] = pd.read_csv(inputcsv[i])
            df_dark[ctr].replace('', np.nan, inplace=True)
            df_dark[ctr] = clean_time(df_dark[ctr])
            print(inputcsv[i])
            df_dark[ctr].head()
        if 'gps' in inputcsv[i]:
            df_gps[ctr] = pd.read_csv(inputcsv[i])
            df_gps[ctr].replace('', np.nan, inplace=True)
            df_gps[ctr] = clean_time(df_gps[ctr])
            print(inputcsv[i])
            df_gps[ctr].head()
        if 'phonecharge' in inputcsv[i]:
            df_phonecharge[ctr] = pd.read_csv(inputcsv[i])
            df_phonecharge[ctr].replace('', np.nan, inplace=True)
            df_phonecharge[ctr] = clean_time(df_phonecharge[ctr])
            print(inputcsv[i])
            df_phonecharge[ctr].head()
        if 'phonelock' in inputcsv[i]:
            df_phonelock[ctr] = pd.read_csv(inputcsv[i])
            df_phonelock[ctr].replace('', np.nan, inplace=True)
            df_phonelock[ctr] = clean_time(df_phonelock[ctr])
            print(inputcsv[i])
            df_phonelock[ctr].head()
        if 'wifi' in inputcsv[i]:
            df_wifi[ctr] = pd.read_csv(inputcsv[i])
            df_wifi[ctr].replace('', np.nan, inplace=True)
            df_wifi[ctr] = clean_time(df_wifi[ctr])
            print(inputcsv[i])
            df_wifi[ctr].head()
        if 'wifi_location' in inputcsv[i]:
            df_wifi_location[ctr] = pd.read_csv(inputcsv[i])
            df_wifi_location[ctr].replace('', np.nan, inplace=True)
            df_wifi_location[ctr] = clean_time(df_wifi_location[ctr])
            print(inputcsv[i])
            df_wifi_location[ctr].head()
        ctr += 1

df_panas.head()
df_fs.head()
# clean()

