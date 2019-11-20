import pandas as pd
import time
import numpy as np

from scipy import stats
import scipy

def audio(path):
    data = pd.read_csv(path)
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    def greets(tm):
        if(tm.hour >= 6 and tm.hour<18):
            return "day"
        elif(tm.hour >= 18 and tm.hour <=24):
            return "evening"
        else:
            return "night"
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_avg').reset_index()
    df_group.dropna(inplace=True)
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_avg','datetime_avg'])
    return df_group

def activity(path):
    data = pd.read_csv(path)
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    def greets(tm):
        if(tm.hour >= 6 and tm.hour<18):
            return "day"
        elif(tm.hour >= 18 and tm.hour <=24):
            return "evening"
        else:
            return "night"
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_avg').reset_index()
    df_group.dropna(inplace=True)
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_avg','datetime_avg'])
    return df_group

def gps(path):
    data = pd.read_csv(path, usecols = ['speed'])
    data['travelstate'] = data['speed']
    data = data.drop(columns=['speed'])
    data['timestamp'] = data.index.values.astype(int)
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    def greets(tm):
        if(tm.hour >= 6 and tm.hour<18):
            return "day"
        elif(tm.hour >= 18 and tm.hour <=24):
            return "evening"
        else:
            return "night"
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    data = data.dropna()
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_avg').reset_index()
    df_group.dropna(inplace=True)
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_avg','datetime_avg'])
    return df_group

def wifi(path):
    data = pd.read_csv(path, usecols = ['time','BSSID'])
    data['timestamp'] = data['time'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    def greets(tm):
        if(tm.hour >= 6 and tm.hour<18):
            return "day"
        elif(tm.hour >= 18 and tm.hour <=24):
            return "evening"
        else:
            return "night"
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    df_group=data.groupby(['date','greets']).agg(lambda x:len(list(np.unique(x)))).add_suffix('_avg').reset_index()
    df_group.dropna(inplace=True)
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_avg','datetime_avg','time_avg'])
    return df_group

def bt(path):
    data = pd.read_csv(path, usecols = ['time','MAC'])
    data['timestamp'] = data['time'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    def greets(tm):
        if(tm.hour >= 6 and tm.hour<18):
            return "day"
        elif(tm.hour >= 18 and tm.hour <=24):
            return "evening"
        else:
            return "night"
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    df_group=data.groupby(['date','greets']).agg(lambda x:len(list(np.unique(x)))).add_suffix('_avg').reset_index()
    df_group.dropna(inplace=True)
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_avg','datetime_avg','time_avg'])
    return df_group



#------------------------------Merging------------------------------------------

source_path = "D:\\UNSW\\Term 3\\Machine Learning\\Project\\StudentLife_Dataset\\Inputs\\sensing\\"

# df_audio = audio(source_path + "audio\\audio_u00.csv")

stdId = 'u00'
print("Processing student", stdId)
df_activity = activity(source_path + "activity\\activity_u00.csv")
df_gps = gps(source_path + "gps\\gps_u00.csv")
df_wifi = wifi(source_path + "wifi\\wifi_u00.csv")
df_bt = bt(source_path + "bluetooth\\bt_u00.csv")

df_merge_all = df_activity.merge(df_gps, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_wifi, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_bt, left_on="date_greets", right_on="date_greets")
df_merge_all['student_id'] = stdId
# df_merge = df_audio.merge(df_activity, left_on="date_greets", right_on="date_greets")


for i in range(1, 60):
    if (i < 10):
        stdId = 'u0' + str(i)
    else:
        stdId = 'u' + str(i)
    try:
        # df_audio = audio(source_path + "audio\\audio_u00.csv")
        print("Processing student", stdId)
        print("activity")
        df_activity = activity(source_path + "activity\\activity_"+ stdId +".csv")
        print("gps")
        df_gps = gps(source_path + "gps\\gps_"+ stdId +".csv")
        print("wifi")
        df_wifi = wifi(source_path + "wifi\\wifi_"+ stdId +".csv")
        print("bt")
        df_bt = bt(source_path + "bluetooth\\bt_"+ stdId +".csv")
        df_merge = df_activity.merge(df_gps, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_wifi, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_bt, left_on="date_greets", right_on="date_greets")
        df_merge['student_id'] = stdId
        df_merge_all = pd.concat([df_merge_all, df_merge])
        print("merge complete")
    except FileNotFoundError:
        pass
    except:
        print("Failed to process", stdId)

# df_activity = activity("D:\\UNSW\\Term 3\\Machine Learning\\Project\\StudentLife_Dataset\\Inputs\\sensing\\activity\\activity_u00.csv")

# df_audio.to_csv("audio.csv", index=False)

# df_audio.dropna(inplace=True)
# df_audio.to_csv("audio_dropNa.csv", index=False)

# df_merge = df_audio.merge(df_activity, left_on="date_greets", right_on="date_greets")
# df_merge['student_id'] = 'u00'

df_merge_all.to_csv("merge_all.csv", index=False)
