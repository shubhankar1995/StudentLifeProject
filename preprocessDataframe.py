import pandas as pd
import time
import numpy as np

from scipy import stats
import scipy

def greets(tm):
    # SPlit data into day segments
    if(tm.hour >= 6 and tm.hour<18):
        return "day"
    elif(tm.hour >= 18 and tm.hour <=24):
        return "evening"
    else:
        return "night"

def audio(path):
    data = pd.read_csv(path)
    # Converting to DateTime format
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_mode').reset_index()
    df_group.dropna(inplace=True)
    # New Columns
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_mode','datetime_mode'])
    return df_group

def activity(path):
    data = pd.read_csv(path)
    # Converting to DateTime format
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_mode').reset_index()
    df_group.dropna(inplace=True)
    # New Columns
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_mode','datetime_mode'])
    return df_group

def gps(path):
    data = pd.read_csv(path, usecols = ['speed'])
    # Converting to DateTime format
    data['travelstate'] = data['speed']
    data = data.drop(columns=['speed'])
    data['timestamp'] = data.index.values.astype(int)
    data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    data = data.dropna()
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).agg(lambda x:x.value_counts().index[0]).add_suffix('_mode').reset_index()
    df_group.dropna(inplace=True)
    # New Columns
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_mode','datetime_mode'])
    return df_group

def wifi(path):
    data = pd.read_csv(path, usecols = ['time','BSSID','level'])
    # Converting to DateTime format
    data['timestamp'] = data['time'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).agg(lambda x:len(list(np.unique(x)))).add_suffix('_mode').reset_index()
    df_group['wifi_level_std'] = data.groupby(['date','greets']).agg(lambda x: x.std()).add_suffix('_std').reset_index()['level_std']
    df_group['wifi_level_mean'] = data.groupby(['date','greets']).agg(lambda x: x.mean()).add_suffix('_mean').reset_index()['level_mean']
    df_group.dropna(inplace=True)
    # New Columns
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_mode','datetime_mode','time_mode','level_mode'])
    return df_group

def bt(path):
    data = pd.read_csv(path, usecols = ['time','MAC','level'])
    # Converting to DateTime format
    data['timestamp'] = data['time'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).agg(lambda x:len(list(np.unique(x)))).add_suffix('_mode').reset_index()
    df_group['bt_level_std'] = data.groupby(['date','greets']).agg(lambda x: x.std()).add_suffix('_std').reset_index()['level_std']
    df_group['bt_level_mean'] = data.groupby(['date','greets']).agg(lambda x: x.std()).add_suffix('_mean').reset_index()['level_mean']
    df_group.dropna(inplace=True)
    # New Columns
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','timestamp_mode','datetime_mode','time_mode','level_mode'])
    return df_group

def conversation(path):
    data = pd.read_csv(path)
    # Converting to DateTime format
    data["duration(s)"] = data[" end_timestamp"] - data["start_timestamp"]
    data = data.drop(" end_timestamp" , axis=1)
    data['timestamp']= data['start_timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    data = data.set_index('date')
    data = data.dropna()
    data.reset_index()
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).mean().add_suffix('_avg_conversation').reset_index()
    df_group2 = data.groupby(['date','greets']).count().add_suffix('_avg').reset_index()
    # New Columns
    df_group["frequency_conversation"] = df_group2["datetime_avg"]
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','start_timestamp_avg_conversation'])
    df_group.dropna(inplace=True)
    return df_group

def dark(path):
    data = pd.read_csv(path)
    # Converting to DateTime format
    data["duration(s)"] = data["end"] - data["start"]
    data = data.drop("end" , axis=1)
    data['timestamp']= data['start'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    data = data.set_index('date')
    data = data.dropna()
    data.reset_index()
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).mean().add_suffix('_avg_dark').reset_index()
    df_group2 = data.groupby(['date','greets']).count().add_suffix('_avg').reset_index()
    # New Columns
    df_group["frequency_dark"] = df_group2["datetime_avg"]
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','start_avg_dark'])
    df_group.dropna(inplace=True)
    return df_group

def phoneLock(path):
    data = pd.read_csv(path)
    # Converting to DateTime format
    data["duration(s)"] = data["end"] - data["start"]
    data = data.drop("end" , axis=1)
    data['timestamp']= data['start'].apply(lambda x: time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['timestamp'])
    data["greets"] = data["datetime"].apply(greets)
    data["date"]=data["datetime"].dt.date
    data = data.set_index('date')
    data = data.dropna()
    data.reset_index()
    # Group data based on date segments
    df_group=data.groupby(['date','greets']).mean().add_suffix('_avg_phoneLock').reset_index()
    df_group2 = data.groupby(['date','greets']).count().add_suffix('_avg').reset_index()
    # New Columns
    df_group["frequency_phoneLock"] = df_group2["datetime_avg"]
    df_group['date_greets'] = df_group['date'].map(str) + " " + df_group['greets']
    df_group = df_group.drop(columns = ['date','greets','start_avg_phoneLock'])
    df_group.dropna(inplace=True)
    return df_group

#------------------------------Merging------------------------------------------

#Get the file path
source_path = "D:\\UNSW\\Term 3\\Machine Learning\\Project\\StudentLife_Dataset\\Inputs\\sensing\\"

# df_audio = audio(source_path + "audio\\audio_u00.csv")

# Generating X features

#Get the student data
stdId = 'u00'
print("Processing student", stdId)
print("activity")
df_activity = activity(source_path + "activity\\activity_u00.csv")
print("gps")
df_gps = gps(source_path + "gps\\gps_u00.csv")
print("wifi")
df_wifi = wifi(source_path + "wifi\\wifi_u00.csv")
print("bt")
df_bt = bt(source_path + "bluetooth\\bt_u00.csv")
print("conversation")
df_conv = conversation(source_path + "conversation\\conversation_u00.csv")
print("dark")
df_dark = dark(source_path + "dark\\dark_u00.csv")
print("phonelock")
df_phonelock = phoneLock(source_path + "phonelock\\phonelock_u00.csv")

# Merging the student data

print("Merging", stdId)
df_merge_all = df_activity.merge(df_gps, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_wifi, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_bt, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_conv, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_dark, left_on="date_greets", right_on="date_greets")
df_merge_all = df_merge_all.merge(df_phonelock, left_on="date_greets", right_on="date_greets")
df_merge_all['student_id'] = stdId
# df_merge = df_audio.merge(df_activity, left_on="date_greets", right_on="date_greets")


#Looping over the studnet records
for i in range(1, 60):
    if (i < 10):
        stdId = 'u0' + str(i)
    else:
        stdId = 'u' + str(i)
    try:
        # df_audio = audio(source_path + "audio\\audio_u00.csv")
        # Processing student
        # Get the activity records
        df_activity = activity(source_path + "activity\\activity_"+ stdId +".csv")
        # get the gps records
        df_gps = gps(source_path + "gps\\gps_"+ stdId +".csv")
        # get the wifi records
        df_wifi = wifi(source_path + "wifi\\wifi_"+ stdId +".csv")
        # Get the bluetooth records
        df_bt = bt(source_path + "bluetooth\\bt_"+ stdId +".csv")
        # Get conversation records
        df_conv = conversation(source_path + "conversation\\conversation_"+ stdId +".csv")
        # Get dark records
        df_dark = dark(source_path + "dark\\dark_"+ stdId +".csv")
        # Get phonelock records
        df_phonelock = phoneLock(source_path + "phonelock\\phonelock_"+ stdId +".csv")
        # Merging all records
        df_merge = df_activity.merge(df_gps, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_wifi, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_bt, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_conv, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_dark, left_on="date_greets", right_on="date_greets")
        df_merge = df_merge.merge(df_phonelock, left_on="date_greets", right_on="date_greets")
        df_merge['student_id'] = stdId
        df_merge_all = pd.concat([df_merge_all, df_merge])
        print("merge complete")
    except FileNotFoundError:
        pass
    except:
        print("Failed to process", stdId)


# Merge teh X- Features
df_merge_all.to_csv("merge_all_x_features.csv", index=False)

print("merge y variables")

#Get the Y-output values
df_FlourishingScale = pd.read_csv("D:\\UNSW\\Term 3\\Machine Learning\\Project\\fs.csv", usecols = ['uid','fs_avg'])
df_PanasScale = pd.read_csv("D:\\UNSW\\Term 3\\Machine Learning\\Project\\panas.csv", usecols = ['uid','positive_avg','negative_avg'])

# Merge the Y-output values with X-Features
df_merge_all_x_y =  df_merge_all.merge(df_FlourishingScale, left_on="student_id", right_on="uid")
df_merge_all_x_y =  df_merge_all_x_y.merge(df_PanasScale, left_on="student_id", right_on="uid")

# Generating the file
df_merge_all_x_y.to_csv("merge_all_x_y.csv", index=False)