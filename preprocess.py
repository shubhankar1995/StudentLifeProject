import pandas as pd
import time
from scipy import stats
import scipy

data = pd.read_csv("/home/mustafa/StudentLife_Dataset/Inputs/sensing/activity/activity_u01.csv")

data['time'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(x)))
data['datetime'] = pd.to_datetime(data['time'])
data.head()

data = data.set_index('datetime')

df_group = data.groupby(pd.TimeGrouper(level='datetime', freq='10T'))[' activity inference'].agg( lambda x: scipy.stats.mode(x)[0])
df_group.dropna(inplace=True)
df_group = df_group.to_frame().reset_index()


print(df_group)
