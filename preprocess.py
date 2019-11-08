import pandas as pd
import time

data = load_csv("activity_u00.csv")

data['time'] = data['timestamp'].apply(lambda x: time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(x)))

data.head()