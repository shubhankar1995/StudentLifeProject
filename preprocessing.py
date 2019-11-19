# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:24:52 2019

@author: Kovid
"""

import numpy as np
import pandas as pd
import collections

def activity():
    counter = 0
    df_list = []
    while counter<=60:
        counter+=1
        temp_list = []
        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)
        temp_list.append(user_id)

        try:
            df = pd.read_csv('StudentLife_Dataset/Inputs/sensing/activity/activity_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_activities = df[' activity inference'].tolist()
        count_activity = collections.Counter(list_activities)
        most_freq = count_activity.most_common(1)
        temp_list.append(most_freq[0][0])
        average = (count_activity[1] + count_activity[2])/float(len(list_activities))
        temp_list.append(round(average, 4))
        df_list.append(temp_list)
    dfObj = pd.DataFrame(df_list)
    dfObj.columns = ['uid','most_freq_activity','proportion_running_walking']
    return(dfObj)

def phonelock():
    counter = 0
    df_list = []
    while counter<=60:
        counter+=1
        temp_list = []
        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)
#         temp_list.append(user_id)
        try:
            df = pd.read_csv('StudentLife_Dataset/Inputs/sensing/phonelock/phonelock_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_lock_duration = []
        for index,item in df.iterrows():
            list_lock_duration.append(item['end']-item['start'])
        temp_list.append(np.mean(list_lock_duration))
        temp_list.append(np.std(list_lock_duration))
        df_list.append(temp_list)
    dfObj = pd.DataFrame(df_list)
    dfObj.columns = ['mean_lock_duration','std_lock']
    return(dfObj)

def coversation():
    counter = 0
    df_list = []
    while counter<=60:
        counter+=1
        temp_list = []
        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)
#         temp_list.append(user_id)
        try:
            df = pd.read_csv('StudentLife_Dataset/Inputs/sensing/conversation/conversation_' + user_id + '.csv',index_col=False)
        except:
            continue
        total_conversations = 0
        total_conversations=len(df)
        temp_list.append(total_conversations)

        list_timeDiff = []
        for index,item in df.iterrows():
            list_timeDiff.append(item[' end_timestamp']-item['start_timestamp'])
        mean_call_duation = np.mean(list_timeDiff)
        std_call_duration = np.std(list_timeDiff)
        temp_list.append(mean_call_duation)
        temp_list.append(std_call_duration)
        df_list.append(temp_list)
    dfObj = pd.DataFrame(df_list)
    dfObj.columns = ['total_conversations','mean_call_duration','std_call_duration']
    return(dfObj)

def dark():
    counter = 0
    df_list = []
    while counter<=60:
        counter+=1
        temp_list = []
        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)
#         temp_list.append(user_id)
        try:
            df = pd.read_csv('StudentLife_Dataset/Inputs/sensing/dark/dark_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_dark_duration = []
        for index,item in df.iterrows():
            list_dark_duration.append(item['end']-item['start'])
        temp_list.append(np.mean(list_dark_duration))
        temp_list.append(np.std(list_dark_duration))
        df_list.append(temp_list)
    dfObj = pd.DataFrame(df_list)
    dfObj.columns = ['mean_dark_duration','std_dark']
    return(dfObj)

def phonecharge():
    counter = 0
    df_list = []
    while counter<=60:
        counter+=1
        temp_list = []
        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)
#         temp_list.append(user_id)
        try:
            df = pd.read_csv('StudentLife_Dataset/Inputs/sensing/phonecharge/phonecharge_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_charge_duration = []
        for index,item in df.iterrows():
            list_charge_duration.append(item['end']-item['start'])
        temp_list.append(np.mean(list_charge_duration))
        temp_list.append(np.std(list_charge_duration))
        df_list.append(temp_list)
    dfObj = pd.DataFrame(df_list)
    dfObj.columns = ['mean_charge_duration','std_charge']
    return(dfObj)


df1 = activity() #most_freq_activity , proportion_running_walking
df2 = phonelock() #mean_lock_duration , std_lock
df3 = coversation()# total_calls , total_conversations , mean_call_duration , std_call_duration
df4 = dark() #'mean_dark_duration','std_dark'
df5 = phonecharge()#'mean_charge_duration','std_charge'
result = pd.concat([df1, df2, df3, df4,df5], axis=1)
print(result)
