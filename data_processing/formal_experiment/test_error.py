#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:20:09 2022

@author: chenglinlin

function: analyze data and draw pictures
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'

participant_list=[]

for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
# errorpath_list_cal=[]
# for i in range(21):
#     x=outputroot+'processed_data/error_l2cs_calibration/error_p'+str(i+1)+'.csv'
#     errorpath_list_cal.append(x)
    
errorpathcal_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_eth_cal/error_p'+str(i+1)+'.csv'
    errorpathcal_list.append(x)
    
errorpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_eth_baseline/error_p'+str(i+1)+'.csv'
    errorpath_list.append(x)
    
errorpathcal_list=[]
for i in [1,3,11,12,13,14,17,18,19,20]:
    x=outputroot+'processed_data/error_l2cs_cal/error_4k_p'+str(i+1)+'.csv'
    errorpathcal_list.append(x)
    
errorpath_list=[]
for i in [1,3,11,12,13,14,17,18,19,20]:
    x=outputroot+'processed_data/error_l2cs/error_4k_p'+str(i+1)+'.csv'
    errorpath_list.append(x)    
# errorpath4k_list=[]
# for x in participant4k_list:
#     x=outputroot+'processed_data/error_eth_baseline/error_4k_'+x+'.csv'
#     errorpath4k_list.append(x)
    

participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]
# df['participant0']=pd.read_csv(errorpath_list[0])['0']
df = pd.DataFrame()
df_cal = pd.DataFrame()

for i in range(21):
    column_name='participant'+str(i)
    df[column_name]=pd.read_csv(errorpath_list[i])['0']
    df_cal[column_name]=pd.read_csv(errorpathcal_list[i])['0']
    
df = pd.DataFrame()
df_cal = pd.DataFrame()

for i in range(10):
    column_name='participant'+str(i)
    df[column_name]=pd.read_csv(errorpath_list[i])['0']
    df_cal[column_name]=pd.read_csv(errorpathcal_list[i])['0']
    
    
    
for i in range(10):
    column_name='participant4k'+str(i+11)
    df[column_name]=pd.read_csv(errorpath4k_list[i])['0']
    
    
    
    
# for i in range(15):
#     column_name='participant'+str(i)
#     plt.plot(df[column_name],'.',label=column_name) 
 
# for i in range(9):
#     x=911+i
#     plt.subplot(x)
#     plt.ylim((0,30)) 
#     column_name='participant'+str(i)
#     plt.plot(df[column_name],'.',label=column_name)   
    
     
# for i in range(6):
#     x=911+i
#     plt.subplot(x)
#     plt.ylim((0,30)) 
#     column_name='participant'+str(i+9)
#     plt.plot(df[column_name],'.',label=column_name)  
 
    

# mean_error_list=[]  
# mean_errorcal_list=[]    
# for i in range(15):
#     column_name='participant'+str(i)
#     mean_error_list.append(df[column_name].mean())
#     mean_errorcal_list.append(df_cal[column_name].mean())
    
    
# plt.plot(mean_error_list,'go',label='l2cs') 
# plt.plot(mean_errorcal_list,'ro',label='l2cs_calibration') 
# plt.legend(loc="upper right")
# plt.xlabel('participant')
# plt.ylabel('mean error of all dots')
# plt.title('gaze error difference among participants') 

# # plt.plot(x, y1, "-b", label="sine")
# # plt.plot(x, y2, "-r", label="cosine")
# # plt.legend(loc="upper left")
# # plt.ylim(-1.5, 2.0)
# # plt.show()


# file='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/error_l2cs/error_p12.csv'
# df=pd.read_csv(file)['0']

# a=[]
# for i in range(9):
#     x=df.iloc[i*150:(i+1)*150].mean()
#     a.append(x)
    
    
# def main(idealgazefile,pregazefile,new_pregazefile,errorgazefile):
    
#     idealgaze = pd.read_csv(idealgazefile)[['x','y','z']]
#     pregaze = pd.read_csv(pregazefile)[['yaw','pitch']]
    
    
#     pregaze=pregaze.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')#just for l2cs and eth
#     pregaze['yaw']=-pregaze['yaw'] #just for eth
#     pregaze['pitch']=-pregaze['pitch'] #just for eth
    
#     # pregaze['yaw']=pregaze['yaw']+0.122#just text
    
#     pregaze=pregaze.apply(sphere2cartesian,axis='columns')  
    
#     # pregaze.to_csv(pregazefile)
#     pregaze.to_csv(new_pregazefile) 
    
    
    
#     error=error3d(pregaze[['x','y','z']],idealgaze[['x','y','z']])
    
#     error.to_csv(errorgazefile)  



# if __name__ == '__main__':
#     import sys
#     main(sys.argv[1:])



participant_list=[]

for i in range(20):
    x=str(i+1)
    participant_list.append(x)


X = participant_list
Yl2cs = df.mean()
Zeth = df_cal.mean()

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Yl2cs, 0.4, label = 'before compensation')
plt.bar(X_axis + 0.2, Zeth, 0.4, label = 'after compensation')

plt.xticks(X_axis, X)
plt.xlabel("participant number")
plt.ylabel("gaze3d error(degree)")
plt.title("mean gaze3d errors of all participant")
plt.legend()
plt.show()





