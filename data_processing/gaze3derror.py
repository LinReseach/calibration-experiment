#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:10:46 2022

@author: chenglinlin
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res2_withidealtotal.csv'
idealgaze = pd.read_csv(inputfile)[['0','1','2']]
pregaze = pd.read_csv(inputfile)[['3','4','5']]
time = pd.read_csv(inputfile)[['time']]-pd.read_csv(inputfile)[['time']].iloc[0]
time_res2=time.copy()

def error3d(a,b):
    a=a.to_numpy()
    b=b.to_numpy()
    e=[]
    for i in range(len(a)):
        
        error = np.arccos(np.dot(a[i], b[i]))
        e.append(error)
    
    dfe = pd.DataFrame(e)*180/3.1415926
    return dfe

errorl2cs=error3d(pregaze,idealgaze)
error_res2=errorl2cs.copy()

inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withidealtotal.csv'
idealgaze = pd.read_csv(inputfile)[['0','1','2']]
pregaze = pd.read_csv(inputfile)[['3','4','5']]
time = pd.read_csv(inputfile)[['time']]-pd.read_csv(inputfile)[['time']].iloc[0]
time_res3=time.copy()

errorl2cs=error3d(pregaze,idealgaze)
error_res3=errorl2cs.copy()



plt.plot(error_res2,'go',label='res640*480') 
plt.plot(error_res3,'ro',label='res1280*960') 
plt.legend(loc="upper left")
plt.xlabel('frame')
plt.ylabel('degree')
plt.title('pos1-9 continue picture.mean error: res2:9.4. res3:10.8')

error_res2.mean()
error_res3.mean()


plt.plot(time_res2,error_res2,'go') 
plt.plot(time_res3,error_res3,'ro') 

df=pd.read_csv(inputfile)
dfc2 = pd.DataFrame()
for i in df['time'].unique():
    
    dfc=df[df['time']==i]
    len=dfc.shape[0]
    a=[]
    for j in range(len):
        a.append(i+j*1/len)
    dfc['realtime']=a
    dfc2=dfc2.append(dfc)
    
time = dfc2[['realtime']]-dfc2[['realtime']].iloc[0]
time_res2=time.copy()

plt.plot(time_res2,'go') 



import seaborn.objects as so
(
    so.Plot(penguins, x="bill_length_mm", y="bill_depth_mm")
    .add(so.Dot())
)



