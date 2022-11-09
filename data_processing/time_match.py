#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:50:15 2022

@author: chenglinlin
"""

frametimefile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test24(lin_res3_timetable_1perdot)/frametime.csv'
gazefile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3.csv'
outputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withtime.csv'

dftime= pd.read_csv(frametimefile,index_col=0)
dfgaze = pd.read_csv(gazefile,index_col=0)

#dftime=dftime.loc[107:1216]
dftime=dftime.loc[43:684]
dftime.reset_index(drop=True, inplace=True)
    

dfgaze['minute']=dftime['0']
dfgaze['second']=dftime['1']

dfgaze.to_csv(outputfile) 




# filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test24(lin_res3_timetable_1perdot)/idealgazewithtime.csv'
# df1 = pd.read_csv(filepath, index_col=0)

# df1.index=df1.index*2
# df2=df1.copy()## do not use df2=df1
# df2['seconds']=df2['seconds']+1

# df2.index=df2.index+1
# df3=df1.append(df2)
# df4=df3.sort_index()
# filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test24(lin_res3_timetable_1perdot)/idealgazewithtimecopy.csv'
# df4.to_csv(filepath)  

##detail time
filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withtime.csv'
df = pd.read_csv(filepath,index_col=0)


df['time']=df['minute']*60+df['second']
dfc2 = pd.DataFrame()
for i in df['time'].unique():
    
    dfc=df[df['time']==i]
    len=dfc.shape[0]
    a=[]
    for j in range(len):
        a.append(i+j*1/len)
    dfc['realtime']=a
    dfc2=dfc2.append(dfc)
        
filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withrealtime.csv'
dfc2.to_csv(filepath)    
   
#### gaze360 average only screen dot appear

filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test24(lin_res3_timetable_1perdot)/idealgazewithtimecopy.csv'
dfi = pd.read_csv(filepath, index_col=0)
dfi['time']=dfi['minute']*60+dfi['seconds']

filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withtime.csv'
df = pd.read_csv(filepath, index_col=0)

df=df.rename({'0': '3','1': '4','2': '5'}, axis='columns')
dfi=dfi.rename({'seconds': 'second'}, axis='columns')

df.set_index(['minute','second'], inplace=True)
dfi.set_index(['minute','second'], inplace=True)

df3=pd.merge(dfi, df, how='inner', left_index=True, right_index=True)

df4=df3.groupby(['minute','second']).mean()
filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3_withidealtotal.csv'
df3.to_csv(filepath) 