#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:41:45 2022
@author: chenglinlin

what can this script do? pick up one gaze every threw gazes
"""

inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test18(lin_1perdot_alwayspicnoarrow)/gaze360/gaze.csv'
outputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test18(lin_1perdot_alwayspicnoarrow)/gaze360/gaze_pickup.csv'
df = pd.read_csv(inputfile, index_col=0)

df=df.iloc[0:243]
b=[]
for i in range(81):
    b.append(3*i)
    b.append(3*i+1)
        
df1=df.drop(df.index[b])
# b=[]
# for i in range(6):
#     b.append(10*i)
# df2=df1.drop(df1.index[b])
df1.to_csv(outputfile)   