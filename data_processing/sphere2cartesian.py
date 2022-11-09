#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:32:05 2022

@author: chenglinlin

what do this script do? get x,y,z from yaw and pitch
"""
import numpy as np
import pandas as pd

inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/pitch_yaw_res3.csv'
outputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test29(res_compare_l2cs)/output_l2cs/xyz_res3.csv'
gazel2cs = pd.read_csv(inputfile)


gazel2cs[3]=gazel2cs['yaw']
def sphere2cartesian(row):
    x1=row['yaw']
    x0=row['pitch']
    row[0]=np.cos(x1) * np.sin(x0)
    row[1]=np.sin(x1)  
    row[2]=-np.cos(x0) * np.cos(x1)
    return row

gazel2cs=gazel2cs.apply(sphere2cartesian,axis='columns')   

gazel2cs=gazel2cs.rename({'yaw': 0,'pitch': 1,3: 2}, axis='columns')
gazel2cs.to_csv(outputfile)