#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:12:41 2022

@author: chenglinlin
"""
import csv
import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# frame to video
inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test26(lin_res2_robotpos)/pos2(headstill)/frame/'+'/'+'*.jpg' 
outputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test26(lin_res2_robotpos)/pos2(headstill)/video.mp4'
import ffmpeg

(
    ffmpeg
    .input(inputfile, pattern_type='glob', framerate=5)
    .output(outputfile)
    .run()
)


# data preprocessing, the output from gaze360 demo is pkl,we need convert pkl to csv

inputfile='/Volumes/CLL_USB1/experiment data/gaze.pkl'
outputfile='/Volumes/CLL_USB1/experiment data/gaze.csv'

with open(inputfile, 'rb') as f:
    a = pickle.load(f)
df = pd.DataFrame(a)  

def splitrow(row):
    s=row[0]
    row[0]=s[0]
    row[1]=s[1]
    row[2]=s[2]
    return row

  
df=df.apply(splitrow,axis='columns')   

df.to_csv(outputfile)  


