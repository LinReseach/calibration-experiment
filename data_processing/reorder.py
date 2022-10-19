#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:57:00 2022

@author: chenglinlin

what do this script can do : to reorder frames.for example,there are  81 pictures 0,1,2,3...80.
I will pick one picture every 9 pictures,and the picked pictures can become a new group,in the end,there are 9 groups.
0,9,18,27,36,45,54,63,72,1,10,19,28,37........80

"""
import csv
import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
###reorder  pictures

inputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test17(lin_1perdot_pos1-9)/frame/'
ffCount=1
data = [] 
for n in range(81): 
    
    fName = inputfile+'/'+'%05d.jpg' % (ffCount+1)
    ffCount+=1
    img=cv2.imread(fName)
    data.append(img)

a=[]
for i in range(81):
    j=int(i/9)
    mod=i-9*j
    ai=data[j+mod*9]
    print(j+mod*9)
    a.append(ai)

#outputfile='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test17(lin_1perdot_pos1-9)/frame_reorder2/'
for n in range(81): 
    # f=outputfile+str(3)+'.jpg'
    # cv2.imwrite(f,images[0])
    cv2.imwrite(f'/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test17(lin_1perdot_pos1-9)/frame_reorder2/{n:05}.jpg',a[n])

