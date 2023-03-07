#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:35:41 2022
@author: chenglinlin

This python document is to get ideal 3d gaze direction(ground truth)
input: 2d sreen dot position(pixel unit),human eye postion .
parameters:necessary experiment setup information,like screen resolution,camera position etc.
output:3d gaze direction in huamn eye coordinate system


symbol explaination
{s}screen coordinate system
{c}robot'camera coordinate system
{e}human eye coordinate system
"""

import csv
import re
import pandas as pd
import numpy as np
import pickle
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math

######### design function to get 3d gaze direction in {e} from 2d screen dot

#get transforming matrix for {c} to {e}
def getLadybugToEyeMatrix(dirEyes):
    # Define left hand coordinate system in the eye plane orthogonal to the camera ray
    upVector = np.array([0,0,1], np.float32)
    zAxis = dirEyes.flatten()
    xAxis = np.cross(upVector, zAxis)
    xAxis /= np.linalg.norm(xAxis)
    yAxis = np.cross(zAxis, xAxis)
    yAxis /= np.linalg.norm(yAxis) # not really necessary
    gazeCS = np.stack([xAxis, yAxis, zAxis], axis=0)
    return gazeCS
# design function to get 3d gaze direction in {e} from dot position and eye position in {c}
def transform(A_p,pos,h_eye_cam):
    # get 9 different human position in {c}
    if pos == 1: 
        eye_pos=[-98,42-r_left,h_eye_cam]
    elif pos == 2:
        eye_pos=[-98,0-r_left,h_eye_cam]
    elif pos == 3:
        eye_pos=[-98,-42-r_left,h_eye_cam]
    elif pos == 4:
        eye_pos=[-198,42-r_left,h_eye_cam]
    elif pos == 5:
        eye_pos=[-198,0-r_left,h_eye_cam]
    elif pos == 6:
        eye_pos=[-198,-42-r_left,h_eye_cam]
    elif pos == 7:
        eye_pos=[-298,42-r_left,h_eye_cam]
    elif pos == 8:
        eye_pos=[-298,0-r_left,h_eye_cam]
    else:
        eye_pos=[-298,-42-r_left,h_eye_cam]
    
    
    eyes3D=np.array(eye_pos).reshape(1,3)#get eye posion
    gazeDirLB=A_p-eyes3D#get gaze direction,A_p is screen dot position in{c}
    gazeDirLB = gazeDirLB/np.linalg.norm(gazeDirLB,axis=1).reshape((m,1))# get normalized vector
    
    
    dirEyes = eyes3D/ np.linalg.norm(eyes3D)  #get normalized vector
    
    gazeCS = getLadybugToEyeMatrix(dirEyes)#get transforming matric
    gazeDir = np.matmul(gazeCS, gazeDirLB.T)#get 3d gaze diration in {e}

    return gazeDir

#########

path='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)'
filename=path+'/original_data/formal_experiment/P1_2022_11_14_17_35_52_78206780/2022_11_14_17_35_52_78206780_Header.csv'
output=path+'/processed_data/idealgaze_p1.csv'
h_eye_cam=155.5-117# 43(169) the distance in height between robot and human actual 40(170) 1
r_left=0#  The robot deviated 1 cm to the left
df=pd.read_csv(filename)
#df=df.dropna()

#get 3d dot position in {s} ,unit:pixel
x=np.array(df['x'])
y=np.array(df['y'])
z=np.zeros(x.shape)

# get 3d dot position in {c}, unit:cm
x1=-z+51 ###39
y1=-x*142/3840
z1=33.5+y*80/2160# 33.5


A=np.array([x1,y1,z1])
A=A.T

# get 3d gaze direction for 90 dots in which people change position for every 30 dots.

m=150#27,9
a=[1,2,3,4,5,6,7,8,9]
#a=[1,2,3]
for i in a:    
    A_p=A[m*(i-1):m*i,:]
    gazeDir_new=transform(A_p,i,h_eye_cam).T
    
    if i>1:
         gazeDir_old=np.concatenate((gazeDir_old,gazeDir_new))      
    else:
         gazeDir_old=gazeDir_new
 
# get dataframe        
dfn=pd.DataFrame(gazeDir_old)

#get yaw and pitch,then unify their units

def cartesian2sphere(row):
    row['yaw']=-math.atan(row[0]/row[2])*180/3.1415926
    row['pitch']=math.asin(row[1])*180/3.1415926    
    return row

dfn=dfn.apply(cartesian2sphere,axis='columns')   
dfn[[0,1,2]]=dfn[[0,1,2]]*180/3.1415926

#save



dfn=dfn.rename({0: 'x',1: 'y',2: 'z'}, axis='columns')
dfn.to_csv(output)  



def get_idealgaze(inputpath,height,outputpath)