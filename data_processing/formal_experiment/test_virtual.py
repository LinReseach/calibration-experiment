#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:32:25 2022

@author: chenglinlin
"""

import csv
import re
import pandas as pd
import numpy as np
import pickle
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
filename='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/pregaze_p1.csv'
df=pd.read_csv(filename) 


##

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




def transform(g_p,pos,h_eye_cam,r_left):
          
    if pos == 1: 
        eye_pos=[-111,42-r_left,h_eye_cam]
    elif pos == 2:
        eye_pos=[-111,0-r_left,h_eye_cam]
    elif pos == 3:
        eye_pos=[-111,-42-r_left,h_eye_cam]
    elif pos == 4:
        eye_pos=[-211,42-r_left,h_eye_cam+1]
    elif pos == 5:
        eye_pos=[-211,0-r_left,h_eye_cam+1]
    elif pos == 6:
        eye_pos=[-211,-42-r_left,h_eye_cam+1]
    elif pos == 7:
        eye_pos=[-311,42-r_left,h_eye_cam+2]
    elif pos == 8:
        eye_pos=[-311,0-r_left,h_eye_cam+2]
    else:
    
        eye_pos=[-311,-42-r_left,h_eye_cam+2]
    
    
    eyes3D=np.array(eye_pos).reshape(1,3)
    # gazeDirLB=A_p-eyes3D
    # gazeDirLB = gazeDirLB/np.linalg.norm(gazeDirLB,axis=1).reshape((m,1))
    
    dirEyes = eyes3D/ np.linalg.norm(eyes3D)  
    
    gazeCS = getLadybugToEyeMatrix(dirEyes)
    gazeDirLB = np.matmul(inv(gazeCS), g_p.T)
    
    k=(63.5-eye_pos[0])/gazeDirLB[0]
    
    gaze_original=gazeDirLB.T.copy()
    
    
    for i in range(len(g_p)):
        gaze_original[i]=gazeDirLB.T[i]*k[i]
        
    target=gaze_original+eye_pos
    
    df_target = pd.DataFrame(target)
    df_target[0]=63.5-df_target[0]
    df_target[1]=-df_target[1]
    df_target[2]=df_target[2]-38
    
    
    return df_target


    


h_eye_cam=155-117
r_left=0
dfx = pd.DataFrame()
dfy = pd.DataFrame()
dft = pd.DataFrame()

for i in [1,2,3,4,5,6,7,8,9]:  # [1,2,3,4,5,6,7,8,9] 
    g_p=df.iloc[(i-1)*150:i*150]
    
    g_p=g_p[['x','y','z']].to_numpy()
    df_target=transform(g_p,i,h_eye_cam,r_left)
    x=df_target[1]
    y=df_target[2]
    dfx[i]=x
    dfy[i]=y
    

xlim=(-100,100)
ylim=(-80,80)

plt.subplot(331)
plt.plot(dfx[1].dropna(),dfy[1].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(332)
plt.plot(dfx[2].dropna(),dfy[2].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(333)
plt.plot(dfx[3].dropna(),dfy[3].dropna(),'g+',ix,iy,'rs')
plt.xlim(xlim)  
plt.ylim(ylim)  
plt.subplot(334)
plt.plot(dfx[4].dropna(),dfy[4].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(335)
plt.plot(dfx[5].dropna(),dfy[5].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(336)
plt.plot(dfx[6].dropna(),dfy[6].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(337)
plt.plot(dfx[7].dropna(),dfy[7].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim)   
plt.subplot(338)
plt.plot(dfx[8].dropna(),dfy[8].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(339)
plt.plot(dfx[9].dropna(),dfy[9].dropna(),'g+',ix,iy,'rs')  
plt.xlim(xlim)  
plt.ylim(ylim)  

##

xlim=(-120,120)
ylim=(-100,80)

plt.subplot(311)
plt.plot(dfx[1].dropna(),dfy[1].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(312)
plt.plot(dfx[2].dropna(),dfy[2].dropna(),'g+',ix,iy,'rs') 
plt.xlim(xlim)  
plt.ylim(ylim) 
plt.subplot(313)
plt.plot(dfx[3].dropna(),dfy[3].dropna(),'g+',ix,iy,'rs')
plt.xlim(xlim)  
plt.ylim(ylim) 

##
filename='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test26(lin_res2_robotpos)/pos2/'
/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test12(stu183)/video.mp4filepath=filename+'idealdocument.csv'
ideal = pd.read_csv(filepath, index_col=0)

t=ideal['seconds'].dropna()
t1=t-t.iloc[0]

for i in range(80):
    t1.iloc[i]=t.iloc[i+1]-t.iloc[i]

filepath2=filename+'idealtimeduration.csv'  

t1.to_csv(filepath2) 

t2=t1.copy()
t2=t1[1.7<t1]
t2=t2[1.88>t1]

filename='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_eth_baseline/pitch_yaw_p7new.csv'
df=pd.read_csv(filename) 

df.to_csv(filename)