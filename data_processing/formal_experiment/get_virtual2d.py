#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:33:49 2022

@author: chenglinlin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:51:23 2022

@author: chenglinlin
"""

import csv
import re
import pandas as pd
import numpy as np
import pickle
from numpy.linalg import inv
import matplotlib.pyplot as plt




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


    


  

    
##


# plt.plot(dfn['virtual2d_x'].iloc[150:150*2],dfn['virtual2d_y'].iloc[150:150*2].dropna(),'g+',ix,iy,'rs') 
    
def main(height,pregazefile,outputfile,camera_height):
    
#idealgazepath='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p12_2022_11_30_13_58_47_93080396_Header.csv'
   # pregazefile='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/pregaze_p1.csv'
    df= pd.read_csv(pregazefile)[['x','y','z']]
    
    h_eye_cam=height-camera_height
    r_left=0
    
    
    for i in [1,2,3,4,5,6,7,8,9]:  # [1,2,3,4,5,6,7,8,9] 
        g_p=df.iloc[(i-1)*150:i*150] 
        g_p=g_p[['x','y','z']].to_numpy()
        gazeDir_new=transform(g_p,i,h_eye_cam,r_left)
        
        if i>1:
             gazeDir_old=np.concatenate((gazeDir_old,gazeDir_new))      
        else:
             gazeDir_old=gazeDir_new
     
    # get dataframe        
        dfn=pd.DataFrame(gazeDir_old)[[1,2]]    
        
    dfn=dfn.rename({1: 'virtual2d_x',2: 'virtual2d_y'}, axis='columns')
    dfn.to_csv(outputfile) 
   



if __name__ == '__main__':
    import sys
    main(sys.argv[1:])


