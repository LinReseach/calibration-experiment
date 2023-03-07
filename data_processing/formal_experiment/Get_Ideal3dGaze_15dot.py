#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:14:36 2022

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
def transform(A_p,pos,h_eye_cam,m,d):
    # get 9 different human position in {c}
    r_left=0
    if pos == 1: 
        eye_pos=[-d,42-r_left,h_eye_cam]
    elif pos == 2:
        eye_pos=[-d,0-r_left,h_eye_cam]
    elif pos == 3:
        eye_pos=[-d,-42-r_left,h_eye_cam]
    elif pos == 4:
        eye_pos=[-d-100,42-r_left,h_eye_cam+1]
    elif pos == 5:
        eye_pos=[-d-100,0-r_left,h_eye_cam+1]
    elif pos == 6:
        eye_pos=[-d-100,-42-r_left,h_eye_cam+1]
    elif pos == 7:
        eye_pos=[-d-200,42-r_left,h_eye_cam+2]
    elif pos == 8:
        eye_pos=[-d-200,0-r_left,h_eye_cam+2]
    else:
        eye_pos=[-d-200,-42-r_left,h_eye_cam+2]
    
    
    eyes3D=np.array(eye_pos).reshape(1,3)#get eye posion
    gazeDirLB=A_p-eyes3D#get gaze direction,A_p is screen dot position in{c}
    gazeDirLB = gazeDirLB/np.linalg.norm(gazeDirLB,axis=1).reshape((m,1))# get normalized vector
    
    
    dirEyes = eyes3D/ np.linalg.norm(eyes3D)  #get normalized vector
    
    gazeCS = getLadybugToEyeMatrix(dirEyes)#get transforming matric
    gazeDir = np.matmul(gazeCS, gazeDirLB.T)#get 3d gaze diration in {e}

    return gazeDir
def cartesian2sphere(row):
    row['yaw']=-math.atan(row[0]/row[2])
    row['pitch']=math.asin(row[1])  
    return row

#########
# inputpath='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p1_2022_11_14_17_35_52_78206780_Header.csv'
# height=155.5


def main(inputpath,height,outputpath,camera_height):

    #camera_height=117,124(4k)
    h_eye_cam=height-camera_height# 43(169) the distance in height between robot and human actual 40(170) 1
    r_left=0#  The robot deviated 1 cm to the left
    df=pd.read_csv(inputpath)
    #df=df.dropna()
    
    #get 3d dot position in {s} ,unit:pixel
    x=np.array(df['x'])*2
    y=np.array(df['y'])*2
    z=np.zeros(x.shape)
    
    # get 3d dot position in {c}, unit:cm
    
    d_horizontal_robot_screen=62.5
    d_vertical_robot_screen=38
    d_horizontal_robot_position2=111
    
    x1=-z+ d_horizontal_robot_screen ###39
    y1=-x*142/3840
    z1=d_vertical_robot_screen+y*80/2160# 33.5
    
    
    A=np.array([x1,y1,z1])
    A=A.T
    
    # get 3d gaze direction for 90 dots in which people change position for every 30 dots.
    
    m=150#27,9
    a=[1,2,3,4,5,6,7,8,9]
    #a=[1,2,3]
    for i in a:    
        A_p=A[m*(i-1):m*i,:]
        gazeDir_new=transform(A_p,i,h_eye_cam,m,d_horizontal_robot_position2).T
        
        if i>1:
             gazeDir_old=np.concatenate((gazeDir_old,gazeDir_new))      
        else:
             gazeDir_old=gazeDir_new
     
    # get dataframe        
    dfn=pd.DataFrame(gazeDir_old)
    
    #get yaw and pitch,then unify their units
    

    #it's wrong,the unit of x,y,z can not be degree.
    dfn=dfn.apply(cartesian2sphere,axis='columns')   
   
    
    #save
    
    
    
    dfn=dfn.rename({0: 'x',1: 'y',2: 'z'}, axis='columns')
    dfn.to_csv(outputpath)  



if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
