#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 00:22:30 2022

@author: chenglinlin
"""

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




def transform(A_p,pos,h_eye_cam,d):
    
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
    
    
    eyes3D=np.array(eye_pos).reshape(1,3)
    gazeDirLB=A_p-eyes3D
    gazeDirLB = gazeDirLB/np.linalg.norm(gazeDirLB,axis=1).reshape((m,1))
    
    
    dirEyes = eyes3D/ np.linalg.norm(eyes3D)  
    
    gazeCS = getLadybugToEyeMatrix(dirEyes)
    gazeDir = np.matmul(gazeCS, gazeDirLB.T)
    
    
    return gazeDir



#filename='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p1_2022_11_14_17_35_52_78206780_Header.csv'
filename='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p12_2022_11_30_13_58_47_93080396_Header.csv'
#filename='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p21_2022_12_12_17_2_54_88988098_Header.csv'
h_eye_cam=175-117# 43(169) the distance in height between robot and human actual 40(170)
r_left=0#  The robot deviated 1 cm to the left
df=pd.read_csv(filename)
#df=df.dropna()
x=np.array(df['x'])*2
y=np.array(df['y'])*2
z=np.zeros(x.shape)

idealx=x*142/3840
idealy=y*80/2160



d_horizontal_robot_screen=62.5
d_vertical_robot_screen=38
d_horizontal_robot_position2=111

x1=-z+ d_horizontal_robot_screen ###39
y1=-x*142/3840
z1=d_vertical_robot_screen+y*80/2160# 33.5

A=np.array([x1,y1,z1])
A=A.T

m=150#27,9
a=[1,2,3,4,5,6,7,8,9]
#a=[1,2,3]
for i in a:    
    A_p=A[m*(i-1):m*i,:]
    gazeDir_new=transform(A_p,i,h_eye_cam,111).T
    
    if i>1:
         gazeDir_old=np.concatenate((gazeDir_old,gazeDir_new))      
    else:
         gazeDir_old=gazeDir_new
         

# filepath='/Users/chenglinlin/Documents/calibration/CalibrationGame_local-master/compare_test12(stu183)/'

# #
# with open(filepath+'idealgaze(d_e&r&dots_0).csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(gazeDir_old)
    
#   
dfideal=pd.DataFrame(gazeDir_old)   


#filepath2='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/p21/pitch_yaw_4k_p21.csv'
#filepath2='/Users/chenglinlin/Downloads/p21_baseline.csv'
filepath2='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_eth_baseline/p12/baseline/pitch_yaw.csv'
df = pd.read_csv(filepath2)
#df=df.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')
def sphere2cartesian(row):
    x0=row['yaw']
    x1=row['pitch']
    row['x']=np.cos(x1) * np.sin(x0)
    row['y']=np.sin(x1)  
    row['z']=-np.cos(x0) * np.cos(x1)
    return row
df=df.apply(sphere2cartesian,axis='columns')   
a=df[['x','y','z']].to_numpy()
b=dfideal.to_numpy()

e=[]
for i in range(1350):
    
    error = np.arccos(np.dot(a[i], b[i]))
    e.append(error)

dfe = pd.DataFrame(e)*180/3.1415926




#test virtual2d
filepath='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/virtual_2d_l2cs/virtual_2d_p21.csv'
virtual=pd.read_csv(filepath)[['virtual2d_x','virtual2d_y']]
plt.plot(virtual['virtual2d_x'],virtual['virtual2d_y'],'o')


