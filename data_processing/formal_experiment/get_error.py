#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:21:09 2022

@author: chenglinlin
"""


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt



def sphere2cartesian(row):
    x0=row['yaw']
    x1=row['pitch']
    row['x']=np.cos(x1) * np.sin(x0)
    row['y']=np.sin(x1)  
    row['z']=-np.cos(x0) * np.cos(x1)
    return row

def error3d(a,b):

    a=a.to_numpy()
    b=b.to_numpy()
    e=[]
    for i in range(len(a)):
        
        error = np.arccos(np.dot(a[i], b[i]))
        e.append(error)
    
    dfe = pd.DataFrame(e)*180/3.1415926
    return dfe


#idealgazefile='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/idealgaze/idealgaze_p12.csv'
#pregazefile='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/p12/pitch_yaw_p12.csv'

def main(idealgazefile,pregazefile,new_pregazefile,errorgazefile,cal):
    
    idealgaze = pd.read_csv(idealgazefile)[['x','y','z']]
    pregaze = pd.read_csv(pregazefile)[['yaw','pitch']]
    
    
    
    if cal =='l2cs_no4k':
        pregaze=pregaze.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')#just for l2cs 
        #pregaze['pitch']=pregaze['pitch']+7.27/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['yaw']=pregaze['yaw']+2.19/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['pitch']=pregaze['pitch']+7.2/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['yaw']=pregaze['yaw']+2.1/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    elif cal =='l2cs_4k':
        pregaze=pregaze.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')#just for l2cs 
        #pregaze['pitch']=pregaze['pitch']+6.8/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['yaw']=pregaze['yaw']+0.92/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['pitch']=pregaze['pitch']+6.48/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['yaw']=pregaze['yaw']+0.95/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    elif cal =='eth_no4k':
        pregaze['pitch']=pregaze['pitch']+7/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['yaw']=pregaze['yaw']+2.57/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['pitch']=pregaze['pitch']+6.27/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['yaw']=pregaze['yaw']+3.99/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    elif cal =='eth_4k':
        pregaze['pitch']=pregaze['pitch']+6.03/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        pregaze['yaw']=pregaze['yaw']+0.83/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['pitch']=pregaze['pitch']+6.43/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        #pregaze['yaw']=pregaze['yaw']+1.11/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
        
    # pregaze=pregaze.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')#just for l2cs 
    # pregaze['pitch']=pregaze['pitch']+7.27/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    # pregaze['yaw']=pregaze['yaw']+2.19/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    # pregaze=pregaze.rename({'yaw': 'pitch','pitch': 'yaw'}, axis='columns')#just for l2cs 
 
    
    #
    # pregaze['pitch']=pregaze['pitch']+7/180*3.14#just test 0.122(7 degree) 0,105(6 degree)
    pregaze=pregaze.apply(sphere2cartesian,axis='columns')  
    
    # pregaze.to_csv(pregazefile)
    pregaze.to_csv(new_pregazefile) 
    
    
    
    error=error3d(pregaze[['x','y','z']],idealgaze[['x','y','z']])
    
    error.to_csv(errorgazefile)  



if __name__ == '__main__':
    import sys
    main(sys.argv[1:])







