#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:47:50 2022

@author: chenglinlin
"""

import os
import re
import pandas as pd

##################################################get groundtruth
#get human eye height
height_humaneye=[155.5,160,148,163.5,161,168,140,178,160,159.5,175,174.5,167,166,158,170,162,151,150,157,177]

# get inputfile path
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/'
participant_list=[]

for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
# regex = re.compile('(.*zip$)|(.*rar$)|(.*r01$)')
# for root, dirs, files in os.walk(rootdir):
#   for file in files:
#     if regex.match(file):
#        print(file)
# regex = re.compile('(.*p1.csv$)')
# for root, dirs, files in os.walk(rootdir):
#   for file in files:
#     if regex.match(file):
#        print(file)     

inputfile_list=[]
for x in participant_list:
    regex = re.compile('((%s)_.*.csv$)'%x)
    for root, dirs, files in os.walk(rootdir):
      for file in files:
        if regex.match(file):
            path=rootdir+file
            inputfile_list.append(path)

# get outputfile path
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'
outputpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/idealgaze_p'+str(i+1)+'.csv'
    outputpath_list.append(x)


          
import Get_Ideal3dGaze_15dots_main as getideal

for i in range(len(inputfile_list)):
    inputfile=inputfile_list[i]
    outputfile=outputpath_list[i]
    height=height_humaneye[i]
    getideal.main(inputfile,height,outputfile,117)
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]
height_humaneye4k=[height_humaneye[i] for i in (1,3,11,12,13,14,17,18,19,20)]

inputfile4k_list=[]
for x in participant4k_list:
    regex = re.compile('((%s)_.*.csv$)'%x)
    for root, dirs, files in os.walk(rootdir):
      for file in files:
        if regex.match(file):
            path=rootdir+file
            inputfile4k_list.append(path)
            
outputpath4k_list=[]
for x in participant4k_list:
    y=outputroot+'processed_data/idealgaze_4k_'+x+'.csv'
    outputpath4k_list.append(y)
    
for i in range(len(outputpath4k_list)):
    inputfile=inputfile4k_list[i]
    outputfile=outputpath4k_list[i]
    height=height_humaneye4k[i]
    getideal.main(inputfile,height,outputfile,124)
    
#################################################get error
    
# get idealgaze path
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'
idealgazepath_list=[]
for i in range(21):
    x=outputroot+'processed_data/idealgaze/idealgaze_p'+str(i+1)+'.csv'
    idealgazepath_list.append(x)    
    
# get pregaze file

participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/'

pregazepath_list=[]
for x in participant_list:
     path=rootdir+x+'/pitch_yaw_'+x+'.csv'
     pregazepath_list.append(path) 
            
new_pregazepath_list=[]
for x in participant_list:
     path=rootdir+'pregaze_cal'+x+'.csv'
     new_pregazepath_list.append(path)  
       
 
         
errorpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_l2cs_cal/error_p'+str(i+1)+'.csv'
    errorpath_list.append(x)
#   
import get_error_easier as geterror
for i in range(21):
   
    idealgaze=idealgazepath_list[i]
    pregaze=pregazepath_list[i]
    error=errorpath_list[i]
    new_pregazefile=new_pregazepath_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error)
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

idealgazepath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/idealgaze/idealgaze_4k_'+x+'.csv'
    idealgazepath4k_list.append(x)  
    
new_pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'pregaze4k_'+x+'.csv'
     new_pregazepath4k_list.append(path)  

pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+x+'/pitch_yaw_4k_'+x+'.csv'
     pregazepath4k_list.append(path) 


errorpath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/error_l2cs/error_4k_'+x+'.csv'
    errorpath4k_list.append(x)
#   

for i in range(10):
   
    idealgaze=idealgazepath4k_list[i]
    pregaze=pregazepath4k_list[i]
    error=errorpath4k_list[i]
    new_pregazefile=new_pregazepath4k_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error)
    
    
############################################get virtual2d


participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/'

            
pregazepath_list=[]
for x in participant_list:
     path=rootdir+'predicted_gaze_l2cs/pregaze_'+x+'.csv'
     pregazepath_list.append(path)  
       
 
         
virtual2dpath_list=[]
for x in participant_list:
    y=rootdir+'virtual_2d_l2cs/'+'virtual_2d_'+x+'.csv'
    virtual2dpath_list.append(y)
#   
import get_virtual2d
for i in range(21):
  get_virtual2d.main(height_humaneye[i],pregazepath_list[i],virtual2dpath_list[i],117)
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

height_humaneye4k=[height_humaneye[i] for i in (1,3,11,12,13,14,17,18,19,20)]
            
pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'predicted_gaze_l2cs/pregaze4k_'+x+'.csv'
     pregazepath4k_list.append(path)  
       
 
         
virtual2dpath4k_list=[]
for x in participant4k_list:
    y=rootdir+'virtual_2d_l2cs/'+'virtual_2d_4k_'+x+'.csv'
    virtual2dpath4k_list.append(y)
#   
import get_virtual2d
for i in range(10):
  get_virtual2d.main(height_humaneye4k[i],pregazepath4k_list[i],virtual2dpath4k_list[i],124)
#   

##########################
#################################################get error for eth baseline
    
# get idealgaze path
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'
idealgazepath_list=[]
for i in range(21):
    x=outputroot+'processed_data/idealgaze/idealgaze_p'+str(i+1)+'.csv'
    idealgazepath_list.append(x)    
    
# get pregaze file

participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_eth_baseline/'

pregazepath_list=[]
for x in participant_list:
     path=rootdir+'pitch_yaw_'+x+'.csv'
     pregazepath_list.append(path) 
            
new_pregazepath_list=[]
for x in participant_list:
     path=rootdir+'pregaze_eth_'+x+'.csv'
     new_pregazepath_list.append(path)  
       
 
         
errorpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_eth_baseline/error_p'+str(i+1)+'.csv'
    errorpath_list.append(x)
#   
import get_error_easier as geterror
for i in range(21):
   
    idealgaze=idealgazepath_list[i]
    pregaze=pregazepath_list[i]
    error=errorpath_list[i]
    new_pregazefile=new_pregazepath_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error)
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

idealgazepath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/idealgaze/idealgaze_4k_'+x+'.csv'
    idealgazepath4k_list.append(x)  
    
new_pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'pregaze4k_'+x+'.csv'
     new_pregazepath4k_list.append(path)  


pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'pitch_yaw_4k_'+x+'.csv'
     pregazepath4k_list.append(path) 


errorpath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/error_eth_baseline/error_4k_'+x+'.csv'
    errorpath4k_list.append(x)
#   

for i in range(10):
   
    idealgaze=idealgazepath4k_list[i]
    pregaze=pregazepath4k_list[i]
    error=errorpath4k_list[i]
    new_pregazefile=new_pregazepath4k_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error)
    
    
############################################get virtual2d for eth baseline
height_humaneye=[155.5,160,148,163.5,161,168,140,178,160,159.5,175,174.5,167,166,158,170,162,151,150,157,177]
participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/'

            
pregazepath_list=[]
for x in participant_list:
     path=rootdir+'predicted_gaze_eth_baseline/pregaze_eth_'+x+'.csv'
     pregazepath_list.append(path)  
       
 
         
virtual2dpath_list=[]
for x in participant_list:
    y=rootdir+'virtual_2d_eth_baseline/'+'virtual_2d_'+x+'.csv'
    virtual2dpath_list.append(y)
#   
import get_virtual2d
for i in range(21):
  get_virtual2d.main(height_humaneye[i],pregazepath_list[i],virtual2dpath_list[i],117)
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

height_humaneye4k=[height_humaneye[i] for i in (1,3,11,12,13,14,17,18,19,20)]
            
pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'predicted_gaze_eth_baseline/pregaze4k_'+x+'.csv'
     pregazepath4k_list.append(path)  
       
 
         
virtual2dpath4k_list=[]
for x in participant4k_list:
    y=rootdir+'virtual_2d_eth_baseline/'+'virtual_2d_4k_'+x+'.csv'
    virtual2dpath4k_list.append(y)
#   
import get_virtual2d
for i in range(10):
  get_virtual2d.main(height_humaneye4k[i],pregazepath4k_list[i],virtual2dpath4k_list[i],124)
############################################get virtual2d for l2cs after calibration
height_humaneye=[155.5,160,148,163.5,161,168,140,178,160,159.5,175,174.5,167,166,158,170,162,151,150,157,177]
participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/'

            
pregazepath_list=[]
for x in participant_list:
     path=rootdir+'predicted_gaze_l2cs/pregaze_cal'+x+'.csv'
     pregazepath_list.append(path)  
       
 
         
virtual2dpath_list=[]
for x in participant_list:
    y=rootdir+'virtual_2d_l2cs_cal/'+'virtual_2d_'+x+'.csv'
    virtual2dpath_list.append(y)
#   
import get_virtual2d
for i in range(21):
  get_virtual2d.main(height_humaneye[i],pregazepath_list[i],virtual2dpath_list[i],117)
  

#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

height_humaneye4k=[height_humaneye[i] for i in (1,3,11,12,13,14,17,18,19,20)]
            
pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'predicted_gaze_l2cs/pregaze4k_cal'+x+'.csv'
     pregazepath4k_list.append(path)  
       
 
         
virtual2dpath4k_list=[]
for x in participant4k_list:
    y=rootdir+'virtual_2d_l2cs_cal/'+'virtual_2d_4k_'+x+'.csv'
    virtual2dpath4k_list.append(y)
#   
import get_virtual2d
for i in range(10):
  get_virtual2d.main(height_humaneye4k[i],pregazepath4k_list[i],virtual2dpath4k_list[i],124)
#################################################get error for l2cs_cal
    
# get idealgaze path
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'
idealgazepath_list=[]
for i in range(21):
    x=outputroot+'processed_data/idealgaze/idealgaze_p'+str(i+1)+'.csv'
    idealgazepath_list.append(x)    
    
# get pregaze file

participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_l2cs/'

pregazepath_list=[]
for x in participant_list:
     path=rootdir+x+'/pitch_yaw_'+x+'.csv'
     pregazepath_list.append(path) 
            
new_pregazepath_list=[]
for x in participant_list:
     path=rootdir+'pregaze_cal'+x+'.csv'
     new_pregazepath_list.append(path)  
       
 
         
errorpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_l2cs_cal/error_p'+str(i+1)+'.csv'
    errorpath_list.append(x)
#   
import get_error_easier as geterror
for i in range(21):
   
    idealgaze=idealgazepath_list[i]
    pregaze=pregazepath_list[i]
    error=errorpath_list[i]
    new_pregazefile=new_pregazepath_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error,'l2cs_no4k')
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

idealgazepath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/idealgaze/idealgaze_4k_'+x+'.csv'
    idealgazepath4k_list.append(x)  
    
new_pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'pregaze4k_cal'+x+'.csv'
     new_pregazepath4k_list.append(path)  

pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+x+'/pitch_yaw_4k_'+x+'.csv'
     pregazepath4k_list.append(path) 


errorpath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/error_l2cs_cal/error_4k_'+x+'.csv'
    errorpath4k_list.append(x)
#   
import get_error_easier as geterror
for i in range(10):
   
    idealgaze=idealgazepath4k_list[i]
    pregaze=pregazepath4k_list[i]
    error=errorpath4k_list[i]
    new_pregazefile=new_pregazepath4k_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error,'l2cs_4k')
    
#################################################get error for eth_cal
    
# get idealgaze path
outputroot='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/'
idealgazepath_list=[]
for i in range(21):
    x=outputroot+'processed_data/idealgaze/idealgaze_p'+str(i+1)+'.csv'
    idealgazepath_list.append(x)    
    
# get pregaze file

participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/predicted_gaze_eth_baseline/'

pregazepath_list=[]
for x in participant_list:
     path=rootdir+'/pitch_yaw_'+x+'.csv'
     pregazepath_list.append(path) 
            
new_pregazepath_list=[]
for x in participant_list:
     path=rootdir+'pregaze_cal'+x+'.csv'
     new_pregazepath_list.append(path)  
       
 
         
errorpath_list=[]
for i in range(21):
    x=outputroot+'processed_data/error_eth_cal/error_p'+str(i+1)+'.csv'
    errorpath_list.append(x)
#   
import get_error_easier as geterror
for i in range(21):
   
    idealgaze=idealgazepath_list[i]
    pregaze=pregazepath_list[i]
    error=errorpath_list[i]
    new_pregazefile=new_pregazepath_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error,'eth_no4k')
    
#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

idealgazepath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/idealgaze/idealgaze_4k_'+x+'.csv'
    idealgazepath4k_list.append(x)  
    
new_pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'pregaze4k_cal'+x+'.csv'
     new_pregazepath4k_list.append(path)  

pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'/pitch_yaw_4k_'+x+'.csv'
     pregazepath4k_list.append(path) 


errorpath4k_list=[]
for x in participant4k_list:
    x=outputroot+'processed_data/error_eth_cal/error_4k_'+x+'.csv'
    errorpath4k_list.append(x)
#   
import get_error_easier as geterror
for i in range(10):
   
    idealgaze=idealgazepath4k_list[i]
    pregaze=pregazepath4k_list[i]
    error=errorpath4k_list[i]
    new_pregazefile=new_pregazepath4k_list[i]
    #print(idealgaze,pregaze,error)
    geterror.main(idealgaze,pregaze,new_pregazefile,error,'eth_4k')
############################################get virtual2d for eth after calibration
height_humaneye=[155.5,160,148,163.5,161,168,140,178,160,159.5,175,174.5,167,166,158,170,162,151,150,157,177]
participant_list=[]
for i in range(21):
    x='p'+str(i+1)
    participant_list.append(x)
    
rootdir = '/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/'

            
pregazepath_list=[]
for x in participant_list:
     path=rootdir+'predicted_gaze_eth_baseline/pregaze_cal'+x+'.csv'
     pregazepath_list.append(path)  
       

         
virtual2dpath_list=[]
for x in participant_list:
    y=rootdir+'virtual_2d_eth_cal/'+'virtual_2d_'+x+'.csv'
    virtual2dpath_list.append(y)
#   
import get_virtual2d
for i in range(21):
  get_virtual2d.main(height_humaneye[i],pregazepath_list[i],virtual2dpath_list[i],117)
  

#4k
participant4k_list=[participant_list[i] for i in (1,3,11,12,13,14,17,18,19,20)]

height_humaneye4k=[height_humaneye[i] for i in (1,3,11,12,13,14,17,18,19,20)]
            
pregazepath4k_list=[]
for x in participant4k_list:
     path=rootdir+'predicted_gaze_eth_baseline/pregaze4k_cal'+x+'.csv'
     pregazepath4k_list.append(path)  
       
 
         
virtual2dpath4k_list=[]
for x in participant4k_list:
    y=rootdir+'virtual_2d_eth_cal/'+'virtual_2d_4k_'+x+'.csv'
    virtual2dpath4k_list.append(y)
#   
import get_virtual2d
for i in range(10):
  get_virtual2d.main(height_humaneye4k[i],pregazepath4k_list[i],virtual2dpath4k_list[i],124)   