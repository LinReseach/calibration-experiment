#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:08:52 2022

@author: chenglinlin

what do this script can do?: sometimes the output video have not only one person's gaze direction. To this problem there are two method
1.change code in gaze360 demo (colab),and only predict the gaze direction of that person with the biggest head box
2. not best method but not bad: when the output video only have a little other person's gaze, we can filter these by using following code
"""

def rid_grosserror(df):
    n=len(df.columns)

   
    if n==3:
         df_3=df.dropna()
         df_2=df[[0,1]].dropna().drop(df_3.index)
    if n==2:
         df_2=df.dropna()
         df_3=df[[0,1]].dropna().drop(df_2.index)# just make it none
         
    df_1=df.drop(df[[0,1]].dropna().index)[0]
    df_1_mean=df_1.mean()
    def compare2(row):
    
    # The row is a single Series object which is a single row indexed by column values
    # Let's extract the firstname and create a new entry in the series
        e1=row[0]-df_1_mean
        e2=row[1]-df_1_mean
        s1=pow(e1[0],2)+pow(e1[1],2)+pow(e1[2],2)
        s2=pow(e2[0],2)+pow(e2[1],2)+pow(e2[2],2)
        if s2<s1:
            row[0]=row[1]
        print([s1,s2])
        print(df_1_mean)
        # Now we just return the row and the pandas .apply() will take of merging them back into a DataFrame
        return row

    def compare3(row):
        
        # The row is a single Series object which is a single row indexed by column values
        # Let's extract the firstname and create a new entry in the series
        e1=row[0]-df_1_mean
        e2=row[1]-df_1_mean
        e3=row[2]-df_1_mean
        s1=pow(e1[0],2)+pow(e1[1],2)+pow(e1[2],2)
        s2=pow(e2[0],2)+pow(e2[1],2)+pow(e2[2],2)
        s3=pow(e3[0],2)+pow(e3[1],2)+pow(e3[2],2)
        
        s_min=min(s1,s2,s3)
        
        if s_min==s2:
            row[0]=row[1]
        if s_min==s3:
            row[0]=row[2]
       
        # Now we just return the row and the pandas .apply() will take of merging them back into a DataFrame
        return row
    df_2.apply(compare2,axis='columns')
    df_3.apply(compare3,axis='columns')
   
    
    df_new=pd.concat([df_1, df_2,df_3]).sort_index()[0]
    return df_new