#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import Imputer
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
plt.style.use('bmh')

#Survey Data

df = pd.read_excel("airport_choice_survey_EN_ver2.0_Capstone.xlsx")
#print(df.describe())
#Reference Data 
ref = pd.ExcelFile("Airport_Airline_data_Supplementary_Reference.xlsx")
traffic_df = pd.read_excel(ref, 'Traffic Info')
price_df = pd.read_excel(ref, 'Price Info')
dist_df = pd.read_excel(ref, 'Airport Province Distance')


# In[2]:


#Check missing data
na = df.isnull().sum()
print(na)

#Airfare has 155 missing values in our raw data 


# In[4]:


#Drop NaN 
df = df.dropna(subset = ['Destination', 'Airport', 'Airline']) #488 to 47
#Had to drop NA columns from these variables to fill in avg prices 

#Destination variable - change float to int (used for later on - filling NaN values)
df['Destination'] = df['Destination'].astype(int) 
df['Airport'] = df['Airport'].astype(int) 
df['Airline'] = df['Airline'].astype(int) 


# In[5]:


#Check missing values after dropping NA from Destination, Airport, Airline
na = df.isnull().sum()
print(na)

#Airfare has 149 missing values after dropping NA values from above columns 


# In[7]:


#Fill in Airfare's missing values with Average 

#Average Prices - GMP 
gmp_oz_CN= price_df.iloc[0][3] #GMP, destination: China
gmp_kor_CN = price_df.iloc[2][3] #GMP, destination: China 
gmp_oz_JP = price_df.iloc[4][3] #GMP, destination: Japan
gmp_kor_JP = price_df.iloc[6][3] #GMP, destination: Japan
gmp_korLLC_JP = price_df.iloc[7][3]#GMP, destination: Japan 

##Average Prices - ICN
icn_oz_CN= price_df.iloc[12][3] #ICN, destination: China
icn_kor_CN = price_df.iloc[14][3]
icn_oz_JP = price_df.iloc[16][3]
icn_kor_JP = price_df.iloc[18][3]
icn_korLLC_JP = price_df.iloc[19][3]

#Average Price Dictionary 
avgP = {'G1':gmp_oz_CN, 'G2': gmp_kor_CN, 'G3':gmp_oz_JP , 'G4':gmp_kor_JP, 'G5': gmp_korLLC_JP, 'I1':icn_oz_CN, 'I2':icn_kor_CN, 'I3':icn_oz_JP, 'I4':icn_kor_JP, 'I5':icn_korLLC_JP }

#original df - ['Airfare'].isnull().sum() #155  Total of 327 Observations #Airport Choice: 1. Inchoen (ICN)   2. Gimpo (GMP)
#Flight destinations: 1. China 2.Japan 3.Southeast Asia 4.Other #Airline Choice: 1. Korean Air(KE) 2. Asiana Airlines (OZ) 3. Korean LCC 4. Foreign Airlines"""

#Fill in NaN values 
#Ex) df['Airfare'].isnull().sum() - After adding G1 values, null values decreased from 155 to 142 & Total obs: 327 to 334
df.loc[(df['Airport'] == 2) & (df['Destination']== 1) & (df['Airline'] ==2) & (df['Airfare'].isnull()), 'Airfare']=avgP['G1']/10000
df.loc[(df['Airport'] == 2) & (df['Destination']== 1) & (df['Airline'] ==1) & (df['Airfare'].isnull()), 'Airfare']=avgP['G2']/10000
df.loc[(df['Airport'] == 2) & (df['Destination']== 2) & (df['Airline'] ==2) & (df['Airfare'].isnull()), 'Airfare']=avgP['G3']/10000
df.loc[(df['Airport'] == 2) & (df['Destination']== 2) & (df['Airline'] ==1) & (df['Airfare'].isnull()), 'Airfare']=avgP['G4']/10000
df.loc[(df['Airport'] == 2) & (df['Destination']== 2) & (df['Airline'] ==3) & (df['Airfare'].isnull()), 'Airfare']=avgP['G5']/10000

df.loc[(df['Airport'] == 1) & (df['Destination']== 1) & (df['Airline'] ==2) & (df['Airfare'].isnull()), 'Airfare']=avgP['I1']/10000
df.loc[(df['Airport'] == 1) & (df['Destination']== 1) & (df['Airline'] ==1) & (df['Airfare'].isnull()), 'Airfare']=avgP['I2']/10000
df.loc[(df['Airport'] == 1) & (df['Destination']== 2) & (df['Airline'] ==2) & (df['Airfare'].isnull()), 'Airfare']=avgP['I3']/10000
df.loc[(df['Airport'] == 1) & (df['Destination']== 2) & (df['Airline'] ==1) & (df['Airfare'].isnull()), 'Airfare']=avgP['I4']/10000
df.loc[(df['Airport'] == 1) & (df['Destination']== 2) & (df['Airline'] ==3) & (df['Airfare'].isnull()), 'Airfare']=avgP['I5']/10000

#Airfare NaN values: decreased from 155 to 81 after filling in with avgPrice 

#Check added values with exported excel (new file)
'''
df.to_excel(r'C:\Users\hamyunjung\Desktop\Airline_avgAirfare\Airport_choice.xlsx')
'''
#Check data again 
na = df.isnull().sum()
print(na)


# In[44]:




