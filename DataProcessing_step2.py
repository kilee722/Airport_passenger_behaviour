# Anh Nguyen
# Final Project
# 3/5/2020

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import Imputer #package to fill NA

plt.style.use('bmh')

df = pd.read_excel('Airport_choice.xlsx')

df = pd.DataFrame(df)

#print(df.info())

'''
print("NaN porportion")
print(((len(df)-df.count())/len(df))*100)
'''

# Droping SeatClass NA

final_df = df.dropna(subset = ['SeatClass'])
final_df = final_df.fillna(0)

# Checking distribution of NA income in each occupation group
occupation_income = final_df[final_df['Income'] ==0]

# Removing Outliers based on our histograms: 
final_df = final_df[final_df['Airfare'] <= 125]
final_df = final_df[final_df['FlyingCompanion'] < 20]
final_df = final_df[final_df['AccessTime'] <= 350]

# Calculate avg income
avg_income = final_df.groupby(['Occupation'])['Income'].agg(pd.Series.mean).reset_index()
avg_income = avg_income.rename(columns = {'Income': 'avg_income'})

# calculate avg access time: 
avg_at = final_df.groupby(['Airport','ProvinceResidence'])['AccessTime'].agg(pd.Series.mean).reset_index()
avg_at = avg_at.rename(columns = {'AccessTime': 'avg_accesstime'})

# Calculate avg airfare
avg_airfare = final_df.groupby(['Airport','Destination','Airline'])['Airfare'].agg(pd.Series.mean).reset_index()
avg_airfare = avg_airfare.rename(columns = {'Airfare': 'avg_airfare'})


# created final dataframe includes avg_income
airport = final_df.merge(avg_income, left_on = ['Occupation'], right_on = ['Occupation'])
airport = airport.merge(avg_at, left_on = ['Airport','ProvinceResidence'], right_on = ['Airport','ProvinceResidence'])
airport = airport.merge(avg_airfare, left_on = ['Airport','Destination','Airline'], right_on = ['Airport','Destination','Airline'])

# Replace NaN with the reference avg_income

for i in range(0,len(airport)):
    if airport.Income[i] == 0:
        airport.Income[i] = round(airport.avg_income[i],0)

# Replace NaN with the reference avg_accesstime

for j in range(0,len(airport)):
    if airport.AccessTime[j] == 0:
        airport.AccessTime[j] = round(airport.avg_accesstime[j],0)
        
    
# Replace NaN with the reference avg_airfare
for k in range(0,len(airport)):
    if airport.Airfare[k] == 0:
        airport.Airfare[k] = round(airport.avg_airfare[k],0)


# Droping columns: 

for c in airport.columns:
    if c in ['ID','FlightNo','AccessCost','MileageAirline','Mileage','FrequentFlightDestination','DepartureHr'
    ,'DepartureMn', 'avg_income','avg_accesstime','avg_airfare']:
        del airport[c]
        
airport = airport[airport['Airfare'] >0]

# Create excel clean_airport for Decision Tree
'''
airport.to_excel(r'C:\Users\Yumi\Desktop\data mining\Project\Clean_Airport.xlsx',index = None, header = True)
'''