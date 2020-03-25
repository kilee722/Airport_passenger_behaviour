# Regrouping codes:
# Group 7

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



airport = pd.read_excel('Clean_Airport.xlsx')
airport['Airport'] = airport['Airport'].astype('category').cat.codes

# Regroup:

## Airline: Korean airlines vs. foreign:
airport['Airline_KA'] = np.where(airport['Airline'] == 4,0,1)

# Airline: the top airlines in South Korea: KE and ASIANA vs. the other, using for airline choice model
airport['Airline_Big'] = np.where(airport['Airline'] < 3,1,0)

# Airline: LCC vs. Other, using for airline choice model
airport['Airline_LCC'] = np.where(airport['Airline'] == 3,1,0)


## Nationality: Korean vs. non Korean
airport['Nationality_Korean'] = np.where(airport['Nationality'] ==1, 1,0)


 ## Younger than 50 vs. older than 50
airport['Age_35'] = np.where(airport['Age'] <=35,1,0)

 ## solo traveler vs. travel with companion
airport['FlyingCompanion_Small'] = np.where(airport['FlyingCompanion'] <=1,1,0)

## Less than Average <5
airport['Income_Avg'] = np.where(airport['Income'] < 5, 1,0)


## Ocupation: Corporate, Non Corporate, Other Using for Airport choices model
# Other is referencing variable
airport['Occupation_Corporate'] = np.where(airport['Occupation'] ==2,1,0)
airport['Occupation_NonCorporate'] = np.where((airport['Occupation'] !=2) | (airport['Occupation'] !=12 ),1,0)

## Ocupation: Business and Professional - Using for LCC model
# Other is referencing variable
airport['Occupation_ProfessionalBusiness'] = np.where((airport['Occupation'] ==2) | (airport['Occupation'] ==5) ,1,0)

# Trip Duration: <= 8 days and others
airport['TripDuration_Short'] = np.where(airport['TripDuration'] <= 8, 1,0)


 ## Leisure travel vs. non leisure
airport['TripPurpose_Leisure'] = np.where(airport['TripPurpose'] ==1,1,0)


 ## late night and redeye flights vs. daytime
airport['DepartureTime_Night'] = np.where((airport['DepartureTime'] == 3) | (airport['DepartureTime'] == 4),1,0)

## DepartureTime afternoon vs. rest, use for airline choice model
airport['DepartureTime_Noon'] = np.where(airport['DepartureTime'] == 2,1,0)


 ##No transfer vs. needs transfer
airport['Transport_Transfer'] = np.where(airport['NoTransport'] == 1,0,1)

 ## SEA vs. others
airport['Destination_SEA'] = np.where(airport['Destination'] == 3,1,0)


## Destination: Japan and China vs. other: 
airport['Destination_Near'] = np.where(airport['Destination'] < 3,1,0)
 
 
 ## Public transportation vs. others 3,4,5,7,8
fill = [3,4,5,7,8]
airport['ModeTransport_Public'] = np.where(airport['ModeTransport'].isin(fill),1,0)

 ## Accesstime divide in 3 buckets: 0-30, 21-60, above 60
 # Reference variable is above 60
airport['Accesstime_[0-30]'] = np.where(airport['AccessTime'] <= 30, 1,0)
airport['Accesstime_[30-60]'] = np.where((airport['AccessTime'] > 30) & (airport['AccessTime'] <=60) , 1,0)

# NoTransportation
airport['NoTrasnport_Easy'] = np.where(airport['NoTransport'] <= 1, 1,0)

# ProvinceResidence: 
airport['ProvinceResidence_Central'] = np.where(airport['ProvinceResidence'] <= 2, 1,0)



# Export new data to excel
'''
airport.to_excel(r'C:\Users\Yumi\Desktop\data mining\Project\Regroup_Airport.xlsx',index = None, header = True)
'''

