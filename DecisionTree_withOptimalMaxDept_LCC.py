#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection  import train_test_split
from matplotlib.legend_handler import HandlerLine2D


#Load Data 
airport = pd.read_excel("Clean_Airport.xlsx")

#Airline Regroup 
airport['Airline'] = np.where(airport['Airline'] ==3, 1,0)

#Data Train & Test 
X = airport[['Airport', 'Age', 'Gender','Nationality','TripPurpose','TripDuration','FlyingCompanion', 'ProvinceResidence', 
             'GroupTravel', 'NoTripsLastYear', 'Destination', 'DepartureTime','SeatClass', 'Airfare', 'NoTransport',
             'ModeTransport', 'AccessTime','Occupation','Income']]

y = airport['Airline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=109)



#Decision Tree model 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6, max_features=None,
max_leaf_nodes=None, min_samples_leaf=20, min_samples_split=20,min_weight_fraction_leaf=0, presort=False,
random_state=100, splitter='best')

#Fit 
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  


def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    return plt.show()

fper, tper, thresholds = roc_curve(y_test, y_pred) 
plot_roc_cur(fper, tper)