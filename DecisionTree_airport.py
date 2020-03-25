#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import numpy as np 
import pandas as pd
import graphviz 
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection  import train_test_split
from matplotlib.legend_handler import HandlerLine2D

os.chdir("C:/Users/yunju/Desktop/DT_3_9_20")

#Load Data 
airport = pd.read_excel("Clean_Airport.xlsx")


#Data Train & Test 
X = airport[['Airline', 'Age', 'Gender','Nationality','TripPurpose','TripDuration','FlyingCompanion', 'ProvinceResidence', 
             'GroupTravel', 'NoTripsLastYear', 'Destination', 'DepartureTime','SeatClass', 'Airfare', 'NoTransport',
             'ModeTransport', 'AccessTime','Occupation','Income']]

y = airport['Airport'].astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=109)  


# In[18]:


#Decision Tree with 'default' parameters - no max_depth 
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
 max_features=None, max_leaf_nodes=None,
 min_impurity_split=1e-07, min_samples_leaf=1,
 min_samples_split=2, min_weight_fraction_leaf=0.0,
 presort=False, random_state=None, splitter='best')
y_pred = clf.predict(X_test)

#Visualize the Tree 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True, feature_names = X.columns,class_names=['INC','GMP'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_airport_default.png')
Image(graph.create_png())
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True, feature_names = X.columns,class_names=['INC','GMP'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_airport_default.png')
Image(graph.create_png())


# In[4]:


#AUC 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#Score: 
0.8049242424242424


# In[19]:


#Max Depth - Compare Train AUC vs Test AUC 
#Test for Overfitting/underfitting 
max_depths = np.linspace(3, 15, 15, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
   
    # Add auc score to previous train results
    train_results.append(roc_auc)
    y_predict = clf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
    roc_AUC = auc(false_positive_rate, true_positive_rate)
   
    # Add auc score to previous test results
    test_results.append(roc_AUC)

    
#Plot the graph - AUC comparison 
plt.title('Depth of Decision Tree')
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

#Result: Beyond max depth around 5 shows that our model overfits for larger Tree depth values. 


# In[14]:


# Max_Depth - Test Accuracies 
dep = np.arange(3, 14)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of the tree depth 
for i, k in enumerate(dep):
    clf = tree.DecisionTreeClassifier(max_depth=k)
    clf.fit(X_train, y_train)

    #Compute accuracy on the train set 
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the test set 
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate Graph 
plt.title('Depth of Decision Tree')
plt.plot(dep, test_accuracy, label = 'TestAccuracy')
plt.plot(dep, train_accuracy, label = 'Train Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:





# In[22]:


#Data Train & Test with optimal max_depth 
X = airport[['Airline', 'Age', 'Gender','Nationality','TripPurpose','TripDuration','FlyingCompanion', 'ProvinceResidence', 
             'GroupTravel', 'NoTripsLastYear', 'Destination', 'DepartureTime','SeatClass', 'Airfare', 'NoTransport',
             'ModeTransport', 'AccessTime','Occupation','Income']]

y = airport['Airport'].astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=109)  

#Decision Tree model 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None,
max_leaf_nodes=None, min_samples_leaf=20, min_samples_split=20,min_weight_fraction_leaf=0, presort=False,
random_state=100, splitter='best')

#Fit 
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  


#Data Visulization 

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True, feature_names = X.columns,class_names=['INC','GMP'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_max_depth_5.png')
Image(graph.create_png())
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True, feature_names = X.columns,class_names=['INC','GMP'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_max_depth_5.png')
Image(graph.create_png())


# In[23]:


#Confusion matrix - Max_depth = 5 
print(metrics.confusion_matrix(y_test, y_pred))  
test_df = metrics.confusion_matrix(y_test, y_pred)
test_df2 = pd.DataFrame(test_df)

tn = test_df2.loc[0][0] # true negative
fp = test_df2.loc[0][1] # false positive
fn = test_df2.loc[1][0] # false negative
tp = test_df2.loc[1][1] # true positive

Accuracy = round((tn + tp) / (tn + fn + fp + tp), 3)
Recall = round(tp / (tp + fn), 3)
Precision = round(tp / (tp + fp), 3)
Test_Error_Rate = round(1 - Accuracy, 3)

print('Accuracy: ' + str(Accuracy))
print('Recall: ' + str(Recall))
print('Precision: ' + str(Precision))
print('Test Error Rate: ' + str(Test_Error_Rate))

