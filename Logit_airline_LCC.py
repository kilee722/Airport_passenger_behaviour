# Airline Choice Model
# Group 7

import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from sklearn.metrics import *
import statsmodels.formula.api as smf
import random
import matplotlib.pyplot as plt

airport = pd.read_excel("Regroup_Airport.xlsx")


LCC = airport[airport['Airline_LCC'] ==1]
Non_LCC = airport[airport['Airline_LCC'] ==0]

random_test = Non_LCC.sample(81)

frames = [LCC,random_test]

df = pd.concat(frames)
 
df['intercept'] = 1.0

X = df[['Airfare','Destination_Near','intercept']]  

y = df['Airline_LCC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=109)

logit_model = sm.Logit(y_train,X_train).fit()

print(logit_model.summary())
print("AIC: ", logit_model.aic)
print("BIC: ", logit_model.bic)

y_pred_test = logit_model.predict(X_test)

y_pred_train = logit_model.predict(X_train)

X_test.loc[:,'prediction'] = 0 
X_test.loc[y_pred_test > 0.5, 'prediction'] = 1

X_train.loc[:,'prediction'] = 0 
X_train.loc[y_pred_train > 0.5, 'prediction'] = 1

#print this to get the confusion matrix

print("train data confusion matrix:")
print(pd.crosstab(y_train,X_train['prediction'],rownames =['actual'],colnames=['predicted']))

print("test data confusion matrix:")
print(pd.crosstab(y_test,X_test['prediction'],rownames =['actual'],colnames=['predicted']))

# Making ROC curve
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    return plt.show()

fper, tper, thresholds = roc_curve(y_test, y_pred_test) 
plot_roc_cur(fper, tper)