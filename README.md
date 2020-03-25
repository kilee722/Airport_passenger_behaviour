# Airport_passenger_behaviour
Modelling for airport passengers' airport and airline choice behaviours. Decision Tree and Logistic Model are employed in attempt to describe the airport passengers' behaiours in airport and airline selection choice.

## Getting started

The survey data is from 488 respondents from air travelers in Seoul Metropolitan Area, who departed either from the Gimpo Airport or Incheon Airport. The data includes 27 variables including the Airport Choice (GMP, ICN), Airline Choices(Korean Air (KE), Asiana Air (OZ), Korean LCC, Foreign Carriers), socio-demographic variables such as age, gender, occupation, income and other variables such as flight information, travel time, and the mode of transport. We can notice many missing fields due to the nature of survey data, privacy issues etc. 

The raw data is not permitted to share in public. Please contact at klee1@seattleu.edu for further information.

### Prerequisites

Require Python and R to run the technical appendices
Require scikit, numpy, pandas, metrics, and seabron packages for python files (Machine learning appendix)
Require gridExtra, tidyverse, readxl, dplyr packages for R files (EDA technical appendix)
Code for package load are included in all files 



### Model Comparison

#### Airport Choice Model

Logistic Regression Model 1 (with Number of Transport variable included in the preprocessed model)

![image](https://user-images.githubusercontent.com/55430338/77512343-13ece800-6e30-11ea-87cf-749d61a1176a.png)


Logistic Regression Model 2 (with Number of AccessTime variable included in the preprocessed model)

![image](https://user-images.githubusercontent.com/55430338/77512348-18b19c00-6e30-11ea-9d47-7e1881c790f7.png)


Decision Tree Model 

![image](https://user-images.githubusercontent.com/55430338/77512356-1ea77d00-6e30-11ea-9649-1eaf2b691417.png)



#### Airline Choice Model


Logistic Regression 1 (Korean & Asiana Airlines vs Foreign airlines & LCC)

![image](https://user-images.githubusercontent.com/55430338/77512407-38e15b00-6e30-11ea-8860-a665bca8465b.png)

Logistic Regression 2 (Korean LCC vs Other Airlines Choice)

![image](https://user-images.githubusercontent.com/55430338/77512422-40086900-6e30-11ea-84ce-c6bf086bd4f1.png)


### Model Evaluation

ROC, AUC, confusion matrix and R-squared values.
While the accuracy scores are very close, the logistic regression model slightly scores higher than the decision tree model.

### Conclusion

The objective of our study is to determine how air travelers in the Seoul Metropolitan area choose one airport and airline over another for traveling, and what factors influence their decision. Incheon International Airport (ICN) is the largest airport in South Korea, and one of the busiest airports in the world located between Yeongjong and Yongyu Islands (west of Incheon’s city center). Before Incheon was built in 2001, Gimpo (GMP) was the main international airport for Seoul located by the western end of the city. Our survey data has customers of the two biggest airlines in South Korea: Korean Air, and Asiana. In addition, the data has Korean LCC (low-cost airline) as well as a set of customers using foreign airlines. 

Given that GMP is the smaller and older airport and mainly has flights to China or Japan, Destination plays a very important role in customer’s decision on choosing between GMP or ICN in our model. The preliminary results of this paper show airline, customer’s nationality, number of transportations needed to get to the airport and whether it is a redeye flight or not are also significant in customer’s airport choice. Considering GMP was built long time ago and is closer to the city center, if the customer doesn’t want to do complicated transport to the airport, they will more likely to go with GMP and not ICN since it is a lot further out. On the other hand, travelers are not allowed to stay overnight at the GMP airport, thus we expect if the person has late evening or nighttime flight, they will most likely to choose ICN since ICN is bigger and open 24/7. If a customer is not Korean and not familiar with Seoul area airports, they might not be aware of the GMP’s condition and will be slightly more likely to choose it while if the customer is Korean, they would know the policy as well as history of GMP and would prefer to use ICN. 


Download and see FINAL_REPORT-With_processed_data.docx for further details.



