#Supervised model -> Classification

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#Data collection
df = pd.read_csv(r'C:\Users\rithu\AppData\Local\Programs\Python\Python39\Programs\HeartDiseaseClassification\heart.csv')
print(df)

#Data Preprocessing

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Convert alphabets to numerics - > LabelEncoding
#gives 0,1,2... based on alphabetic order
#df['Sex'] = le.fit_transform(df['Sex'])
#df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
#df['RestingECG'] = le.fit_transform(df['RestingECG'])
#df['ExercisingAngina'] = le.fit_transform(df['ExercisingAngina'])
#df['ST_Slope'] = le.fit_transform(df['ST_Slope'])

print(df)

x = df.drop(columns=['HeartDisease']) #Input
y = df['HeartDisease'] #Output

print("XXXX" ,x)
print("YYYY" ,y)

#Split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)

#0.2 -> Train:80% ,Test = 20%.If 0.3 ->Train:70,Test:30

print("DF", df.shape) #shape -> displays rows and columns
print("x_train",x_train.shape) #80% input data
print("x_test",x_test.shape)   #20% input data
print("y_train",y_train.shape) #80% output data
print("y_test",y_test.shape)   #80% output data

#Model training

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(x_train, y_train)

#Model eval

y_pred = NB.predict(x_test)

print("y_pred" , y_pred) #prediction
print("y_test" , y_test) #Actual op

from sklearn.metrics import accuracy_score
print("Accuracy is" , accuracy_score(y_test ,y_pred)) #comparing prediction and Actual op  

#Model prediction:

testPred = NB.predict([[59,1,3,170,288,0,0,159,0,0.2,1,0,3]]) #Data from dataset
if testPred ==1:
    print("The patient has heart disease")
else:
    print("The patient Normal")
      
