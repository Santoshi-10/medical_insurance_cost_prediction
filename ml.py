#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#loading dataset
insurance=pd.read_csv("data.csv")
#converting categorical data into numerical data
encoder = LabelEncoder()
insurance["sex"] = encoder.fit_transform(insurance["sex"])
insurance["smoker"] = encoder.fit_transform(insurance["smoker"])
insurance["region"] = encoder.fit_transform(insurance["region"])
#taking x and y values
x = insurance.drop(columns='charges',axis = 1)
y = insurance['charges']
#splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=42)
#model fitting
regression= LinearRegression()
regression.fit(x_train,y_train)
#taking input values from user 
a=int(input('enter the age of the person: '))
b=input('enter gender: ')
c=float(input('enter BMI of the person: '))
e=int(input('enter how many children he/she have: '))
f=input('smoker/not: ')
region=input('enter region of the person: ')
be=encoder.fit_transform([b])[0]
fe=encoder.fit_transform([f])[0]
re=encoder.fit_transform([region])[0]
new_data=[a,be,c,e,fe,re]
reshape_new=np.asarray(new_data).reshape(1,-1)
prediction=regression.predict(reshape_new)
print('the charges of the person have',prediction[0])