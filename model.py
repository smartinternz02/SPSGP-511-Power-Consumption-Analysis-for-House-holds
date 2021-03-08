# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:28:07 2020

@author: ANIL KUMAR REDDY
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("household_power_consumption.txt",sep=';',low_memory=False)

p= data.isnull().any()

data['Global_active_power'] = pd.to_numeric(data['Global_active_power'],errors='coerce')
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'],errors='coerce')
data['Voltage'] = pd.to_numeric(data['Voltage'],errors='coerce')
data['Global_intensity'] = pd.to_numeric(data['Global_intensity'],errors='coerce')
data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'],errors='coerce')
data['Sub_metering_2'] = pd.to_numeric(data['Sub_metering_2'],errors='coerce')

p=data.isnull().any()

sns.heatmap(data.corr())

data['Global_active_power'].fillna(data['Global_active_power'].mean(),inplace = True)
data['Global_reactive_power'].fillna(data['Global_reactive_power'].mean(),inplace = True)
data['Voltage'].fillna(data['Voltage'].mean(),inplace = True)
data['Global_intensity'].fillna(data['Global_intensity'].mean(),inplace = True)
data['Sub_metering_1'].fillna(data['Sub_metering_1'].mean(),inplace = True)
data['Sub_metering_2'].fillna(data['Sub_metering_2'].mean(),inplace = True)
data['Sub_metering_3'].fillna(data['Sub_metering_3'].mean(),inplace = True)

data_reindex=data.reindex(columns=['Global_active_power','Global_reactive_power','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','Voltage'])

x = data_reindex.iloc[:50000,0:6].values
y = data_reindex.iloc[:50000,6:7].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state =1) 

from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 100)

regressor.fit(x_train, y_train) 
y_pred =regressor.predict(x_test)

from sklearn.metrics import r2_score
accuracy = r2_score(y_test , y_pred)

import pickle
with open ('household_power_consumption.pkl','wb') as f: pickle.dump(regressor,f)
model = pickle.load(open('household_power_consumption.pkl','rb'))

