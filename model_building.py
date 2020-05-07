#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:18:53 2020

@author: Jordan Chow
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

df = pd.read_csv('salary_data_cleaned.csv')

df_model = pd.DataFrame(df[['Rating','Size','Type of ownership','Industry',
                            'Sector','Revenue','hourly','python','R','spark',
                            'excel','avg_salary']])

# dreate dummy variables
df_dum = pd.get_dummies(df_model)

# Train test split
X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# multiple linear regression
s_sm = X = sm.add_constant(X)
model = sm.OLS(y,s_sm)
model.fit().summary()

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv=3))

# lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_1,X_train,y_train, scoring = 'neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/10)
    lm1 = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lm_1,X_train,y_train, scoring = 'neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha,error)
 
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])    
df_err[df_err.error == max(df_err.error)]    


# Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv=3))

# Tuning models with GridSearch
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

# Test ensenbles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])

 