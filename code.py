import pandas as pd
import numpy as np
from error import rmse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv("Data.csv")

'''For Normalization '''
min_rating = min(data["Rating"])
max_rating = max(data["Rating"])



X=data[['User','Item']]      #Independent Variables

'''Min-Max Normalization on Y'''
Y=data['Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values   #Dependent variable


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)


'''1. LinearRegression Model'''
print("**************** LinearRegression Model ****************")
reg = LinearRegression()
reg.fit(train_x, train_y)

predicted_L = reg.predict(test_x)
print('Model Score: ',reg.score(test_x, test_y))

'''RMSE Value for Predicted'''
print('RMSE value for LinearRegression Model : ', rmse(predicted_L, test_y))


'''2. Random Forest Model'''
print("**************** Random Forest Model ****************")
regr_R = RandomForestRegressor(max_depth=2, random_state=0)
regr_R.fit(train_x, train_y)

predicted_R = regr_R.predict(test_x)
print('Model Score: ',regr_R.score(test_x, test_y))

'''RMSE Value for Predicted'''
print('RMSE value for Random Forest Model : ', rmse(predicted_R, test_y))

#print(model.predict([[209,260]]))  #5
#print(model.predict([[130,367]]))  #5
#print(model.predict([[561,930]]))  #1
#print(model.predict([[483,1025]])) #4

#Actual_Y = (model.predict([[209,260]]) * (max_rating - min_rating)) + min_rating