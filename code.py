#!/usr/bin/env python
import pandas as pd
import os
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from error import rmse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier


class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("Recommenders System")
        self.minsize(640,400)
        self.configure(background = "#A9A9A9")
        self.labelFrame = ttk.LabelFrame(self, text = "------Recommenders System------")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        global n_flag
        self.n_flag = False
        self.button_find_file()
        self.button_LinearRegression()
        self.button_RandomForestRegressor()
        self.button_KNNClustering()
        self.button_exit()

    def button_find_file(self):
        self.button = ttk.Button(self.labelFrame, text = "Import a File", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def button_LinearRegression(self):
        self.button = ttk.Button(self.labelFrame, text = "Linear Regression", command = self.LinearRegression)
        self.button.grid(column = 3, row = 1)

    def button_RandomForestRegressor(self):
        self.button = ttk.Button(self.labelFrame, text = "Random Forest Regressor", command = self.RandomForestRegressor)
        self.button.grid(column = 3, row = 2)

    def button_KNNClustering(self):
        self.button = ttk.Button(self.labelFrame, text = "Generate Output file", command = self.KNNClustering)
        self.button.grid(column = 3, row = 3)

    def button_exit(self):
        self.button = ttk.Button(self.labelFrame, text = "Exit", command = self.close_window)
        self.button.grid(column = 1, row = 4)

    def close_window(self):
        self.destroy()

    def fileDialog(self):
        global filename
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes = (("csv files","*.csv"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.import_content()
        if self.flag == True :
            ttk.Label(text = "CSV File imported successfully!").grid(column = 0,row = 5)
        else:
            ttk.Label(text = "Some error occured!").grid(column = 0,row = 5)

    def import_content(self):
        global data
        self.flag = False
        cdir = os.path.dirname(__file__)
        file_res = os.path.join(cdir, self.filename)
        data = pd.read_csv(file_res)
        self.flag = True

    def matrix_init(self):
        users = 943
        items = 1682
        col = 3

        matrix_initialisation = np.zeros(users * items, col)     #Matrix initialisation with 0's


    def DataSplit(self):
        '''For Normalization '''
        global train_x, test_x, train_y, test_y
        min_rating = min(data["Rating"])
        max_rating = max(data["Rating"])

        X = data[['User','Item']]      #Independent Variables

        '''Min-Max Normalization on Y'''
        Y = data['Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values   #Dependent variable

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.n_flag = True

    def DataSplit2(self):
        '''For Normalization '''
        global train_x, test_x, train_y, test_y
        min_rating = min(data["Rating"])
        max_rating = max(data["Rating"])

        X = data[['User','Item']]      #Independent Variables
        y=data['Rating']

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        self.n_flag = True

    def LinearRegression(self):
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        if self.n_flag == True :
            '''1. LinearRegression Model'''
            print("**************** LinearRegression Model ****************")
            self.DataSplit()
            reg = LinearRegression()
            reg.fit(train_x, train_y)
            predicted_L = reg.predict(test_x)
            print('Model Score: ',reg.score(test_x, test_y))
            '''RMSE Value for Predicted'''
            print('RMSE value for LinearRegression Model : ', rmse(predicted_L, test_y))
        else:
            ttk.Label(text = "Import CSV file first!").grid(column = 0,row = 5)

    def RandomForestRegressor(self):
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        if self.n_flag == True :
            '''2. Random Forest Model'''
            print("**************** Random Forest Model ****************")
            self.DataSplit()
            regr_R = RandomForestRegressor(max_depth=2, random_state=0)
            regr_R.fit(train_x, train_y)

            predicted_R = regr_R.predict(test_x)
            print('Model Score: ',regr_R.score(test_x, test_y))

            '''RMSE Value for Predicted'''
            print('RMSE value for Random Forest Model : ', rmse(predicted_R, test_y))
        else:
            ttk.Label(text = "Import CSV file first!").grid(column = 0,row = 5)


    def KNNClustering(self):
        '''4. KNN Clustering Classifier '''
        self.DataSplit2()
        print("\n\n")
        print("**************** KNN Model ****************")
        neigh = KNeighborsClassifier()
        neigh.fit(train_x, train_y)

        predicted_K = neigh.predict(test_x)
        print('Model Score: ',neigh.score(test_x, test_y))

        '''RMSE Value for Predicted'''
        print('RMSE value for Random Forest Model : ', rmse(predicted_K, test_y))
        print("\n\n")

        users =  15
        items = 10
        ratings = 0#1682

        file_output = open("Predicted_values.txt", "w+")

        print("\n")
        print("Predicting the ratings given for 15 items by 10 users  ....")
        Predicted_Ratings = []

        for i in range(1,users+1):
            for j in range(1,items+1):
                rating = neigh.predict([[i,j]])
                #Actual_rating = ( rating * (max_rating - min_rating)) + min_rating
                #Predicted_Ratings.append(rating)
                #print(i, j, rating)
                val = str(i)+" "+str(j)+" "+str(rating[0]) + "\n"
                file_output.write(val)
            #print( )

        print("Done!")
        file_output.close()
       # print("Predicted Ratings are : ", Predicted_Ratings)

if __name__ == '__main__':
    root = Root()
    root.mainloop()