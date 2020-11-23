#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from error import rmse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox



class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("Recommenders System")
        self.minsize(640,400)
        self.configure(background = "#A9A9A9")
        self.labelFrame = ttk.LabelFrame(self, text = "------Recommenders System------")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        self.textarea = tk.Text(self)
        self.textarea.grid(column = 0, row = 8)
        global n_flag
        self.n_flag = False
        self.button_find_file()
        self.labelFrame = ttk.LabelFrame(self, text = "****** Algorithms ******")
        self.labelFrame.grid(column = 0, row = 3, padx = 20, pady = 20)
        self.button_LinearRegression()
        self.button_RandomForestRegressor()
        self.button_GenerateOutput()
        self.button_exit()

    def button_find_file(self):
        self.button = ttk.Button(self.labelFrame, text = "Import a File", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def button_LinearRegression(self):
        self.button = ttk.Button(self.labelFrame, text = "Linear Regression", command = self.LinearRegression)
        self.button.grid(column = 0, row = 3)

    def button_RandomForestRegressor(self):
        self.button = ttk.Button(self.labelFrame, text = "Random Forest Regressor", command = self.RandomForestRegressor)
        self.button.grid(column = 0, row = 4)

    def button_GenerateOutput(self):
        self.button = ttk.Button(self.labelFrame, text = "Generate Output file using Random Forest Regressor", command = self.GenerateOutput)
        self.button.grid(column = 0, row = 5)

    def button_exit(self):
        self.button = ttk.Button(self.labelFrame, text = "Exit", command = self.close_window)
        self.button.grid(column = 0, row = 7)

    def close_window(self):
        self.destroy()

    def fileDialog(self):
        global filename
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes = (("csv files","*.csv"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.import_content()
        self.DataSplit()
        if self.flag == True :
            self.textarea.delete("1.0","end")
            self.textarea.insert(END, "CSV File imported successfully!")
        else:
            self.textarea.delete("1.0","end")
            self.textarea.insert(END, "Some error occured!")

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
        if self.n_flag == True :
            '''1. LinearRegression Model'''
            self.textarea.delete("1.0","end")
            self.textarea.insert(END, "**************** LinearRegression Model ****************\n")

            reg = LinearRegression()
            reg.fit(train_x, train_y)
            predicted_L = reg.predict(test_x)
            text = 'Model Score: ' + str(reg.score(test_x, test_y))
            self.textarea.insert(END, text + "\n")
            '''RMSE Value for Predicted'''
            text = 'RMSE value for LinearRegression Model : ' + str(rmse(predicted_L, test_y))
            self.textarea.insert(END, text + "\n")
        else:
            self.textarea.delete("1.0","end")
            self.textarea.insert(END, "Import CSV file first!")

    def RandomForestRegressor(self):
        if self.n_flag == True :
            '''2. Random Forest Model'''

            self.MsgBox = tk.messagebox.askyesno ('Clear TextArea','Do you want to clear TextArea?\n')
            if self.MsgBox == True:
                self.textarea.delete("1.0","end")
                self.textarea.insert(END, "**************** Random Forest Model ****************\n")
            else:
                self.textarea.insert(END, "**************** Random Forest Model ****************\n")

            regr_R = RandomForestRegressor(max_depth=2, random_state=0)
            regr_R.fit(train_x, train_y)

            predicted_R = regr_R.predict(test_x)
            text = 'Model Score: ' + str(regr_R.score(test_x, test_y))
            self.textarea.insert(END, text + "\n")

            '''RMSE Value for Predicted'''
            text = 'RMSE value for Random Forest Model : ' + str(rmse(predicted_R, test_y))
            self.textarea.insert(END, text + "\n")
        else:
            self.textarea.delete("1.0","end")
            self.textarea.insert(END, "Import CSV file first!\n")

    def rmse_cal(self):
        self.DataSplit()
        regr_R = RandomForestRegressor(max_depth=2, random_state=0)
        regr_R.fit(train_x, train_y)

        predicted_R = regr_R.predict(test_x)
        text = 'Model Score: ' + str(regr_R.score(test_x, test_y))
        self.textarea.insert(END, text + "\n")

        '''RMSE Value for Predicted'''
        text = 'RMSE value for Random Forest Model : ' + str(rmse(predicted_R, test_y))
        self.textarea.insert(END, text + "\n")

    def GenerateOutput(self):
        self.DataSplit2()
        self.textarea.insert(END, "\n\n**************** Random Forest Model ****************\n")
        neigh = KNeighborsClassifier()
        neigh.fit(train_x, train_y)

        predicted_K = neigh.predict(test_x)
        text = 'Model Score: ' + str(neigh.score(test_x, test_y))
        self.textarea.insert(END, text + "\n")

        self.rmse_cal()

        self.DataSplit2()

        users =  943 #1...943
        items = 1682 #1...1682
        ratings = 0  #1...5

        file_output = open("Predicted_values.txt", "w+")

        self.textarea.insert(END, "Predicting the ratings given for 943 users and 1682 items ....")
        self.textarea.insert(END, "\n\n\t\tWait time 12 mins or less...\n")
        self.MsgBox = tk.messagebox.showinfo("Generated Output File", "Predicted the ratings given for 943 users and 1682 items ....\nOUTPUT File name: Predicted_values.txt")

        Predicted_Ratings = []

        for i in range(1,users+1):
            for j in range(1,items+1):
                rating = neigh.predict([[i,j]])
                val = str(i)+" "+str(j)+" "+str(rating[0]) + "\n"
                file_output.write(val)

        self.MsgBox = tk.messagebox.showinfo("Done", "File Generated with name Predicted_values.txt!")
        file_output.close()

if __name__ == '__main__':
    root = Root()
    root.mainloop()
