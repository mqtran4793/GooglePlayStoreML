import numpy as np
import scipy as sp
import pandas as pd
import tkinter as tk
import csv
import graphviz
import io
import pydotplus
import os

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from PIL import Image, ImageTk
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
from PIL import Image
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as TTS
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt



#############################################read the original csv into a dataframe
df = pd.read_csv('googleplaystore.csv')

#############################################Preprocess data
#drop NAN in used features
df2 = df.dropna(subset=['Type', 'Content Rating', 'Current Ver', 'Android Ver'])
#impute missing value of Rating with mean value
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df2['Rating'] = imp.fit_transform(df2[['Rating']])
#Remove '+' in the 'Installs' collumn of data frame
df2['Installs'] = df2['Installs'].map(lambda x: x.rstrip('+'))
#Replace ',' in the 'Installs' collumn of data frame with empty string
df2['Installs'] = df2['Installs'].str.replace(',', '')
#Convert string to int for 'Installs' number
df2['Installs'] = df2['Installs'].astype("int")
#Create new 'Popularity' column
df2['Popularity'] = ''
df2.to_csv('googleplaystore_Update1.csv', index=False)

