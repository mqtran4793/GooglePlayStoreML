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

#Start to use the new update data frame for next tasks
df3 = pd.read_csv('googleplaystore_Update1.csv')
# Inport data into new column using info of Installs
row = 0
for data in df3['Installs']:
	if data < 100000:
		df3.loc[row,'Popularity'] = 0#Low
	elif data < 10000000:
		df3.loc[row,'Popularity'] = 1#Medium
	else:
		df3.loc[row,'Popularity'] = 2#High
	row = row+1
#Store to new csv file after preprocess data
df3.to_csv('googleplaystore_Update2.csv', index=False)
# lists holding unique of each features
Cat_list = tuple(df3['Category'].unique())
Cont_list = tuple(df3['Content Rating'].unique())
Vers_list = tuple(df3['Android Ver'].unique())
Genres_list = tuple(df3['Genres'].unique())

i =0;
c = tuple(range(33))
# 33 unique of Category
df3['Category'] = df3['Category'].map({Cat_list[0] : c[0],Cat_list[11] : c[11],Cat_list[22] : c[22],
									   Cat_list[1] : c[1],Cat_list[12] : c[12],Cat_list[23] : c[23],
									   Cat_list[2] : c[2],Cat_list[13] : c[13],Cat_list[24] : c[24],
									   Cat_list[3] : c[3],Cat_list[14] : c[14],Cat_list[25] : c[25],
									   Cat_list[4] : c[4],Cat_list[15] : c[15],Cat_list[26] : c[26],
									   Cat_list[5] : c[5],Cat_list[16] : c[16],Cat_list[27] : c[27],
									   Cat_list[6] : c[6],Cat_list[17] : c[17],Cat_list[28] : c[28],
									   Cat_list[7] : c[7],Cat_list[18] : c[18],Cat_list[29] : c[29],
									   Cat_list[8] : c[8],Cat_list[19] : c[19],Cat_list[30] : c[30],
									   Cat_list[9] : c[9],Cat_list[20] : c[20],Cat_list[31] : c[31],
									   Cat_list[10] : c[10],Cat_list[21] : c[21],Cat_list[32] : c[32]})
# 2 unique of Type
df3['Type'] = df3['Type'].map({'Free': 0, 'Paid': 1})
# 6 unique of Content Rating
df3['Content Rating'] = df3['Content Rating'].map({Cont_list[0]: c[0], Cont_list[2]: c[2], Cont_list[4]: c[4],
												   Cont_list[1]: c[1], Cont_list[3]: c[3], Cont_list[5]: c[5],})
# 33 unique of ANdroid Version
df3['Android Ver'] = df3['Android Ver'].map({Vers_list[0] : c[0],Vers_list[11] : c[11],Vers_list[22] : c[22],
										     Vers_list[1] : c[1],Vers_list[12] : c[12],Vers_list[23] : c[23],
										     Vers_list[2] : c[2],Vers_list[13] : c[13],Vers_list[24] : c[24],
										     Vers_list[3] : c[3],Vers_list[14] : c[14],Vers_list[25] : c[25],
										     Vers_list[4] : c[4],Vers_list[15] : c[15],Vers_list[26] : c[26],
										     Vers_list[5] : c[5],Vers_list[16] : c[16],Vers_list[27] : c[27],
										     Vers_list[6] : c[6],Vers_list[17] : c[17],Vers_list[28] : c[28],
										     Vers_list[7] : c[7],Vers_list[18] : c[18],Vers_list[29] : c[29],
										     Vers_list[8] : c[8],Vers_list[19] : c[19],Vers_list[30] : c[30],
										     Vers_list[9] : c[9],Vers_list[20] : c[20],Vers_list[31] : c[31],
										     Vers_list[10] : c[10],Vers_list[21] : c[21],Vers_list[32] : c[32]})	

# 119 unique of Genres
i = 0
while i < 119:
	df4 = df3['Genres'].map({Genres_list[i] : tuple(range(119))[i]})
	df4 = df4.dropna()
	if i == 0:
		df5 = df4.dropna()
	if i > 0 :
		df5 = df5.append(df4)
	i += 1
df3['Genres'] = df5										 
# store frames into new file after finish precessing												   
df3.to_csv('googleplaystore_Update3.csv', index=False)
# start with newly create file
df4 = pd.read_csv('googleplaystore_Update3.csv')
# reduce unused column to have better viewitems
columns = ['Last Updated','Current Ver','App', 'Reviews', 'Size', 'Price']
df4 = df4.drop(columns, axis=1)  
df4.to_csv('googleplaystore_Update4.csv', index=False)

###################################################Decisionn tree
train, test = TTS(df4, test_size = 0.2)	
print("Trainning size: {}; Test size: {}".format(len(train), len(test)))
c = tree.DecisionTreeClassifier(max_depth= None)
features = ['Category','Type','Content Rating','Genres','Android Ver']

x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']

# Decision Tree model
dt= c.fit(x_train,y_train)

#predict test set and train set
y_predictions = dt.predict(x_test)
y_train_predictions = dt.predict(x_train)

# Accuracy check for test and train set
print('Class Decision Tree:')
print('accuracy of test set = ', accuracy_score(y_test, y_predictions))
print('accuracy of train set = ', accuracy_score(y_train, y_train_predictions))
#export tree to image
dot_data = StringIO()
tree.export_graphviz(c,
                     out_file = dot_data,
                     feature_names = features,
                     class_names= 'Popularity', 
                     filled = True,
                     rounded = True,
                     impurity = False
                    )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_jpg('ggplay.jpg')  #save graph as image
print('File saved as a JPG.')
##################################################################Random forest
train, test = TTS(df4, test_size = 0.2)	
x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)

print('Random Forest:')
print("Accuracy on training set: {:.3f}".format(forest.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(x_test, y_test)))

###############################################################################KNN
train, test = TTS(df4, test_size = 0.2)	
x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']
# KNN model
classifier = KNeighborsClassifier(n_neighbors=5)  
KNN = classifier.fit(x_train, y_train)  
# prediction for both train and test set
y_predictions = KNN.predict(x_test)
y_train_predictions = KNN.predict(x_train)

print('KNN classifier: ')
print('KNN accuracy of test set = ', accuracy_score(y_test, y_predictions))
print('KNN accuracy of train set = ', accuracy_score(y_train, y_train_predictions))
##########################################################################SVC
train, test = TTS(df4, test_size = 0.2)	
x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']
#SVC model
SVCmodel = SVC()
SVCmodel.fit(x_train, y_train)
print('SVC algorithm:')
print("Accuracy on training set: {:f}".format(SVCmodel.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(SVCmodel.score(x_test, y_test)))

##########################################################################GradientBoostingClassifier
train, test = TTS(df4, test_size = 0.2)	
x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']
#SVC model
GBCmodel = GradientBoostingClassifier()
GBCmodel.fit(x_train, y_train)
print('GradientBoostingClassifier algorithm:')
print("Accuracy on training set: {:f}".format(GBCmodel.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(GBCmodel.score(x_test, y_test)))

##########################################################################GradientBoostingClassifier
train, test = TTS(df4, test_size = 0.2)	
x_train = train[features]
y_train = train['Popularity']
x_test = test[features]
y_test = test['Popularity']
#GNB model
GNBmodel = GaussianNB()
GNBmodel.fit(x_train, y_train)
print('GNB algorithm:')
print("Accuracy on training set: {:f}".format(GNBmodel.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(GNBmodel.score(x_test, y_test)))
