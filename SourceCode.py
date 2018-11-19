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

################################################################# GUI
root = tk.Tk()
root.title('Blue Machine')
root.geometry('850x900')
root.resizable(width = False, height = False)
mainColor = 'misty rose'
root['bg']=mainColor
appPath = os.path.abspath(os.path.dirname(__file__))

topFrame = tk.Frame(root)
topFrame.pack(side='top')
panelImage = ImageTk.PhotoImage(file = os.path.join(appPath, 'panel.png'))
topLabel = tk.Label(topFrame, image=panelImage)
topLabel.pack()

midFrame = tk.Frame(root, bg=mainColor)
midFrame.pack()

emptyFrame = tk.Frame(root, width=600, heigh=30, bg=mainColor)
emptyFrame.pack()

bottomFrame = tk.Frame(root, bg=mainColor)
bottomFrame.pack()

aBottomFrame = tk.Frame(root)
aBottomFrame.pack()

OPTIONS = ['Free', 'Paid']
CategoryVar = tk.StringVar()
TypeVar = tk.StringVar()
ContentRatingVar = tk.StringVar()
GenresVar = tk.StringVar()
AndroidVerVar = tk.StringVar()
messageVar = tk.StringVar()
predictVar= tk.StringVar()
selection = tk.Message(bottomFrame, textvariable=messageVar, bg=mainColor)

y_predictions
def printSelection():
	messageVar.set('Category: ' + CategoryVar.get() + '\n'
				   'Type: ' + TypeVar.get() + '\n'
				   'Content Rating: ' + ContentRatingVar.get() + '\n'
				   'Genres: ' + GenresVar.get() + '\n'
				   'Android Version: ' + AndroidVerVar.get() + '\n')
	
	############################ Processing data input such that it can be applied to algorithm 
	x_test = [[CategoryVar.get(), TypeVar.get(), ContentRatingVar.get(), GenresVar.get(), AndroidVerVar.get()]]
	cc = pd.DataFrame(x_test, columns = features)
	c = tuple(range(33))
	cc['Category'] = cc['Category'].map({Cat_list[0] : c[0],Cat_list[11] : c[11],Cat_list[22] : c[22],
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
	cc['Type'] = cc['Type'].map({'Free': 0, 'Paid': 1})
	cc['Content Rating'] = cc['Content Rating'].map({Cont_list[0]: c[0], Cont_list[2]: c[2], Cont_list[4]: c[4],
													   Cont_list[1]: c[1], Cont_list[3]: c[3], Cont_list[5]: c[5],})
	cc['Android Ver'] = cc['Android Ver'].map({Vers_list[0] : c[0],Vers_list[11] : c[11],Vers_list[22] : c[22],
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
	c = tuple(range(119))	
	cc['Genres'] = cc['Genres'].map({Genres_list[0] : c[0],Genres_list[11] : c[11],Genres_list[22] : c[22],
												 Genres_list[1] : c[1],Genres_list[12] : c[12],Genres_list[23] : c[23],
												 Genres_list[2] : c[2],Genres_list[13] : c[13],Genres_list[24] : c[24],
												 Genres_list[3] : c[3],Genres_list[14] : c[14],Genres_list[25] : c[25],
												 Genres_list[4] : c[4],Genres_list[15] : c[15],Genres_list[26] : c[26],
												 Genres_list[5] : c[5],Genres_list[16] : c[16],Genres_list[27] : c[27],
												 Genres_list[6] : c[6],Genres_list[17] : c[17],Genres_list[28] : c[28],
												 Genres_list[7] : c[7],Genres_list[18] : c[18],Genres_list[29] : c[29],
												 Genres_list[8] : c[8],Genres_list[19] : c[19],Genres_list[30] : c[30],
												 Genres_list[9] : c[9],Genres_list[20] : c[20],Genres_list[31] : c[31],
												 Genres_list[10] : c[10],Genres_list[21] : c[21],Genres_list[91] : c[91],
												 Genres_list[33] : c[33],Genres_list[62] : c[62],Genres_list[92] : c[92],
												 Genres_list[34] : c[34],Genres_list[63] : c[63],Genres_list[93] : c[93],
												 Genres_list[35] : c[35],Genres_list[64] : c[64],Genres_list[94] : c[94],
												 Genres_list[36] : c[36],Genres_list[65] : c[65],Genres_list[95] : c[95],
												 Genres_list[37] : c[37],Genres_list[66] : c[66],Genres_list[96] : c[96],
												 Genres_list[38] : c[38],Genres_list[67] : c[67],Genres_list[97] : c[97],
												 Genres_list[39] : c[39],Genres_list[68] : c[68],Genres_list[98] : c[98],
												 Genres_list[40] : c[40],Genres_list[69] : c[69],Genres_list[99] : c[99],
												 Genres_list[41] : c[41],Genres_list[70] : c[70],Genres_list[100] : c[100],
												 Genres_list[42] : c[42],Genres_list[71] : c[71],Genres_list[101] : c[101],
												 Genres_list[43] : c[43],Genres_list[72] : c[72],Genres_list[102] : c[102],
												 Genres_list[44] : c[44],Genres_list[73] : c[73],Genres_list[103] : c[103],
												 Genres_list[45] : c[45],Genres_list[74] : c[74],Genres_list[104] : c[104],
												 Genres_list[46] : c[46],Genres_list[75] : c[75],Genres_list[105] : c[105],
												 Genres_list[47] : c[47],Genres_list[76] : c[76],Genres_list[106] : c[106],
												 Genres_list[48] : c[48],Genres_list[77] : c[77],Genres_list[107] : c[107],
												 Genres_list[49] : c[49],Genres_list[78] : c[78],Genres_list[108] : c[108],
												 Genres_list[50] : c[50],Genres_list[79] : c[79],Genres_list[109] : c[109],
												 Genres_list[51] : c[51],Genres_list[80] : c[80],Genres_list[110] : c[110],
												 Genres_list[52] : c[52],Genres_list[81] : c[81],Genres_list[111] : c[111],
												 Genres_list[53] : c[53],Genres_list[82] : c[82],Genres_list[112] : c[112],
												 Genres_list[54] : c[54],Genres_list[83] : c[83],Genres_list[113] : c[113],
												 Genres_list[55] : c[55],Genres_list[84] : c[84],Genres_list[114] : c[114],
												 Genres_list[56] : c[56],Genres_list[85] : c[85],Genres_list[115] : c[115],
												 Genres_list[57] : c[57],Genres_list[86] : c[86],Genres_list[116] : c[116],
												 Genres_list[58] : c[58],Genres_list[87] : c[87],Genres_list[117] : c[117],
												 Genres_list[59] : c[59],Genres_list[88] : c[88],Genres_list[118] : c[118],
												 Genres_list[60] : c[60],Genres_list[89] : c[89],
												 Genres_list[61] : c[61],Genres_list[90] : c[90]})	
	#Decision tree for GUI
	y_predictions = dt.predict(cc)#result
	print("prediction")
	predictVar.set(y_predictions)
	print(y_predictions)

	if y_predictions == 0:
		result = "Low"	#Low
	elif y_predictions == 1:
		result = "Medium" #Medium
	else:
		result = "High" #High

	print(result)
	########################################
	predictVar.set(result)
def createDropDownMenu(frameLabel, var, opt):
	frame = tk.Frame(midFrame, bg=mainColor)
	frame.pack(side='left')
	f = tk.Frame(midFrame, width=20, bg=mainColor)
	f.pack(side='left')
	label = tk.Label(frame, text=frameLabel, font='Helvetica 12 bold', bg=mainColor)
	label.pack(side='top')
	var.set('------') # default value
	var.trace('w', printSelection)
	window = tk.OptionMenu(frame, var, *opt)
	window.pack(side='bottom')
	
createDropDownMenu('Category', CategoryVar, Cat_list)
createDropDownMenu('Type', TypeVar, OPTIONS)
createDropDownMenu('Content Rating', ContentRatingVar, Cont_list)
createDropDownMenu('Genres', GenresVar, Genres_list)
createDropDownMenu('Android Ver.', AndroidVerVar, Vers_list)

##############################
submitIcon = ImageTk.PhotoImage(file = os.path.join(appPath, 'submit.png'))
submit = tk.Button(bottomFrame, image=submitIcon, command=printSelection)
submit.pack(side='top')
selection = tk.Message(bottomFrame, textvariable=messageVar, bg=mainColor, width=300)
selection.pack(side='bottom')

predictFrame = tk.Frame(aBottomFrame, bg=mainColor)
predictFrame.pack()
predictLabel = tk.Label(predictFrame, text='Predict Value', font='Helvetica 14 bold', bg=mainColor)
predictLabel.pack(side='top')
predictVal = tk.Message(predictFrame, textvariable=predictVar, bg=mainColor, width=100, font='Helvetica 12 bold')
predictVal.pack(side='bottom')

root.mainloop()
################################################################# end GUI
