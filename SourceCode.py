import pandas as pd
import csv

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

#read the csv into a dataframe
df = pd.read_csv('googleplaystore.csv')

#Preprocess data
df2 = df.dropna(subset=['Type', 'Content Rating', 'Current Ver', 'Android Ver'])
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
row = 0
for data in df3['Installs']:
	if data < 100000:
		df3.loc[row,'Popularity'] = 'Low'
	elif data < 10000000:
		df3.loc[row,'Popularity'] = 'Medium'
	else:
		df3.loc[row,'Popularity'] = 'High'
	row = row+1
	
#Store to new csv file after preprocess data
df3.to_csv('googleplaystore_Update2.csv', index=False)
