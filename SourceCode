
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

#read the csv into a dataframe
df = pd.read_csv('googleplaystore.csv')
#Let print out the dataset to the console
#print(df)
print(df.count())
print(df.isnull().sum())
#print(df.describe())
#print(df['Reviews'].describe())
df2 = df.dropna(subset=['Type', 'Content Rating', 'Current Ver', 'Android Ver'])
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df2['Rating'] = imp.fit_transform(df2[['Rating']])
#imputed_DF.columns = df2.columns
#imputed_DF.index = df2.index
print(df2.count())
print(df2.isnull().sum())
#print(df2.count())
#df3 = df.dropna()
#print(df3.count())
#print(df3.describe())
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputed_DF = pd.DataFrame(imp.fit_transform(df2))
#imputed_DF.columns = df2.columns
#imputed_DF.index = df2.index
#print(imputed_DF.describe())
