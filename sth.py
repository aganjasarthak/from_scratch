import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a sample dataset similar to an Excel file with 10 columns
data = {
    "A": ["Yes", "No", "Yes", "No", "Yes"],
    "B": [6, 7, 8, 9, 10],
    "C": [11, 12, 13, 14, 15],
    "D": [16, 17, 18, 19, 20],
    "E": [21, 22, 23, 24, 25],
    "F": [26, 27, 28, 29, 30],
    "G": [31, 32, 33, 34, 35],
    "H": [36, 37, 38, 39, 40],
    "I": [41, 42, 43, 44, 45],
    "J": [46, 47, 48, 49, 50],
     "K":[1,0,1,0,0],
}
data=pd.DataFrame(data)




column1 = data.iloc[:, 0]

print("Initial values in column1:")
print(column1)

# Check if all values in column1 are "Yes"
for i in range(len(data)):
   if data.loc[i, 'A'] == "Yes":
       data.loc[i, 'A'] = 1
   else:
       data.loc[i, 'A'] = 0
print("\nModified DataFrame:")
print(data)

print("xx")




column_others=data.iloc[:,2:6]
y_true_label=data.iloc[:,10]
finalx=pd.concat([column1,column_others],axis=1)

scaler = StandardScaler()

# Fit and transform the data to calculate Z-scores
finalx = pd.DataFrame(scaler.fit_transform(finalx), columns=finalx.columns)

print(finalx)




forcorr=pd.concat([finalx,y_true_label],axis=1)
print(forcorr)


corr_matrix=forcorr.corr(method="pearson")
print(corr_matrix["K"])


finalx.columns=[f'x{i}' for i in range(finalx.shape[1])]
finalx.columns

import itertools as it
total_pairs=list(it.combinations(finalx.columns,2))
total_pairs

from sklearn.tree import DecisionTreeClassifier as dt

d=dt(max_depth=20)

print(y_true_label)

allpredictions=[]

for pair in total_pairs :
  X=finalx[list(pair)]
  d.fit(X,y_true_label)
  predict=d.predict(X)
  print(predict)
  allpredictions.append(predict)

 
new=pd.DataFrame(allpredictions).T
print(new)
