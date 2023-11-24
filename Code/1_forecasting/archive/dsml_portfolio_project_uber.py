
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_percentage_error


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data into a DataFrame
# df = pd.read_csv('your_data_file.csv')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#own imports
import fhnw_colourmap

"""# **Building linear regression model**

**SGD Regressor**
"""

import os 

# working directory
wd = r"G:\My Drive\FHNW\2023-12-04_Präsentation\1_forecasting\DSMLPortfolioProject\source"
os.chdir(wd)
print("Current Working Directory: ", os.getcwd())

#january data
jdata = pd.read_csv('uber_jdata.csv')

#february data
fdata = pd.read_csv('uber_fdata.csv')

jdata.head(50)

fdata.head()


X = jdata[['PULocationID', 'sdate', 'hour_minute']]
X_test = fdata[['PULocationID', 'sdate', 'hour_minute']]




#%% do data transformation
minmax = MinMaxScaler(feature_range=(-1,1))
minmax.fit(X)
xtrain = minmax.transform(X)
ytrain = jdata['number_of_taxis']
xtest = minmax.transform(X_test)
ytest = fdata['number_of_taxis']

ya = ytrain.to_numpy()
y_test = ytest.to_numpy()

#%%
sgd = SGDRegressor() #njobs=-1
sgd.fit(xtrain, ytrain)
ypred = sgd.predict(xtrain)



m = len(ypred)
rmse = np.sqrt(np.sum((ya-ypred)**2)/m)
print(rmse)

alpha = [100, 10, 1 , 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
train_mse_error_list = []
train_r2_score_list = []
test_mse_error_list = []
test_r2_score_list = []
for a in alpha:
  sgd = SGDRegressor(random_state=0, alpha=a)    # njobs=-1
  sgd.fit(xtrain, ytrain)
  y = sgd.predict(xtrain)
  e = mean_squared_error(ya, y)
  train_mse_error_list.append(e)
  r2 = r2_score(ya, y)
  train_r2_score_list.append(r2)

  yt = sgd.predict(xtest)
  et = mean_squared_error(y_test, yt)
  test_mse_error_list.append(et)
  r2t = r2_score(y_test, yt)
  test_r2_score_list.append(r2t)

print(train_mse_error_list)
print(train_r2_score_list, test_r2_score_list)

plt.figure()
plt.plot(train_mse_error_list, label = 'train')
plt.plot(test_mse_error_list, label = 'test')
plt.legend(loc='upper left')
plt.show()

std = StandardScaler()
std.fit(X)
xtrain_std = std.transform(X)
sgd.fit(xtrain_std, ytrain)
ypred_std = sgd.predict(xtrain_std)

print('Root mean square error - ', np.sqrt(np.sum((ya-ypred_std)**2)/m))

#%%

lr = LinearRegression()
lr.fit(xtrain, ytrain)
ypred_lr = lr.predict(xtrain)

mse = mean_squared_error(ya, ypred_lr)  # np.sqrt(np.sum((ya-ypred_lr)**2)/m)
print('Root mean square error - ', mse**0.5)

#%%

"""**Polynomial regression**"""

train_mse_error = []
train_scores = []
test_mse_error = []
test_scores = []
for d in range(1, 10):
  pipe = make_pipeline(PolynomialFeatures(d), StandardScaler(), SGDRegressor(random_state=0, max_iter= 100))
  pipe.fit(X, ytrain)
  y = pipe.predict(X)
  train_score = pipe.score(X, ytrain)
  test_score = pipe.score(X_test, ytest)
  train_scores.append(train_score)
  test_scores.append(test_score)

train_maxarg = np.argmax(train_scores)
test_maxarg = np.argmax(test_scores)
print('train max score degree = ', train_maxarg+1)
print('test max score degree = ', test_maxarg+1)
print('training max score - ', train_scores[train_maxarg])
print('test max score - ', test_scores[test_maxarg])

plt.figure()
plt.plot(range(1,10), train_scores, label='train')
plt.plot(range(1,10), test_scores, label = 'test')
plt.legend(loc='lower left')
plt.show()

"""Neural Network"""

#from sklearn.neural_network import MLPRegressor

'''regr = MLPRegressor(random_state=1, max_iter=500).fit(xtrain, ytrain)
y_neural = regr.predict(xtrain)
print('score - ', regr.score(xtrain, ytrain))
print('MSE - ', mean_squared_error(ya, y_neural))'''

jdata.head()

X1 = jdata[['PULocationID', 'sdate', 'hour_minute', 'Daytime', 'holiday', 'c1',	'c2',	'c3',	'c4',	'c5']]
ytrain1 = jdata['number_of_taxis']
ya1 = ytrain1.to_numpy()

std.fit(X1)
x_train1 = std.transform(X1)

sgd.fit(x_train1, ytrain1)
ypred1 = sgd.predict(x_train1)

m1 = len(ypred1)
rmse1 = np.sqrt(np.sum((ya1-ypred1)**2)/m1)
print('Training Root mean square error - ', rmse1)
print('Training score - ', r2_score(ya1, ypred1))

mape = np.sum(np.abs((ya1-ypred1)/ya1))/m1
print('Training Mean absolute percentage error - ', mape)
print('Training Mean absolute percentage error from sklearn - ', mean_absolute_percentage_error(ya1, ypred1))

"""Test Data"""

Xtest1 = fdata[['PULocationID', 'sdate', 'hour_minute', 'Daytime', 'holiday', 'c1',	'c2',	'c3',	'c4',	'c5']]
ytest1 = fdata['number_of_taxis']
ya_test1 = ytest1.to_numpy()
xtest1 = std.transform(Xtest1)

ytest_pred1 = sgd.predict(xtest1)

m2 = len(ytest_pred1)
rmse2 = np.sqrt(np.sum((ya_test1-ytest_pred1)**2)/m2)
print('Test Root mean square error - ', rmse2)
mape2 = np.sum(np.abs((ya_test1-ytest_pred1)/ya_test1))/m2
print('Test Mean absolute percentage error - ', mape2)
print('Test Mean absolute percentage error from sklearn - ', mean_absolute_percentage_error(ya_test1, ytest_pred1))
print('Test score - ', r2_score(ya_test1, ytest_pred1))

#%%


#%%

# Load your DataFrame here if needed

# 1. Time Series Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour_minute', y='number_of_taxis', data=df)
plt.title('Number of Taxis Over Different Times of Day')
plt.xlabel('Hour and Minute')
plt.ylabel('Number of Taxis')
plt.show()

#%%

# 2. Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['c1', 'c2', 'c3', 'c4', 'c5']].corr(), annot=True, cmap=fhnw_colourmap.custom_colormap)
plt.title('Correlation Heatmap between Time Intervals')
plt.show()

#%%

# 3. Bar Chart for Taxi Availability on Holidays
plt.figure(figsize=(10, 6))
sns.barplot(x='holiday', y='number_of_taxis', data=df, color= fhnw_colourmap.fhnw_colour)

plt.title('Taxi Availability During Holidays')
plt.xlabel('Holiday')
plt.ylabel('Number of Taxis')
plt.show()

#%%

# 4. Box Plot for Daytime Categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='Daytime', y='number_of_taxis', data=df)
plt.title('Distribution of Number of Taxis Across Daytime Categories')
plt.xlabel('Daytime')
plt.ylabel('Number of Taxis')
plt.show()

# 5. Scatter Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x='sdate', y='number_of_taxis', hue='Daytime', data=df)
plt.title('Scatter Plot of Taxi Availability Over Dates')
plt.xlabel('Date')
plt.ylabel('Number of Taxis')
plt.show()

