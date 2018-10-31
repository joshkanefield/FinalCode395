
# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load the Ladybug data
'''In 1983 an article was published about ladybird beetles and their 
behavior changes under different temperature conditions (N. H. Copp. Animal Behavior, 31,:424-430). 
An experiment was run to see how many beetles stayed in light as temperature changed.
'''

# Read the CSV file into a DataFrame: df
df = pd.read_csv("LadyBugs.csv")

# Create arrays for features (Lighted) and target variable (Temp)
y = df[['Lighted']]
X = df[['Temp']]

# # Linear Regression 

# Create training and test sets with 0.3 test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Create linear regression object call ols
ols = LinearRegression()
  
# Train the model using the training sets

ols.fit(X_train, y_train)

# Report the coefficient 

ols.coef_


# ## Plot outputs

# This plots the predicted ols fitted line
min = X_test.min()
max = X_test.max() 
predictor_space = pd.DataFrame(np.arange(min, max,  0.05))
plt.plot(predictor_space, ols.predict(predictor_space), color='blue', linewidth=3)

# Scatter plot the actual test data 

plt.scatter(X_test, y_test,  color='black')


# ## Predict on the Test data

# Get the predicted y_pred using the test data
y_pred = ols.predict(X_test)

# Compute and print the R^2 and RMSE
print("R^2: {}".format(ols.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# # Polynomial Regression

# Reload the CSV file into a DataFrame: df
df = pd.read_csv("LadyBugs.csv")

# Create arrays for features (Lighted) and target variable (Temp)

yp = df[['Lighted']]
Xp = df[['Temp']]

# Add in 15-degree polynomial of the X variables
poly = PolynomialFeatures(degree = 15)
Xp = pd.DataFrame(poly.fit_transform(Xp))
print("Dimensions of X after reshaping: {}".format(Xp.shape))

# Create training and test sets with .3 test size

Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size = 0.3)
print(Xp_test)
print(yp_test)


# Create linear regression object, ols2

ols2 = LinearRegression()
  
# Fit the model using the training sets 


ols2.fit(Xp_train, yp_train)
  
# Report the coefficients   

ols2.coef_


# ## Plot outputs

# This plots the predicted ols fitted line
predictor_space = pd.DataFrame(np.arange(min, max, 0.05)) # Creates prediction space on x interval
predictor_poly = pd.DataFrame(poly.fit_transform(predictor_space)) # Creates data to predict on
plt.plot(predictor_space, ols2.predict(predictor_poly), color='blue') # Plot fitted model 

# Scatter plot the actual test data

plt.scatter(Xp_test.iloc[ : ,[1]], yp_test,  color='black')



# Predict on the test data: y_pred


yp_pred =  ols2.predict(Xp_test)

# Computer and print R^2 and RMSE
# your code here

print("R^2: {}".format(ols2.score(Xp_test, yp_test)))
rmse = np.sqrt(mean_squared_error(yp_test, yp_pred))
print("Root Mean Squared Error: {}".format(rmse))

# # Ridge Regression, Part III

# Read the CSV file into a DataFrame: df

df = pd.read_csv("LadyBugs.csv")

# Create arrays for features (Lighted) and target variable (Temp)

yr = df[['Lighted']]
Xr = df[['Temp']]

# Add in 15-degree polynomial of the X variables

Xr = pd.DataFrame(PolynomialFeatures(degree = 15).fit_transform(Xr))

  
# Create training and test sets

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.3)

# Create a ridge regressor object called ridge with lambda = 0.1
ridge = Ridge(normalize = True, alpha = 0.1)

# Train the ridge model using the training sets (the polynomial factors are in the data)

ridge.fit(Xr_train, yr_train)

# The coefficients
ridge.coef_


# ## Plot outputs

# This plots the predicted ols fitted line

predictor_space = pd.DataFrame(np.arange(min, max, 0.05)) # Creates prediction space on x interval
predictor_poly = pd.DataFrame(poly.fit_transform(predictor_space)) # Creates data to predict on
plt.plot(predictor_space, ridge.predict(predictor_poly), color='blue') # Plot fitted model 

# Scatter plot the actual test data


plt.scatter(Xr_test.iloc[ : ,[1]], yr_test,  color='black')


#predict
yr_pred = ridge.predict(Xr_test)

#Find R^2 and RMSE

print("R^2: {}".format(ridge.score(Xr_test, yr_test)))
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
print("Root Mean Squared Error: {}".format(rmse))

# # Hyper Tune Lambda, K-fold Grid Search, Ridge Regression, Part IV


# Read the CSV file into a DataFrame: df


df = pd.read_csv("LadyBugs.csv")

# Create arrays for features (Lighted) and target variable (Temp)


y = df[['Lighted']]
X = df[['Temp']]


# Add in 15-degree polynomial of the X variables


X = pd.DataFrame(PolynomialFeatures(degree = 15).fit_transform(X))

# Create training and test sets with 0.3 hold out for test data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Setup a grid of lambdas (aka alphas) called param_grid of 20 lambdas from .001 to 2 
param_grid = {'alpha': np.linspace(0.001, 2.0, num = 20)}

# Create a ridge regressor object called ridge


ridge = Ridge(normalize = True, alpha = 0.1)

# Setup the GridSearchCV object called grid_ridge for 5 folds using the param_grid above and ridge object
grid_ridge = GridSearchCV(ridge, param_grid, cv = 5)

# Train the model using the training sets 5 folds for all lambdas!


grid_ridge.fit(X_train, y_train)


#Get the best lambda
best = grid_ridge.best_params_

# Create a final ridge regressor object called ridge_final using the best lambda from hypertuning
ridge_final = Ridge(alpha = best['alpha'], normalize = True)

# Now fit this model on the test data 


ridge_final.fit(X_train, y_train)


# ## Plot outputs



# This plots the predicted ols fitted line
predictor_space = pd.DataFrame(np.arange(min, max, 0.05)) # Creates prediction space on x interval
predictor_poly = pd.DataFrame(poly.fit_transform(predictor_space)) # Creates data to predict on
plt.plot(predictor_space, ridge_final.predict(predictor_poly), color='blue') # Plot fitted model 

# Scatter plot the actual test data

plt.scatter(X_test.iloc[ : ,[1]], y_test,  color='black')


# ## Final scores given tuned lambda

# Predict on the test data: y_pred
y_pred = ridge_final.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(ridge_final.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


