import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

#%% Clean & Visualise Data

df = pd.read_csv("kc_house_data.csv")

# understand data
print(df.iloc[:5,:5])
print(df.shape)
print(df.columns)
print(df.isna().sum())
print(df.dtypes)

for i in df.columns:
    print(df[i].head(5))

# plot raw data 
for col in df.columns:
    if col != 'date':
        plt.figure(figsize=(10,6))
        plt.scatter(df[col], df['price'])
        plt.ylabel('Price')
        plt.xlabel(col)
        plt.title(f'Price vs {col}')
        plt.show()
        plt.figure(figsize=(10,6))
        plt.boxplot(df[col])
        plt.title(f'{col} Boxplot')
        plt.show()

# remove outliers
cols_to_skip = ['id', 'date', 'waterfront', 'view', 'yr_renovated','price']
for col in df.columns:
    if col in cols_to_skip:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3*IQR
    upper = Q3 + 3*IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# plot cleaned data 
for col in df.columns:
    if col != 'date':
        plt.figure(figsize=(10,6))
        plt.scatter(df[col], df['price'])
        plt.ylabel('Price')
        plt.xlabel(col)
        plt.title(f'Price vs {col}')
        plt.show()
        plt.figure(figsize=(10,6))
        plt.boxplot(df[col])
        plt.title(f'{col} Boxplot')
        plt.show()

#%% Feature Selection & Engineering

X = df
X['lat_long'] = X['lat'] * X['long']
X['bath_per_bed'] = X['bathrooms'] / (X['bedrooms']+1)
X['lot_ratio_15'] = X['sqft_lot'] / (X['sqft_lot15'] + 1)
X['renovation'] = (X['yr_renovated'] > 0).astype(int)
X['zipcode1'] = (X['zipcode'] > 98150).astype(int)
X['zipcode2'] = ((X['zipcode'] > 98100) & (X['zipcode'] <= 98150)).astype(int)
X['zipcode3'] = ((X['zipcode'] > 98050) & (X['zipcode'] <= 98100)).astype(int)
X['zipcode4'] = ((X['zipcode'] > 98000) & (X['zipcode'] <= 98050)).astype(int)

y = X['price']
X = X.drop(['id','date','price','yr_built','yr_renovated','zipcode'], axis=1)

#% Split & Scale Data    

# data split
X_train, X_, y_train, y_ = train_test_split(X, y, test_size = 0.4, random_state=42) # train with 60% of data
X_cv, X_test,y_cv, y_test = train_test_split(X_, y_, test_size = 0.5, random_state=42) # cross validate with 20% of data and test with 20% of data

# data scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.transform(X_cv)

#%% Linear Regression

degree = [1,2,3]
train_mse_list = []
cv_mse_list = []

for deg in degree:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_cv_poly = poly.transform(X_cv_scaled)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    yhat_train = model.predict(X_train_poly)
    yhat_cv = model.predict(X_cv_poly)
    
    MSE_train = mean_squared_error(y_train, yhat_train) / 2
    train_mse_list.append(MSE_train)
    MSE_cv = mean_squared_error(y_cv, yhat_cv) / 2
    cv_mse_list.append(MSE_cv)
    
    print(f'For degree {deg}, Train MSE is {MSE_train}')
    print(f'For degree {deg}, Cross Validation MSE is {MSE_cv}')

plt.figure(figsize=(10,6))
plt.plot(degree, train_mse_list, color='red', label='Train MSE', marker='o')
plt.plot(degree, cv_mse_list, color='blue', label='Cross Validation MSE', marker='x')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training MSE VS Cross Validation MSE')
plt.show()

best_degree = degree[np.argmin(cv_mse_list)]
print(f'Best Degree is {best_degree}')


#%% Ridge Regression

poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_cv_poly = poly.transform(X_cv_scaled)

lambda_range = np.logspace(-5, 4, 10)
ridge_train_mse = []
ridge_cv_mse = []

for i in lambda_range:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train_poly,y_train)
    
    yhat_train = ridge.predict(X_train_poly)
    yhat_cv = ridge.predict(X_cv_poly)
    
    MSE_train = mean_squared_error(y_train, yhat_train) / 2
    ridge_train_mse.append(MSE_train)
    MSE_cv = mean_squared_error(y_cv, yhat_cv) / 2
    ridge_cv_mse.append(MSE_cv)
    print(f' For lambda: {i:.5f}, Train MSE is {MSE_train:.2f}')
    print(f'For lambda: {i:.5f}, Cross Validation MSE: {MSE_cv:.2f}')

plt.figure(figsize=(10,6))
plt.plot(lambda_range, ridge_train_mse, color='red', label='Train MSE', marker='o')
plt.plot(lambda_range, ridge_cv_mse, color='blue', label='Cross Vaolidation MSE', marker='x')
plt.legend()
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('Training MSE VS Cross Validation MSE')
plt.show()

best_alpha = lambda_range[np.argmin(ridge_cv_mse)]
print(f'Best alpha is {best_alpha}')


#%% Test Chosen Model

X_test_scaled = scaler.transform(X_test)
poly = PolynomialFeatures(best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = Ridge(alpha=best_alpha)
model.fit(X_train_poly, y_train)

yhat_test = model.predict(X_test_poly)
MSE_test = mean_squared_error(y_test, yhat_test)  / 2
error = round(np.sqrt(MSE_test),2)

ss_res = np.sum((y_test - yhat_test) ** 2) # Residual sum of squares
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2) # Total sum of squares
r2 = 1 - (ss_res / ss_tot)

print(f'Test MSE: {MSE_test}')
print(f'Model predictions varies ${error} from true house value')
print(f'R^2 is {round(r2*100,2)}%')

#%% Test Results

# Test MSE: 8925154692.459467
# Model predictions varies $94473.04 from true house value
# R^2 with test set is 81.13%
# Highest house price $3650000
# Lowest house price $78000

#%%








