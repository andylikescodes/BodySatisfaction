import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

kf = KFold(n_splits=10, random_state=0)

def score(y, y_pred, X):
    SS_Residual = sum((y-y_pred)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return adjusted_r_squared

def preprocess(X_train, X_test, dummy_cats, num_cats):

	scaler = StandardScaler()

	X_train_cats = X_train.loc[:, dummy_cats].values
	X_train_num = X_train.loc[:, num_cats].values
	X_test_cats = X_test.loc[:, dummy_cats].values
	X_test_num = X_test.loc[:, num_cats].values

	scaler.fit(X_train_num)
	X_train_num = scaler.transform(X_train_num)
	X_train_cv = np.hstack([X_train_cats, X_train_num])
	X_test_num = scaler.transform(X_test_num)
	X_test_cv = np.hstack([X_test_cats, X_test_num])

	return X_train_cv, X_test_cv


def lr_model(X, y, dummies, num, param_grid=None):
	X, y = shuffle(X, y)
	X = X.reset_index(drop=True)
	# Linear Regression
	lr_cv_test_scores = []
	lr_cv_train_scores = []
	lr_cv_test_scores = []
	for train, test in kf.split(X):
		X_train = X.loc[train,:]
		y_train = y[train]
		X_test = X.loc[test,:]
		y_test = y[test]

		X_train, X_test = preprocess(X_train, X_test, dummies, num)
		lr = LinearRegression()
		lr.fit(X_train, y_train)
		pred = lr.predict(X_test)
		lr_cv_test_scores.append(score(y_test, pred, X_train))

		pred = lr.predict(X_train)
		lr_cv_train_scores.append(score(y_train, pred, X_train))
	return lr_cv_train_scores, lr_cv_test_scores

def rf_model(X, y, dummies, num, param_grid):
	X, y = shuffle(X, y)
	X = X.reset_index(drop=True)

	rf_cv_test_scores = []
	rf_cv_train_scores = []
	rf_best_params = []
	for train, test in kf.split(X):
		X_train = X.loc[train,:]
		y_train = y[train]
		X_test = X.loc[test, :]
		y_test = y[test]

		X_train, X_test = preprocess(X_train, X_test, dummies, num)

		# make this scorer for the scikit-learn cv function
		adjusted_rsquared_scorer = make_scorer(score, X=X_train)

		rf_cv = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, 
							scoring=adjusted_rsquared_scorer, cv=10, n_jobs=-1, verbose=1)
		rf_cv.fit(X_train, y_train)
		rf_best_params.append(rf_cv.best_params_)
		pred = rf_cv.predict(X_test)
		rf_cv_test_scores.append(score(y_test, pred, X_train))
		pred = rf_cv.predict(X_train)
		rf_cv_train_scores.append(score(y_train, pred, X_train))

	return rf_cv_train_scores, rf_cv_test_scores, rf_best_params


# def nn_model(X, y, dummies, num, param_grid={1: 12, 2: 8, 3: 1}):
# 	X, y = shuffle(X, y)
# 	X = X.reset_index(drop=True)

# 	nn_cv_test_scores = []
# 	nn_cv_train_scores = []
# 	for train, test in kf.split(X):
# 		X_train = X.loc[train,:]
# 		y_train = y[train]
# 		X_test = X.loc[test, :]
# 		y_test = y[test]

# 		X_train, X_test = preprocess(X_train, X_test, dummies, num)

# 		model = Sequential()
# 		model.add(Dense(param_grid[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
# 		model.add(Dense(param_grid[2], activation='relu'))
# 		model.add(Dense(param_grid[3], activation='linear'))
# 		model.summary()
# 		model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
# 		history = model.fit(X_train, y_train, epochs=100, batch_size=50,  verbose=1, validation_split=0.2)

# 		pred = model.predict(X_test).reshape(-1)
# 		nn_cv_test_scores.append(score(y_test, pred, X_train))

# 		pred = model.predict(X_train).reshape(-1)
# 		nn_cv_train_scores.append(score(y_train, pred, X_train))
# 	return nn_cv_train_scores, nn_cv_test_scores