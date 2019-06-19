import numpy as np
import pandas as pd

import models

from constants import *


# Some globle variables for development
TEST = False

# Lets load the data
df = pd.read_excel('../data/NationalBodyProjectTurk.xlsx')

for key in data_sets.keys():
	print('We are running on this dataset: ' + key)


	#### Specifically for man
	X_cat = df[cat_demo]
	man_index = X_cat['Sex']==1.
	X_cat = X_cat.loc[man_index, :].astype('object')
	X_cat_dummies = pd.get_dummies(X_cat, drop_first=True)
	X_num = df.loc[man_index, num_demo]
	X_survey = df.loc[man_index, data_sets[key]]
	ys = df.loc[man_index, y_variables]

	dummies = list(X_cat_dummies.columns)

	X = pd.concat([X_cat_dummies, X_num, X_survey], axis=1)
	X = X.reset_index(drop=True)
	ys = ys.reset_index(drop=True)
	#### Specifically for man

	if TEST == True:
		n = X.shape[0]
		test_size = np.floor(n*0.1)
		X = X.loc[0:test_size, :]
		ys = ys.loc[0:test_size, :]

	all_scores = {}
	for dependent_variable in y_variables:
		print('Running variables: ' + dependent_variable)
		y = ys[dependent_variable].values

		print('Running Linear Regression')
		# Linear Regression
		lr_cv_train_scores, lr_cv_test_scores = models.lr_model(X, y, dummies, num_demo+data_sets[key])
		
		all_scores['lr_train_'+dependent_variable] = lr_cv_train_scores
		all_scores['lr_test_'+dependent_variable] = lr_cv_test_scores

		print('Running Random Forest')
		#Random Forest

		if TEST == True:
			param_grid = {
			    'max_depth': [15],
			    'min_samples_leaf': [20],
			    'min_samples_split': [20],
			    'n_estimators': [500]
			}
		else: 
			param_grid = {
			    'max_depth': [5, 8, 15],
			    'min_samples_leaf': [5, 10, 20],
			    'min_samples_split': [5, 10, 20],
			    'n_estimators': [100, 300, 500]
			}

		rf_cv_train_scores, rf_cv_test_scores, rf_best_params = models.rf_model(X, y, dummies, num_demo+data_sets[key], param_grid)

		all_scores['rf_train_'+dependent_variable] = rf_cv_train_scores
		all_scores['rf_test_'+dependent_variable] = rf_cv_test_scores
		all_scores['rf_best_params_'+dependent_variable] = rf_best_params

	output = pd.DataFrame(all_scores)
	output.to_csv('../outputs/' + 'man_' + key + '.csv')












