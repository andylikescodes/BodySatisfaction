import numpy as np
import pandas as pd

import models

from constants import *


# Some globle variables for development
TEST = False

# Lets load the data
df = pd.read_excel('../data/NationalBodyProjectTurk.xlsx')

## implementation

for key in data_sets.keys():
	print('We are running on this dataset: ' + key)

	X_cat = df[cat_demo].astype('object')
	X_cat_dummies = pd.get_dummies(X_cat, drop_first=True)
	X_num = df[num_demo]
	X_survey = df[data_sets[key]]
	ys = df[y_variables]

	dummies = list(X_cat_dummies.columns)

	X = pd.concat([X_cat_dummies, X_num, X_survey], axis=1)

	if TEST == True:
		n = X.shape[0]
		test_size = np.floor(n*0.1)
		X = X.loc[0:test_size, :]
		ys = ys.loc[0:test_size, :]

	all_scores = {}
	for dependent_variable in y_variables:
		y = ys[dependent_variable].values

		param_grid={1: 12, 2: 8, 3: 1}

		nn_cv_train_scores, nn_cv_test_scores = models.nn_model(X, y, dummies, num_demo+data_sets[key], param_grid)

		all_scores['nn_1_train_'+dependent_variable] = nn_cv_train_scores
		all_scores['nn_1_test_'+dependent_variable] = nn_cv_test_scores
		#all_scores['nn_1_params_'+dependent_variable] = param_grid

		param_grid={1: 10, 2: 5, 3: 1}

		nn_cv_train_scores, nn_cv_test_scores = models.nn_model(X, y, dummies, num_demo+data_sets[key], param_grid)

		all_scores['nn_2_train_'+dependent_variable] = nn_cv_train_scores
		all_scores['nn_2_test_'+dependent_variable] = nn_cv_test_scores
		# #all_scores['nn_2_params_'+dependent_variable] = param_grid

		param_grid={1: 5, 2: 5, 3: 1}

		nn_cv_train_scores, nn_cv_test_scores = models.nn_model(X, y, dummies, num_demo+data_sets[key], param_grid)

		all_scores['nn_3_train_'+dependent_variable] = nn_cv_train_scores
		all_scores['nn_3_test_'+dependent_variable] = nn_cv_test_scores
		#all_scores['nn_3_params_'+dependent_variable] = param_grid

	output = pd.DataFrame(all_scores)
	output.to_csv('../outputs/' + 'nn_' + key + '.csv')


