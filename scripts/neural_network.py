import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# Some globle variables for development
TEST = False

# Lets load the data
df = pd.read_excel('data/NationalBodyProjectTurk.xlsx')

# Useful categories
cat_demo = ['Sex', 'SexualOrientationX4', 'EthnicityX5', 'RelationshipStatus','CollegeStudent']
num_demo = ['BMI', 'Age', 'LevelEducation']
cont_avg = ['AETOTAL', 'SATAQThinInternalization', 'SATAQMuscleInternalization', 'SATAQFamilyPressure', 
            'SATAQPeerPressure', 'SATAQMediaPressure', 'FaceSatisfactionTotal', 'OverweightPreoccupationTotal', 
            'BIQLITotal', 'SURVEILLANCETotal']
survey_data_aggregate = ['SATAQThinInternalization', 'SATAQMuscleInternalization', 'SATAQFamilyPressure', 
            'SATAQPeerPressure', 'SATAQMediaPressure', 'FaceSatisfactionTotal', 'SURVEILLANCETotal']

survey_data_raw = ['SATAQThinInternalization', 'SATAQMuscleInternalization',
       'SATAQFamilyPressure', 'SATAQPeerPressure', 'SATAQMediaPressure',
       'ThinSATAQ1BodyThin', 'ThinSATAQ2ThinkThin', 'ThinSATAQ3BodyLean',
       'ThinSATAQ4LittleFat', 'MuscleSATAQ1AthleticImportant',
       'MuscleSATAQ2Muscular', 'MuscleSATAQ3AthleticThings',
       'MuscleSATAQ4ThinkAthletic', 'MuscleSATAQ5MuscularThings',
       'FamilySATAQ1PressureThin', 'FamilySATAQ2PressureAppearance',
       'FamilySATAQ3DecreaseBodyFat', 'FamilySATAQ4BetterShape',
       'PeersSATAQ1Thinner', 'PeersSATAQ2ImproveAppearance',
       'PeersSATAQ3BetterShape', 'PeersSATAQ4DecreaseBodyFat',
       'MediaSATAQ1BetterShape', 'MediaSATAQ2Thinner',
       'MediaSATAQ3ImproveAppearance', 'MediaSATAQ4decreaseBodyFat',
       'FaceSatisfaction1HappyFace', 'FaceSatisfaction2HappyNose',
       'FaceSatisfaction3HappyEyes', 'FaceSatisfaction4HappyShape',
       'Surveillance1ThinkAboutLooksRECODED',
       'Surveillance2ComfortableClothesRECODED',
       'Surveillance3BodyFeelsOverLooksRECODED',
       'Surveillance4CompareLooksRECODED', 'Surveillance5LooksDuringDay',
       'Surveillance6WorryClothes', 'Surveillance7LooksToOtherPeopleRECODED',
       'Surveillance8BodyDoesBodyLooksRECODED']

y_variables = ['AETOTAL', 'OverweightPreoccupationTotal', 
               'OverweightPreoccupation3Diet', 'OverweightPreoccupation4TriedFasting',
               'BIQLITotal']

# interation setting
data_sets = {'aggregate': survey_data_aggregate, 'raw': survey_data_raw}

# create the folds

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


def nn_model(X, y, param_grid={1: 12, 2: 8, 3: 1}):
	
	nn_cv_test_scores = []
	for train, test in kf.split(X):
		X_train = X.loc[train,:]
		y_train = y[train]
		X_test = X.loc[test, :]
		y_test = y[test]

		print('Running Variables: ' + dependent_variable)

		X_train, X_test = preprocess(X_train, X_test, dummies, num_demo+data_sets[key])

		model = Sequential()
		model.add(Dense(param_grid[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
		model.add(Dense(param_grid[2], activation='relu'))
		model.add(Dense(param_grid[3], activation='linear'))
		model.summary()
		model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
		history = model.fit(X_train, y_train, epochs=1, batch_size=50,  verbose=1, validation_split=0.2)

		pred = model.predict(X_test).reshape(-1)
		nn_cv_test_scores.append(score(pred, y_test, X_train))
	return nn_cv_test_scores

## implementation

for key in data_sets.keys():
	print('We are running on this dataset: ' + key)

	# preprocess = make_column_transformer((num_demo+data_sets[key], StandardScaler()),
	# 								(cat_demo, OneHotEncoder(categories='auto', drop='first')))

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
		X, y = shuffle(X, y, random_state=0)

		param_grid={1: 12, 2: 8, 3: 1}

		nn_cv_test_scores = nn_model(X, y, param_grid)

		all_scores['nn_1'] = nn_cv_test_scores
		#all_scores['nn_1_params'] = param_grid

		param_grid={1: 10, 2: 5, 3: 1}

		nn_cv_test_scores = nn_model(X, y, param_grid)

		all_scores['nn_2'] = nn_cv_test_scores
		#all_scores['nn_2_params'] = param_grid

		param_grid={1: 5, 2: 5, 3: 1}

		nn_cv_test_scores = nn_model(X, y, param_grid)

		all_scores['nn_3'] = nn_cv_test_scores
		#all_scores['nn_3_params'] = param_grid

	output = pd.DataFrame(all_scores)
	output.to_csv('outputs/' + 'nn_' + key + '.csv')


