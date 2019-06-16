import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, Cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle