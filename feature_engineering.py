from AgeImputer import AgeImputer

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder

import numpy as np
import pandas as pd


def gender_pipeline():
    return Pipeline([('gender_onehot', OneHotEncoder(drop='first', sparse=False))])


def class_pipeline():
    return Pipeline([('class_onehot', OneHotEncoder(drop='first', sparse=False))])


def harbour_pipeline():
    return Pipeline([('harbour_imputer', SimpleImputer(strategy='most_frequent')),
                     ('harbour_onehot', OneHotEncoder(drop='first', sparse=False))])


def age_pipeline():
    return Pipeline([('age_interpolation', AgeImputer()),
                     ('age_imputer', SimpleImputer(strategy='median')),
                     ('age_diskretizer', KBinsDiscretizer(encode='onehot-dense',
                                                          strategy='uniform'))])


def title_pipeline():
    return Pipeline([('title_onehot', OneHotEncoder(handle_unknown='ignore',
                                                    sparse=False))])


def famsize_pipeline():

    def _calc_fam_size(df):
        temp = df['SibSp'] + df['Parch'] + 1
        return pd.DataFrame(temp.map({1: 'alone',
                                      2: 'small',
                                      3: 'small',
                                      4: 'small',
                                      5: 'middle',
                                      6: 'middle',
                                      7: 'large',
                                      8: 'large',
                                      9: 'large',
                                      10: 'large',
                                      11: 'large'}))

    return Pipeline([('family_function', FunctionTransformer(_calc_fam_size)),
                     ('family_onehot', OneHotEncoder(drop='first', sparse=False))])


def fare_pipeline():
    return Pipeline([('fare_imputer', SimpleImputer(strategy='median')),
                     ('fare_diskretizer', KBinsDiscretizer(n_bins=6,
                                                           encode='onehot-dense',
                                                           strategy='uniform'))])


def setup_col_transformer():

    gender_pipe = gender_pipeline()
    class_pipe = class_pipeline()
    harbour_pipe = harbour_pipeline()
    age_pipe = age_pipeline()
    title_pipe = title_pipeline()
    famsize_pipe = famsize_pipeline()
    fare_pipe = fare_pipeline()

    return ColumnTransformer([
        ('gender_pipe', gender_pipe, ['Sex']),
        ('class_pipe', class_pipe, ['Pclass']),
        ('harbour_pipe', harbour_pipe, ['Embarked']),
        ('age_pipe', age_pipe, ['Age', 'title']),
        ('title_pipe', title_pipe, ['title']),
        ('fam_size', famsize_pipe, ['SibSp', 'Parch']),
        ('fare_pipe', fare_pipe, ['Fare'])])


def clf_pipe():
    col_trans = setup_col_transformer()
    ranfor_classifier = RandomForestClassifier()
    return Pipeline([('feature_engineering', col_trans ),
                     ('ran_for_clf', ranfor_classifier)])


def grid_parameters():

    return {'ran_for_clf__n_estimators': [int(x) for x in np.linspace(start=50, stop=250, num=5)],
            'ran_for_clf__max_features': ['auto', 'sqrt'],
            'ran_for_clf__max_depth': [None, 20, 40, 60, 80],
            'ran_for_clf__min_samples_split': [2, 5, 10],
            'ran_for_clf__min_samples_leaf': [1, 2, 4],
            'feature_engineering__age_pipe__age_diskretizer__n_bins': [5, 6, 7, 8, 9, 10, 11, 12]}



