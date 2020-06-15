from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import helper as h
import numpy as np
import os
import pandas as pd

import feature_engineering as fe


def data_preparation(param, param1):
    df_train = h.read_titanic_data('train.csv')
    df_train['title'] = h.encode_title(df_train)

    df_test = h.read_titanic_data('predict.csv')
    df_test['title']  = h.encode_title(df_test)
    return df_train, df_test


def titanic():

    df_train, df_test = data_preparation('train.csv', 'predict.csv')
    y_train = df_train[['Survived']]

    classification_pipe = fe.clf_pipe()

    param_grid = fe.grid_parameters()
    param_grid = {'feature_engineering__age_pipe__age_diskretizer__n_bins': [6],
                  'ran_for_clf__max_depth': [None],
                  'ran_for_clf__max_features': ['sqrt'],
                  'ran_for_clf__min_samples_leaf': [2],
                  'ran_for_clf__min_samples_split': [10],
                  'ran_for_clf__n_estimators': [125]}


    grid_search = GridSearchCV(
        estimator=classification_pipe,
        param_grid=param_grid,
        scoring= {'acc': make_scorer(accuracy_score)},
        return_train_score=True,
        cv=10,
        refit='acc',
        n_jobs=-1
    )

    best_model_ranfor = grid_search.fit(df_train, y_train)

    h.evaluate_summary(best_model_ranfor)

    h.create_submission(best_model_ranfor, df_train, df_test)


if __name__ == '__main__':

    titanic()
