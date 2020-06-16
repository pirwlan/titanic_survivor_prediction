from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

import feature_engineering as fe
import helper as h


def data_preparation(train, test):
    df_train = h.read_titanic_data(train)
    df_train['title'] = h.encode_title(df_train)

    df_test = h.read_titanic_data(test)
    df_test['title'] = h.encode_title(df_test)
    return df_train, df_test


def titanic():

    df_train, df_test = data_preparation('train.csv', 'predict.csv')
    y_train = df_train[['Survived']]

    classification_pipe = fe.clf_pipe()

    param_grid = fe.grid_parameters()

    grid_search = GridSearchCV(estimator=classification_pipe,
                               param_grid=param_grid,
                               scoring= {'acc': make_scorer(accuracy_score)},
                               return_train_score=True,
                               cv=10,
                               refit='acc',
                               n_jobs=-1)

    best_model_ranfor = grid_search.fit(df_train, y_train.values.ravel())

    h.evaluate_summary(best_model_ranfor)
    h.create_submission(best_model_ranfor, df_train, df_test)


if __name__ == '__main__':

    titanic()
