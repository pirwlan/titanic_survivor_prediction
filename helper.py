

import os
import pandas as pd


def read_titanic_data(file: str) -> pd.DataFrame:
    """
    reads dataset from data directory, into DataFrame
    Args:
        file: string - Path to file to read

    Returns:
        df: pd.DataFrame - DataFrame containing path
    """

    return pd.read_csv(os.path.join(os.getcwd(), 'data', file), index_col='PassengerId')


def encode_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pd.Series containing the title from the name column.

    Args:
        df: pd.DataFrame -  containing the titanic data

    Returns:
        df_temp: pd.DataFrame - gives back the title column as DataFrame

    """
    df_temp = df['Name'].str.split('.', expand=True)
    df_temp = df_temp[0].str.split(',', expand=True)
    df_temp = pd.DataFrame(df_temp[1])
    return df_temp


def evaluate_summary(model):

    summary = model.cv_results_
    print()
    print('Best parameter:', model.best_params_)
    print('-----------Training-----------')
    print(f'acc_score: {summary["mean_train_acc"][-1]:.2%}')
    print('-------------Test-------------')
    print(f'acc_score: {summary["mean_test_acc"][-1]:.2%}')


def verify_folder_existence(path):
    try:
        os.mkdir(path)

    except FileExistsError:
        pass


def create_submission(best_model, df_train, df_test):
    """
    creates submission file for Kaggle competiton
    """
    submission_path = os.path.join(os.getcwd(), 'submission')
    verify_folder_existence(submission_path)

    df_test['Survived'] = len(df_test) * ['target']
    df_test = df_test[df_train.columns]

    y_ranfor = best_model.predict(df_test)

    df_submit = pd.DataFrame(y_ranfor, columns=['Survived'], index=df_test.index)
    df_submit.to_csv(os.path.join(submission_path, 'submission.csv'))
