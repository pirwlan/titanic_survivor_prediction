

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


def get_feature_names_from_ColumnTransformer(column_transformer):
    """
    Gets feature names from Columntransformer

    Args:
        column_transformer: sklean.ColTransformer:
            fitted Column Transformer

    Returns:
        col_names: list
            list of feature names

    """
    col_name = []
    for idx, feature_pipe in enumerate(
            column_transformer.transformers_[:-1]):  # the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = feature_pipe[2]
        transformer_type = column_transformer.transformers_[:-1][idx][1].steps[-1][0].split('_')[1]

        if transformer_type == 'diskretizer':

            bin_edges = feature_pipe[1].steps[-1][-1].bin_edges_[0]
            for idx, bin_edge in enumerate(bin_edges):
                if idx == len(bin_edges) - 1:
                    break
                else:
                    col_name.append(f'{raw_col_name[0]}-{bin_edge:.2f}_{bin_edges[idx + 1]:.2f}')

        elif transformer_type == 'onehot':
            feature_names = (feature_pipe[1][-1].get_feature_names())
            col_name.extend(feature_names)

    return col_name


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
        os.makedir(path)

    except FileExistsError:
        pass



def create_submission(best_model, df_train, df_test):

    submission_path = os.path.join(os.getcwd(), 'submission')
    verify_folder_existence(submission_path)

    df_test['Survived'] = len(df_test) * ['target']
    df_test = df_test[df_train.columns]

    y_ranfor = best_model.predict(df_test)

    df_submit = pd.DataFrame(y_ranfor, columns=['Survived'], index=df_test.index)
    df_submit.to_csv(os.path.join(submission_path, 'submission.csv'))
