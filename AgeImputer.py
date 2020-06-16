from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import pandas as pd


class AgeImputer(BaseEstimator, TransformerMixin):
    """
    Class used for imputing missing values in the Age column
    in the df_titanic dataset.

    Args:
        group_cols : list
            List of columns used for calculating the aggregated value
        target : str
            The name of the column to impute

    Returns:
    -------
        X : array-like
            The array with imputed values in the target column
    """

    def __init__(self):
        self.target = 'title'
        self.title_median_age = dict()

    def fit(self, X, y=None):
        for title in X['title'].unique():
            self.title_median_age[title] = X.loc[:][X['title'] == title]['Age'].median()

        self.impute_map_ = self.title_median_age

        return self

    def transform(self, X, y=None):
        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map_')

        # Seperate df in with and without age
        X_o_age = X[X['Age'].isna()].copy()
        X_w_age = X[~X['Age'].isna()].copy()

        # Interpolate Age
        X_o_age['Age'] = X_o_age['title'].map(self.impute_map_)

        # Add df and reorder according to index
        X_ = pd.concat([X_w_age, X_o_age])
        X = X_.loc[X_.index]

        return X[['Age']].values

