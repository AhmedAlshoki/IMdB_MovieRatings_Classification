import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

movies_dataset = pd.read_csv('Movies_training_classification.csv')


def replace_drop():
    global movies_dataset

    movies_dataset['Age'].replace('all', '1', inplace=True)

    movies_dataset['Language'] = movies_dataset['Language'].str.replace('None,', "")
    movies_dataset['Age'] = pd.to_numeric(movies_dataset['Age'].str.replace('+', ""))
    movies_dataset['Rotten Tomatoes'] = pd.to_numeric(movies_dataset['Rotten Tomatoes'].str.replace('%', ""))

    movies_dataset['rate'] = movies_dataset['rate'].str.replace('High','2')
    movies_dataset['rate'] = movies_dataset['rate'].str.replace('Intermediate','1')
    movies_dataset['rate'] = pd.to_numeric(movies_dataset['rate'].str.replace('Low','0'))

    # drop Rows without  rate
    movies_dataset = movies_dataset[movies_dataset['rate'].notnull()]

    movies_dataset = movies_dataset.reset_index(drop=True)  # renumbering indices again correctly

    # drop (Title, Type)
    movies_dataset.drop(['Title', 'Type'], axis=1, inplace=True)

    # drop columns that have 45% or more nulls
    num_of_rows = movies_dataset.shape[0]  # gives number of row count
    rows_threshold = int((num_of_rows * 55) / 100)  # get 55% of number of rows
    # keep only rows with at least 55% of non-NA values:
    movies_dataset = movies_dataset.dropna(thresh=rows_threshold, axis=1)
    print(movies_dataset)


def handle_missing_data(self):
    self['Language'] = self['Language'].str.replace('None', self['Language'].mode()[0])

    for feature in self:
        # fill Categorical with 'mode' & Numerical with 'mean'
        if self[feature].dtype == np.dtype('O'):  # O is for dtype Object (string)
            self[feature].fillna((self[feature].mode()[0]), inplace=True)
        else:
            self[feature].fillna((self[feature].mean()), inplace=True)

    return self


def feature_encoding(dataset_copy):
    # getting names of columns that have categorical data only
    feature_cols = list(dataset_copy.select_dtypes(include='O').columns)

    # apply one hot encoding on the feature with minimum unique values
    min_unique_feature = (dataset_copy[feature_cols].nunique(axis=0)).idxmin()
    dataset_copy = pd.concat(
        [dataset_copy.drop(min_unique_feature, 1), dataset_copy[min_unique_feature].str.get_dummies(sep=",")], 1)

    feature_cols.remove(min_unique_feature)

    # explode: separating columns with multiple values into multiple rows so that each row has one director, etc.
    # that makes each row differ from the other in a feature, that will also help in label encoding
    for feature in feature_cols:
        dataset_copy[feature] = dataset_copy[feature].str.split(',').values.tolist()
        dataset_copy = dataset_copy.explode(feature).reset_index(drop=True)
        label_encoder = LabelEncoder()
        label_encoder.fit(list(dataset_copy[feature].values))
        dataset_copy[feature] = label_encoder.transform(list(dataset_copy[feature].values))

    return dataset_copy


def correlate_data(self):
    movies_corr = self.corr()
    # Top features with correlation 5%
    top_feature_columns = movies_corr.index[abs(movies_corr['rate'] > 0.01)]
    top_corr = self[top_feature_columns].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    top_feature_columns = top_feature_columns[:-1]
    return top_feature_columns


def feature_scaling(X):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))

    for c in X:
        feature_arr = np.array(X[c]).reshape(-1, 1)
        X[c] = min_max_scaler.fit_transform(feature_arr)  # MinMax Scale
    return X


def Preprocess():
    replace_drop()
    print("Replacing and dropping: Done!")

    dataset_copy = movies_dataset

    dataset_copy = handle_missing_data(dataset_copy)
    print("Handling missing data: Done!")

    dataset_copy = feature_encoding(dataset_copy)
    print("Encoding: Done!")

    Y = dataset_copy['rate']  # target after changing rows
    X = dataset_copy.drop(['rate'], axis=1, inplace=False)  # extract features only
    dataset_copy = pd.concat([X, dataset_copy['rate']], 1)  # moving IMDb to be the last column

    top_features_columns = correlate_data(dataset_copy)
    print("Correlation: Done!")

    X = X[top_features_columns]
    X = feature_scaling(X)
    print("Feature_scaling: Done!")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

    return X_train, X_test, y_train, y_test
