import math
from typing import List

import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from playlist.labelling import season_labels, label_heuristics_col


def get_twice_count_samples(data: pd.DataFrame, sort_col: str, count: int):
    sorted_data = data.sort_values(sort_col, ascending=False)
    return pd.concat([sorted_data.head(count), sorted_data.tail(count)], ignore_index=True)


def evaluate_model(X: pd.DataFrame, y: pd.Series, model, model_name: str):
    scores = cross_val_score(model, X, y, cv=5)
    print("Accuracy of %s model: %0.2f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))


def predict_model(X, y, model):
    res = model.predict(X)
    acc_cnt = res == y
    if isinstance(acc_cnt, pd.DataFrame):
        all_true = pd.Series(True, index=acc_cnt.index)
        for col in acc_cnt.columns:
            all_true &= acc_cnt[col]
        acc_cnt = all_true

    correct = acc_cnt[acc_cnt].shape[0]
    print("Percentage correct = %0.2f" % (correct / acc_cnt.shape[0]))


def evaluate_and_predict_model(train_X, train_y, real_X, real_y, model_class, model_name):
    model = model_class.fit(train_X, train_y)
    evaluate_model(train_X, train_y, model, model_name)
    predict_model(real_X, real_y, model)


def evaluate_and_predict_model_by_cols(train: pd.DataFrame, real: pd.DataFrame,
                                       model_cols: List[str], label_cols: List[str]):
    train_x = train.loc[:, model_cols]
    train_y = train.loc[:, label_cols]
    real_x = real.loc[:, model_cols]
    real_y = real.loc[:, label_cols]

    print(f'Evaluating columns {model_cols} for labels {label_cols}')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, MultiOutputClassifier(LogisticRegression()),
                               'Logistic Regression')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, MultiOutputClassifier(SGDClassifier()),
                               'Stochastic Gradient Descent')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, MultiOutputClassifier(SVC()),
                               'Support Vector')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, KNeighborsClassifier(),
                               'K Nearest Neighbours (5 - default)')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, MLPClassifier(),
                               'Multi-layer Perceptron')
    evaluate_and_predict_model(train_x, train_y, real_x, real_y, DecisionTreeClassifier(),
                               'Decision Tree')


def _select_subset(filtered_data: pd.DataFrame, extract_count: int):
    extract_half = math.floor(extract_count / 2)
    temp = filtered_data.sort_values('name')
    top_part = temp.head(extract_half)
    bottom_part = temp.tail(extract_half)
    return pd.concat([top_part, bottom_part], ignore_index=True)


def extract_test_data(data: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    filtered_data = data.loc[:, required_cols].copy()

    count_col = 'season_count'
    filtered_data[count_col] = 0
    for season in season_labels:
        filtered_data.loc[filtered_data[season], count_col] += 1

    max_count = filtered_data.shape[0]

    min_season_data = {
        'one-season': max_count,
        'cluster-labelled': max_count,
    }
    has_one_season = filtered_data[count_col] == 1
    is_clustering_populated = filtered_data[label_heuristics_col] == ''

    for season in season_labels:
        season_count = filtered_data.loc[
            has_one_season & ~is_clustering_populated & filtered_data[season], season
        ].count()
        if season_count < min_season_data['one-season']:
            min_season_data['one-season'] = season_count

        season_count = filtered_data.loc[is_clustering_populated & filtered_data[season], season].count()
        if 0 < season_count < min_season_data['cluster-labelled']:
            min_season_data['cluster-labelled'] = season_count

    for key in min_season_data.keys():
        min_season_data[key] = math.floor(min_season_data[key] / 2)

    result_arr = []
    for season in season_labels:
        one_season_data = filtered_data.loc[has_one_season & ~is_clustering_populated & filtered_data[season], :]
        sampled_one_season = _select_subset(one_season_data, min_season_data['one-season'])
        result_arr.append(sampled_one_season)

        cluster_labelled = filtered_data.loc[is_clustering_populated & filtered_data[season], :]
        if cluster_labelled['name'].count() > 0:
            sampled_cluster = _select_subset(cluster_labelled, min_season_data['cluster-labelled'])
            result_arr.append(sampled_cluster)

    has_multiple_seasons = filtered_data[count_col] > 1
    multi_season_counts = filtered_data.loc[has_multiple_seasons, :]\
        .groupby(season_labels).count()['name'].reset_index()

    min_multi_sample = 10
    min_multi_season = multi_season_counts.loc[multi_season_counts['name'] > min_multi_sample, 'name'].min()
    min_multi_season = math.floor(min_multi_season / 2)
    for _, row in multi_season_counts.iterrows():
        selected_rows = pd.Series(True, index=filtered_data.index)
        for season in season_labels:
            if row[season]:
                selected_rows &= (filtered_data[season] & has_multiple_seasons)
            else:
                selected_rows &= ~(filtered_data[season] & has_multiple_seasons)

        row_count = row['name']
        multi_season_data = filtered_data[selected_rows]
        if row_count <= min_multi_sample:
            result_arr.append(multi_season_data)
        else:
            sampled_multi_season_data = _select_subset(multi_season_data, min_multi_season)
            result_arr.append(sampled_multi_season_data)

    return pd.concat(result_arr, ignore_index=True)
