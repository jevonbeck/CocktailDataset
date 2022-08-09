import json
import math
import numpy as np
import pandas as pd
from playlist.data import oz_in_ml
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans

label_heuristics_col = 'all-heuristics'
season_labels = ['spring', 'summer', 'autumn', 'winter']


def find_common_labels(data: pd.DataFrame, unique_id_col: str, a_labels: str, b_labels: str):
    """
    Identifies data items that should be in the same group (confirmed),
    likely groups (likely) or 'uncertain' when comparing the labelling output
    from 2 clustering algorithms.

    :param data: DataFrame with items to be labelled
    :param unique_id_col: Any other column in DataFrame with all values populated
    :param a_labels: Column name with algorithm A clustering labels
    :param b_labels: Column name with algorithm B clustering labels
    :return: Nothing
    """
    b_has_less = len(data[b_labels].unique()) < len(data[a_labels].unique())
    first_group_col, second_group_col = (b_labels, a_labels) if b_has_less else (a_labels, b_labels)

    group_by_cols = [first_group_col, second_group_col]
    groups = data.groupby(group_by_cols).count()[unique_id_col].reset_index()

    second_group = groups.groupby([second_group_col]).count()[unique_id_col].reset_index()
    fully_immersed = second_group.loc[second_group[unique_id_col] == 1, second_group_col].unique()

    confirmed_col = 'confirmed_groups'
    data[confirmed_col] = np.nan
    confirmed_groups = data[second_group_col].isin(fully_immersed)
    data.loc[confirmed_groups, confirmed_col] = data.loc[confirmed_groups, first_group_col]

    group_by_cols = [second_group_col, first_group_col]
    groups = data[~confirmed_groups].groupby(group_by_cols).count()[unique_id_col].reset_index()
    second_group = groups.groupby([second_group_col]).sum()[unique_id_col].reset_index()

    totals_col = 'totals'
    second_group.rename(columns={unique_id_col: totals_col}, inplace=True)
    groups = groups.merge(second_group, on=second_group_col)

    percent_col = 'percentage'
    groups[percent_col] = groups[unique_id_col] / groups[totals_col]

    likely_col = 'likely_groups'
    data[likely_col] = np.nan
    likely_groups = groups.loc[groups[percent_col] > 0.5, [first_group_col, second_group_col]]
    for _, row in likely_groups.iterrows():
        data.loc[data[second_group_col] == row[second_group_col], likely_col] = row[first_group_col]

    data['uncertain_groups'] = data.loc[data[confirmed_col].isna() & data[likely_col].isna(), second_group_col]

    data['uncertain_distribution'] = None
    is_uncertain = data['uncertain_groups'].notna()
    for uncertain_group in data.loc[is_uncertain, second_group_col].unique():
        current_group = data[second_group_col] == uncertain_group
        group_data = data.loc[current_group, first_group_col]
        group_data.index = group_data
        data.loc[current_group, 'uncertain_distribution'] = json.dumps(group_data.groupby(level=0).count().to_dict())


def _add_taste_features(data: pd.DataFrame):
    total_col = 'total-liquid'
    labels = ['citrus', 'fruit', 'herbal', 'milk', 'bubbles', 'wine', 'water', 'dessert']
    data['bubbles-percent'] = data['alco-pop'] + data['soda'] + data['sparkling wine']
    data['citrus-percent'] = data['citrus juice'] + data['citrus spirit']
    data['fruit-percent'] = data['fruit juice'] + data['fruit spirit']
    data['wine-percent'] = data['wine'] + data['fortified wine']
    data['herbal-percent'] = data['herbal spirit'] + data['bitters']
    data['milk-percent'] = data['milk'] + data['cream spirit']
    data['water-percent'] = data['water'] + data['flavoured water']
    data['dessert-percent'] = data['dessert spirit'] + data['tea']

    for label in labels:
        data[label + '-percent'] /= data[total_col]
    data['egg-percent'] = data['egg'] / data[total_col]
    data['syrup-percent'] = data['syrup'] / data[total_col]
    data['beer-percent'] = data['beer'] / data[total_col]
    data['sauce-percent'] = data['sauce'] / data[total_col]

    data['size-percent'] = data[total_col] / data[total_col].max()

    features = data.columns[data.columns.to_series().str.contains('percent')].to_list()

    feature_count_col = 'total-features'
    data[feature_count_col] = 0
    for feature in features:
        has_feature = data[feature] != 0
        data.loc[has_feature, feature_count_col] += 1

    return features


def _add_label_heuristics(data: pd.DataFrame):
    # Alcohol content
    is_strong = data['alco-percent'] >= 0.7
    is_low_alco = data['alco-percent'] <= 0.3
    is_medium_alco = ~is_strong & ~is_low_alco

    # Content / Flavours
    has_fruit_flavour = data['fruit-percent'] > 0
    is_fruity = (data['fruit juice'] + data['fruit spirit'] >= 2 * oz_in_ml)
    is_milky = data['milk-percent'] > 0
    has_citrus = data['citrus-percent'] > 0
    has_light_citrus = (0 < data['citrus-percent']) & (data['citrus-percent'] < 0.4)
    has_medium_citrus = (0.35 <= data['citrus-percent']) & (data['citrus-percent'] < 0.65)
    has_strong_citrus = data['citrus-percent'] >= 0.6
    has_strong_wine = data['fortified wine'] > 0
    is_strong_herbal = data['herbal-percent'] >= 0.5
    has_beer = data['beer'] > 0
    has_syrup = data['syrup'] > 0

    # Texture
    has_light_bubbles = (0 < data['bubbles-percent']) & (data['bubbles-percent'] < 0.5)
    has_heavy_bubbles = data['bubbles-percent'] > 0.5
    is_foamed = (data['shake'] | data['blend']) & (data['egg'] > 0)

    # Size
    is_tall_drink = data['total-liquid'] >= 120

    is_summer = is_low_alco & (is_tall_drink | has_light_bubbles | is_fruity | has_light_citrus)
    is_autumn = is_foamed | has_heavy_bubbles | has_strong_wine | has_beer
    is_winter = (is_strong & ~is_tall_drink) | is_milky | is_strong_herbal
    is_spring = has_strong_citrus | is_fruity | (is_medium_alco & has_fruit_flavour & has_citrus)

    data['spring'] = is_spring
    data['summer'] = is_summer
    data['autumn'] = is_autumn
    data['winter'] = is_winter


def _compute_all_heuristics(data: pd.DataFrame):
    data[label_heuristics_col] = ''
    for season in season_labels:
        data.loc[data[season], label_heuristics_col] += ',' + season

    has_heuristic = data[label_heuristics_col] != ''
    data.loc[has_heuristic, label_heuristics_col] = data.loc[has_heuristic, label_heuristics_col].str[1:]


def _compute_label_seasons(data: pd.DataFrame, label_col: str):
    modes = {}
    for label in data[label_col].unique():
        is_current_label = data[label_col] == label
        label_count = data.loc[is_current_label, label_col].count()

        max_percent = 0
        max_season = ''
        for season in season_labels:
            season_count = data.loc[is_current_label & data[season], season].count()
            season_percent = season_count / label_count
            if season_percent > max_percent:
                max_percent = season_percent
                max_season = season

        modes[label] = max_season
    return modes


def label_data(data: pd.DataFrame):
    feature_cols = _add_taste_features(data)
    avg_dimensions = math.floor(data['total-features'].mean())
    X = data.loc[:, feature_cols].copy()

    max_feature_variance = 0.15
    distance_threshold = math.sqrt(avg_dimensions) * max_feature_variance
    baseline_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit(X)

    cluster_count = len(season_labels)
    kmeans_clustering = KMeans(n_clusters=cluster_count).fit(X)
    spec_clustering = SpectralClustering(n_clusters=cluster_count).fit(X)

    baseline_col = 'baseline_labels'
    kmeans_col = 'kmeans_labels'
    spec_col = 'spec_labels'
    data[baseline_col] = baseline_clustering.labels_
    data[kmeans_col] = kmeans_clustering.labels_
    data[spec_col] = spec_clustering.labels_ + cluster_count

    label_col = 'accepted_labels'
    find_common_labels(data, 'name', kmeans_col, spec_col)
    data.loc[data['confirmed_groups'].notna(), label_col] = \
        data.loc[data['confirmed_groups'].notna(), 'confirmed_groups']
    data.loc[data['likely_groups'].notna(), label_col] = \
        data.loc[data['likely_groups'].notna(), 'likely_groups']

    find_common_labels(data, 'name', label_col, baseline_col)

    _add_label_heuristics(data)

    _compute_all_heuristics(data)

    label_modes = _compute_label_seasons(data, label_col)
    for label_group, season in label_modes.items():
        data.loc[(data[label_heuristics_col] == '') & (data[label_col] == label_group), season] = True
