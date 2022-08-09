from playlist.data import get_data
import pandas as pd

from playlist.labelling import label_data, season_labels, label_heuristics_col
from playlist.modelling import extract_test_data, evaluate_and_predict_model_by_cols

_liquid_cols = [
    'alco-pop', 'beer', 'bitters', 'brandy', 'citrus juice', 'citrus spirit', 'cream spirit', 'dessert spirit', 'egg',
    'flavoured water', 'floral spirit', 'fortified wine', 'fruit juice', 'fruit spirit', 'gin', 'herbal spirit',
    'milk', 'rum', 'sauce', 'soda', 'sparkling wine', 'syrup', 'tea', 'tequila', 'water', 'vodka', 'whisky', 'wine',
]

_feature_cols = [
    'alco-percent', 'citrus-percent', 'fruit-percent', 'herbal-percent', 'milk-percent', 'bubbles-percent',
    'wine-percent', 'water-percent', 'dessert-percent', 'beer-percent', 'egg-percent', 'sauce-percent',
    'syrup-percent'
]


def main():
    # data, liquid_cols = get_data()

    data = pd.read_csv('cleaned_data.csv')
    liquid_cols = _liquid_cols

    label_data(data)

    training_data = extract_test_data(data, ['name'] + liquid_cols + season_labels + [label_heuristics_col])

    evaluate_and_predict_model_by_cols(training_data, data, liquid_cols, season_labels)
    print("hello world!")


if __name__ == "__main__":
    main()
