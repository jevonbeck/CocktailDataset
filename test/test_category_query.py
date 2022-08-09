from unittest import TestCase
# from unittest.mock import call, patch, create_autospec, ANY

import pandas as pd
from pandas.testing import assert_series_equal

from playlist.category_query import CategoryQuery, is_phrase_in_string, \
    ingredient_str_has_at_least_one_phrase, ingredient_has_at_least_one_phrase, \
    ingredient_str_has_at_least_one_phrase_from_each_list, \
    ingredient_has_at_least_one_phrase_from_each_list


class TestCategoryQuery(TestCase):

    def test_phrase_is_in_string(self):
        # Given
        phrase = 'gin'
        strings = ['random', 'staging', 'vodka,gin', 'ginger', 'gin,vodka', 'gin', 'ginger gin', 'ginger login']
        expected_result = [False, False, True, False, True, True, True, False]

        # When
        results = [is_phrase_in_string(phrase, string) for string in strings]

        # Then
        self.assertEqual(expected_result, results)

    def test_phrase_in_ingredient_series(self):
        # Given
        ingredients = pd.Series(['some blah', 'great,rum', 'syrup super', 'random', 'syrup'])
        contain_phrases = ['rum', 'syrup']

        expected_result = pd.Series([False, True, True, False, True])

        # When
        result_series = ingredient_has_at_least_one_phrase(ingredients, contain_phrases)
        result_str = ingredients.apply(ingredient_str_has_at_least_one_phrase, args=(contain_phrases,))

        # Then
        assert_series_equal(expected_result, result_series)
        assert_series_equal(result_str, result_series)

    def test_multiple_phrases_in_ingredient_series(self):
        # Given
        ingredients = pd.Series(['some blah', 'great rum', 'syrup super', 'random', 'syrup'])
        phrase_lists = [['rum', 'syrup'], ['great', 'super']]

        expected_result = pd.Series([False, True, True, False, False])

        # When
        result_series = ingredient_has_at_least_one_phrase_from_each_list(ingredients, phrase_lists)
        result_str = ingredients.apply(ingredient_str_has_at_least_one_phrase_from_each_list, args=(phrase_lists,))

        # Then
        assert_series_equal(expected_result, result_series)
        assert_series_equal(result_str, result_series)

    def test_basic_category_query(self):
        # Given
        ingredients = pd.Series(['cherry vodka', 'rand', 'vodka', 'smirnoff aquavit'])
        query = CategoryQuery('vodka')

        expected_result = pd.Series([True, False, True, False])

        # When
        result = query.ingredient_is_in_category(ingredients)

        # Then
        assert_series_equal(expected_result, result)
        
    def test_category_query_contains(self):
        # Given
        ingredients = pd.Series(['cherry vodka', 'rand', 'vodka', 'smirnoff aquavit'])
        query = CategoryQuery('vodka', contains=['aquavit'])

        expected_result = pd.Series([True, False, True, True])

        # When
        result = query.ingredient_is_in_category(ingredients)

        # Then
        assert_series_equal(expected_result, result)

    def test_category_query_contains_and(self):
        # Given
        ingredients = pd.Series(['cherry vodka', 'rand', 'vodka', 'smirnoff aquavit'])
        query = CategoryQuery('rum', contains_and=[['aquavit', 'cherry'], ['smirnoff', 'vodka']])

        expected_result = pd.Series([True, False, False, True])

        # When
        result = query.ingredient_is_in_category(ingredients)

        # Then
        assert_series_equal(expected_result, result)

    def test_category_query_not_contains(self):
        # Given
        ingredients = pd.Series(['cherry vodka', 'rand', 'vodka', 'smirnoff vodka'])
        query = CategoryQuery('vodka', not_contains=['smirnoff'])

        expected_result = pd.Series([True, False, True, False])

        # When
        result = query.ingredient_is_in_category(ingredients)

        # Then
        assert_series_equal(expected_result, result)
