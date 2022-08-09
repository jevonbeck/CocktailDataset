from unittest import TestCase
import pandas as pd
from pandas.testing import assert_series_equal

from playlist.data import replace_phrase_in_string, standardise_measurement_str, standardise_measurement


class TestData(TestCase):

    def test_replace_phrase_in_string(self):
        # Given
        phrase = 'gin'
        replacement = 'spirit'
        strings = ['random', 'staging', 'vodka,gin', 'ginger', 'gin,vodka', 'gin', 'ginger gin', 'ginger login']
        expected_result = [
            'random', 'staging', 'vodka,spirit', 'ginger', 'spirit,vodka', 'spirit', 'ginger spirit', 'ginger login'
        ]

        # When
        results = [replace_phrase_in_string(string, phrase, replacement) for string in strings]

        # Then
        self.assertEqual(expected_result, results)

    def test_standardise_measurement(self):
        # Given
        col_names = ['egg', 'citrus juice', 'other']
        measurements = [
            '1 3/4 ounces', '2 1/2 bottles', '1 750-ml', '2 dashes', '1 splash',
            '2 tsps', '1/4 c', '1/2 tsp grenadine', 'dash', '', '1/3 or 1', '1'
        ]
        measurement_series = pd.Series(measurements)

        expected_result_base = [52.5, 1875, 750, 4, 2, 10, 60, 2.5, 2, 0]
        expected_result_egg = expected_result_base + [15, 45]
        expected_result_citrus = expected_result_base + [10, 30]
        expected_result_other = expected_result_base + [0, 0]
        expected_result = expected_result_egg + expected_result_citrus + expected_result_other

        # When
        results = [
            standardise_measurement_str(measurement, col_name) for col_name in col_names for measurement in measurements
        ]
        result_series = standardise_measurement(measurement_series, 'egg')
        result_str = measurement_series.apply(standardise_measurement_str, args=('egg',))

        # Then
        self.assertEqual(expected_result, results)
        assert_series_equal(result_str, result_series)
