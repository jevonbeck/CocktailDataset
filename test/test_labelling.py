from playlist.labelling import find_common_labels
from unittest import TestCase
# from unittest.mock import call, patch, create_autospec, ANY

import json
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


class TestLabelling(TestCase):

    def test_find_common_labels(self):
        # Given
        c_distribution = json.dumps({'Q': 2, 'R': 2})
        expected_result = pd.DataFrame(np.array([
            ['A', 'P', 'P', None, None, None], ['A', 'P', 'P', None, None, None],
            ['A', 'P', 'P', None, None, None], ['A', 'P', 'P', None, None, None],
            ['B', 'P', None, 'Q', None, None], ['B', 'Q', None, 'Q', None, None],
            ['B', 'Q', None, 'Q', None, None], ['B', 'Q', None, 'Q', None, None],
            ['C', 'Q', None, None, 'C', c_distribution], ['C', 'Q', None, None, 'C', c_distribution],
            ['C', 'R', None, None, 'C', c_distribution], ['C', 'R', None, None, 'C', c_distribution],
            ['D', 'R', None, 'R', None, None], ['D', 'R', None, 'R', None, None],
            ['D', 'R', None, 'R', None, None], ['D', 'S', None, 'R', None, None],
            ['E', 'S', 'S', None, None, None], ['E', 'S', 'S', None, None, None],
            ['E', 'S', 'S', None, None, None], ['E', 'S', 'S', None, None, None],
        ]), columns=['a_labels', 'b_labels', 'confirmed_groups', 'likely_groups',
                     'uncertain_groups', 'uncertain_distribution'])
        expected_result.insert(0, 'id',
                               expected_result.index.to_series().add_prefix('id-').reset_index()['index']
                               )

        data = expected_result.loc[:, ['id', 'a_labels', 'b_labels']].copy()

        # When
        find_common_labels(data, 'id', 'a_labels', 'b_labels')

        # Then
        assert_frame_equal(expected_result, data)

