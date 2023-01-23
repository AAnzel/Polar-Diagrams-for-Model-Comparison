import unittest

from plotly import graph_objects as go
import numpy as np
import pandas as pd

import polar_diagrams as diag

__author__ = 'Aleksandar Anžel'
__copyright__ = ''
__credits__ = ['Aleksandar Anžel']
__license__ = 'GNU General Public License v3.0'
__version__ = '1.0'
__maintainer__ = 'Aleksandar Anžel'
__email__ = 'aleksandar.anzel@uni-marburg.de'
__status__ = 'Dev'

# ==============================================
# This script is used to test the 'diag' library
# ==============================================

_INT_RANDOM_SEED = 42
_STRING_REFERENCE_MODEL = 'Model 0'
_INT_ROWS_FULL_DATA_SET = 100
_INT_ROWS_SCALAR_DATA_SET = 1
np.random.seed(_INT_RANDOM_SEED)

dict_mi_parameters_features_continous_target_continous = dict(
    string_entropy_method='auto',
    int_mi_n_neighbors=3,
    bool_discrete_reference_model=False,
    discrete_models=False,
    int_random_state=_INT_RANDOM_SEED)


def df_generate(int_number_of_rows):
    """
    df_generate_new_timepoint creates a new DataFrame for testing purposes.

    Returns:
        pandas.DataFrame(): An output DataFrame.
    """

    df_result = pd.DataFrame()
    int_row_len = int_number_of_rows
    int_column_len = 10
    for int_num_model in range(int_column_len):
        df_result['Model ' + str(int_num_model)] = np.random.normal(
            0, 1, int_row_len)

    return df_result


class TestDiagrams(unittest.TestCase):
    def test_df_calculate_td_properties(self):
        self.assertIsInstance(
            diag.df_calculate_td_properties(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_df_calculate_mid_properties(self):
        self.assertIsInstance(
            diag.df_calculate_mid_properties(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_df_calculate_all_properties(self):
        self.assertIsInstance(
            diag.df_calculate_all_properties(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_chart_create_taylor_diagram(self):
        self.assertIsInstance(
            diag.chart_create_taylor_diagram(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            go.Figure)

    def test_chart_create_mi_diagram(self):
        self.assertIsInstance(
            diag.chart_create_mi_diagram(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            go.Figure)

    def test_chart_create_all_diagrams(self):
        self.assertIsInstance(
            diag.chart_create_all_diagrams(
                df_generate(_INT_ROWS_FULL_DATA_SET),
                string_reference_model=_STRING_REFERENCE_MODEL),
            go.Figure)


if __name__ == '__main__':
    unittest.main()
