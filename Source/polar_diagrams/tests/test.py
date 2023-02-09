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
_LIST_TAYLOR_COLUMNS = [
    'Model', 'Standard Deviation', 'Correlation', 'Angle', 'RMS',
    'Normalized RMS', 'Normalized Standard Deviation']
_LIST_MI_COLUMNS = [
    'Model', 'Entropy', 'Mutual Information', 'Fixed_MI', 'Normalized Entropy',
    'Normalized MI', 'Angle_NMI', 'Root Entropy', 'Joint_entropies',
    'Scaled MI', 'Angle_SMI', 'Normalized Root Entropy', 'VI', 'RVI']
_LIST_COMBINED_COLUMNS = ['Model'] + _LIST_TAYLOR_COLUMNS[1:] +\
    _LIST_MI_COLUMNS[1:]

np.random.seed(_INT_RANDOM_SEED)

dict_mi_parameters_features_continous_target_continous = dict(
    string_entropy_method='auto',
    int_mi_n_neighbors=3,
    bool_discrete_reference_model=False,
    discrete_models=False,
    int_random_state=_INT_RANDOM_SEED)
# TODO: Assert column names and the number of columns of calculate_* return
# TODO: values. Assert the number of rows of return values (must be the same as
# TODO: the number of models)


class DiagramData:
    def __init__(self):
        self.list_input_all_possibilities = None

    def list_generate(self, int_number_of_rows):
        """
        list_generate creates a list of possible inputs for testing purposes.

        Args:
            int_number_of_rows (int): The number of rows in the data set.

        Returns:
            list: A list of possible inputs for the polar_diagrams library.
        """

        df_result = pd.DataFrame()
        int_row_len = int_number_of_rows
        int_column_len = 10

        for int_num_model in range(int_column_len):
            df_result['Model ' + str(int_num_model)] = np.random.normal(
                0, 1, int_row_len)

        df_result_second_scalar = pd.DataFrame().from_dict(
            {'row_1': np.random.random_sample(df_result.shape[1])},
            orient='index', columns=df_result.columns)

        df_result_second_version = pd.DataFrame()
        for string_one_column in df_result.columns:
            df_result_second_version[string_one_column] = df_result[
                string_one_column].to_numpy() + np.random.normal(
                    0, 1, int_row_len)

        self.list_input_all_possibilities = [
            None,
            df_result.copy(),
            [df_result.copy(), df_result_second_version.copy()],
            [df_result.copy(), df_result_second_scalar.copy()]]


class TestDiagrams(unittest.TestCase):
    def setUp(self):
        self.diagram_data = DiagramData()
        self.diagram_data.list_generate(_INT_ROWS_FULL_DATA_SET)

    def test_df_calculate_td_properties(self):
        tuple_bad_inputs = (type(None), list)

        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, tuple_bad_inputs):
                    with self.assertRaises(TypeError):
                        diag.df_calculate_td_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.df_calculate_td_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        pd.DataFrame)

    def test_df_calculate_mid_properties(self):
        tuple_bad_inputs = (type(None), list)

        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, tuple_bad_inputs):
                    with self.assertRaises(TypeError):
                        diag.df_calculate_mid_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.df_calculate_mid_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        pd.DataFrame)

    def test_df_calculate_all_properties(self):
        tuple_bad_inputs = (type(None), list)

        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, tuple_bad_inputs):
                    with self.assertRaises(TypeError):
                        diag.df_calculate_all_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.df_calculate_all_properties(
                            df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        pd.DataFrame)

    def test_chart_create_taylor_diagram(self):
        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, type(None)):
                    with self.assertRaises(TypeError):
                        diag.chart_create_taylor_diagram(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.chart_create_taylor_diagram(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        go.Figure)

    def test_chart_create_mi_diagram(self):
        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, type(None)):
                    with self.assertRaises(TypeError):
                        diag.chart_create_mi_diagram(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.chart_create_mi_diagram(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        go.Figure)

    def test_chart_create_all_diagrams(self):
        for one_input in self.diagram_data.list_input_all_possibilities:
            with self.subTest():
                if isinstance(one_input, type(None)):
                    with self.assertRaises(TypeError):
                        diag.chart_create_all_diagrams(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL)
                else:
                    self.assertIsInstance(
                        diag.chart_create_all_diagrams(
                            list_df_input=one_input,
                            string_reference_model=_STRING_REFERENCE_MODEL),
                        go.Figure)


if __name__ == '__main__':
    unittest.main()
