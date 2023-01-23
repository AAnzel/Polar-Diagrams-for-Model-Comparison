import os
import unittest

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

path_root_data = os.path.join('..', 'Data')
path_root_data_results = os.path.join(path_root_data, 'Results')

INT_RANDOM_SEED = 42
STRING_REFERENCE_MODEL = 'Model 0'
np.random.seed(INT_RANDOM_SEED)


def df_generate():
    """
    df_generate_new_timepoint creates a new DataFrame for testing purposes.

    Returns:
        pandas.DataFrame(): An output DataFrame.
    """

    df_result = pd.DataFrame()
    int_row_len = 100
    int_column_len = 10
    for int_num_model in range(int_column_len):
        df_result['Model ' + str(int_num_model)] = np.random.normal(
            0, 1, int_row_len)

    return df_result


def chart_scatter_anscombes_quartet(df_input):
    chart_result = make_subplots(rows=2, cols=2)

    chart_result.add_trace(
        go.Scatter(x=df_input['x1'], y=df_input['y1'], mode="markers"),
        row=1, col=1)

    chart_result.add_trace(
        go.Scatter(x=df_input['x2'], y=df_input['y2'], mode="markers"),
        row=1, col=2)

    chart_result.add_trace(
        go.Scatter(x=df_input['x3'], y=df_input['y3'], mode="markers"),
        row=2, col=1)

    chart_result.add_trace(
        go.Scatter(x=df_input['x4'], y=df_input['y4'], mode="markers"),
        row=2, col=2)

    chart_result.update_layout(
        template='simple_white', height=1000, title_text="Anscombe's quartet",
        showlegend=False)
    chart_result.update_xaxes(showgrid=True)
    chart_result.update_yaxes(showgrid=True)

    return chart_result


def test_output():
    dict_mi_parameters_features_continous_target_continous = dict(
        string_entropy_method='auto',
        int_mi_n_neighbors=3,
        bool_discrete_reference_model=False,
        discrete_models=False,
        int_random_state=INT_RANDOM_SEED)

    # The original dataset comes from the following paper
    # F. J. Anscombe (1973) Graphs in Statistical Analysis,
    # The American Statistician, 27:1, 17-21,
    # DOI: 10.1080/00031305.1973.10478966
    path_anscombes_quartet = os.path.join(
        path_root_data, 'Dataset_0', 'Anscombes_Quartet.csv')

    df_anscombes_quartet = pd.read_csv(path_anscombes_quartet)
    chart_anscombes = chart_scatter_anscombes_quartet(df_anscombes_quartet)
    chart_anscombes.show()

    df_anscombes_quartet_modified = df_anscombes_quartet.drop(
        ['x1', 'x2', 'x3', 'x4'], axis=1)

    chart_taylor_res = diag.chart_create_taylor_diagram(
        df_anscombes_quartet_modified, string_reference_model='y4',
        bool_normalized_measures=True)
    chart_taylor_res.show()

    chart_mid_res = diag.chart_create_mi_diagram(
        df_anscombes_quartet_modified, string_reference_model='y4',
        bool_normalized_measures=True,
        dict_mi_parameters=dict_mi_parameters_features_continous_target_continous) # noqa
    chart_mid_res.show()

    chart_both_res = diag.chart_create_all_diagrams(
        df_anscombes_quartet_modified, string_reference_model='y4',
        bool_normalized_measures=True,
        dict_mi_parameters=dict_mi_parameters_features_continous_target_continous) # noqa
    chart_both_res.show()

    return None


class TestDiagrams(unittest.TestCase):
    def test_df_calculate_td_properties(self):
        self.assertIsInstance(
            diag.df_calculate_td_properties(
                df_generate(), string_reference_model=STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_df_calculate_mid_properties(self):
        self.assertIsInstance(
            diag.df_calculate_mid_properties(
                df_generate(), string_reference_model=STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_df_calculate_all_properties(self):
        self.assertIsInstance(
            diag.df_calculate_all_properties(
                df_generate(), string_reference_model=STRING_REFERENCE_MODEL),
            pd.DataFrame)

    def test_chart_create_taylor_diagram(self):
        return None

    def test_chart_create_mi_diagram(self):
        return None

    def test_chart_create_all_diagrams(self):
        return None


if __name__ == '__main__':
    unittest.main()
