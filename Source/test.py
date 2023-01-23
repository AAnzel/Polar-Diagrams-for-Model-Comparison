import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import diagrams as diag

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
np.random.seed(INT_RANDOM_SEED)


def df_generate_new_timepoint(df_input):
    """
    df_generate_new_timepoint adds noise to each column of the input dataframe,
    thus creating a slightly modified version of that dataframe. This version
    can be considered a snapshot of a phenomenon in a different time point.

    Args:
        pd.DataFrame(): An input dataframe that contains models in columns.

    Returns:
        pd.DataFrame(): An output dataframe which comes from the procedure
        described above.
    """

    df_result = pd.DataFrame()
    int_len = df_input.shape[0]
    for string_one_column in df_input.columns:
        df_result[string_one_column] = df_input[
            string_one_column].to_numpy() + np.random.normal(0, 1, int_len)

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


def main():
    if not os.path.exists(path_root_data_results):
        os.mkdir(path_root_data_results)

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
        df_anscombes_quartet_modified, string_reference_model='y4')
    chart_taylor_res.show()

    return None


if __name__ == '__main__':
    main()
