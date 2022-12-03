import math
import pandas as pd
import numpy as np

from npeet import entropy_estimators
from sklearn.metrics import mean_squared_error
from scipy.stats import differential_entropy
from sklearn.feature_selection import mutual_info_regression

import plotly.graph_objects as go
from plotly.subplots import make_subplots


__author__ = 'Aleksandar Anžel'
__copyright__ = ''
__credits__ = ['Aleksandar Anžel']
__license__ = 'GNU General Public License v3.0'
__version__ = '1.0'
__maintainer__ = 'Aleksandar Anžel'
__email__ = 'aleksandar.anzel@uni-marburg.de'
__status__ = 'Dev'


# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html?highlight=entropy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html
# https://github.com/paulbrodersen/entropy_estimators/
# https://github.com/gregversteeg/NPEET <- USE THIS ONE FIRST !!!!!!!!!!!!!!!!
# https://github.com/BiuBiuBiLL/NPEET_LNC <- And this one second (an improvement) # noqa


STRING_BACKGROUND_COLOR = '#FFFFFF'
STRING_GRID_COLOR = '#C0C0C0'
STRING_LABEL_TITLE_COLOR = '#404040'

# Note: Color acquired from: https://public.tableau.com/views/TableauColors/ColorsbyHexCode?%3Aembed=y&%3AshowVizHome=no&%3Adisplay_count=y&%3Adisplay_static_image=y # noqa
LIST_TABLEAU_10 = ['#1f77b4', '#2ca02c', '#7f7f7f', '#8c564b', '#17becf',
                   '#9467bd', '#bcbd22', '#d62728', '#e377c2', '#ff7f0e']
LIST_TABLEAU_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                   '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                   '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                   '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def df_calculate_td_properties(df_input, string_reference_model,
                               string_corr_method='pearson'):
    """
    df_calculate_td_properties caclulates all necessary statistical information
    for the Taylor diagram from the input data set.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant statistical
        information.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        string_corr_method (str, optional): This string contains the name of
        the method to be used when calculating the correlation. Defaults to
        'pearson'.

    Raises:
        ValueError: The error is raised if the string_corr_method is not one of
        the following 'pearson', 'kendall', 'spearman'.

    Returns:
        pandas.DataFrame: This dataframe contains model names as indices and
        statistical properties as columns.
    """

    list_valid_corr_methods = ['pearson', 'kendall', 'spearman']
    if string_corr_method not in list_valid_corr_methods:
        raise ValueError('string_corr_method is not one of the following:' +
                         str(list_valid_corr_methods))

    list_all_features = df_input.columns.to_list()

    # Initialize dict
    dict_result = {}
    for string_one_model in list_all_features:
        dict_result[string_one_model] = []

    for string_one_model in list_all_features:
        # Calculating standard deviations
        dict_result[string_one_model].append(
            df_input[string_one_model].std(ddof=0))

        # Calculating Pearson's correlation
        dict_result[string_one_model].append(
            df_input[string_reference_model].corr(
                df_input[string_one_model], method=string_corr_method))

        # Calculate arccos of Pearson's correlation
        dict_result[string_one_model].append(math.degrees(
            math.acos(dict_result[string_one_model][-1])))

        # Calculating RMS
        dict_result[string_one_model].append(
            mean_squared_error(
                df_input[string_reference_model],
                df_input[string_one_model], squared=False))

    for string_one_model in list_all_features:
        # Calculating correlation using RMS formula
        # dict_result[string_one_model].append(
        #    (dict_result[string_one_model][0]**2 + dict_result[string_reference_model][0]**2 - dict_result[string_one_model][3]**2) / (2 * dict_result[string_one_model][0] * dict_result[string_reference_model][0]) # noqa
        # )

        # Calculating the angle using calculated correlation
        # dict_result[string_one_model].append(math.degrees(
        #    math.acos(dict_result[string_one_model][-1])))

        # Normalizing the RMS as in the paper
        dict_result[string_one_model].append(
            dict_result[string_one_model][3] /
            dict_result[string_reference_model][0])

        # Calculating normalized standard deviation
        dict_result[string_one_model].append(
            dict_result[string_one_model][0] / dict_result[
                string_reference_model][0])

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        # columns=['Standard Deviation', 'Correlation', 'Angle', 'RMS',
        #          'Calculated_Corr', 'Calculated_Angle', 'Normalized_RMS',
        #          'Normalized_STD']
        columns=['Standard Deviation', 'Correlation', 'Angle', 'RMS',
                 'Normalized_RMS', 'Normalized_STD'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def list_adapt_to_npeet(list_input):
    """
    list_adapt_to_npeet function takes list of values from one model and adapts
    that list to the npeet functions.

    Args:
        list_input (list): This list holds values of one model.

    Returns:
        list: The elements of this list are lists that contain one value from
        the passed argument.
    """
    return [[i] for i in list_input]


def float_calculate_discrete_entropy(list_labels, int_base=2):
    """
    float_calculate_discrete_entropy calculates Shannon (discrete) entropy of
    the passed list of labels.

    Args:
        list_labels (list): This elements of this list are labels that can be
        of any type (numerical, characters, etc.).
        int_base (int, optional): This parameter defines the base of the
        logarithmic function which is used when the resulting entropy is
        calculated. Defaults to 2.

    Returns:
        float: The resulting Shannon (discrete) entropy.
    """

    int_num_of_labels = len(list_labels)

    if int_num_of_labels <= 1:
        return 0

    list_value, list_counts = np.unique(list_labels, return_counts=True)
    list_probabilities = list_counts / int_num_of_labels
    int_num_of_classes = np.count_nonzero(list_probabilities)

    if int_num_of_classes <= 1:
        return 0

    float_entropy = 0.0

    for float_one_probability in list_probabilities:
        float_entropy -= float_one_probability * math.log(
            float_one_probability, int_base)

    return float_entropy


def df_calculate_mid_properties(df_input, string_reference_model,
                                dict_mi_parameters=dict(
                                    string_library='scipy_sklearn',
                                    string_entropy_method='auto',
                                    bool_discrete_features=False)):
    """
    df_calculate_mid_properties caclulates all necessary information theory
    properties for the Mutual Information diagram from the input data set.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant information
        theory properties
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        dict_mi_parameters (dict, optional): This dictionary contains
        configuration parameters for the calculation of entropy and mutual
        information. Defaults to
        dict(string_library='scipy_sklearn', string_entropy_method='auto',
             bool_discrete_features=False).

    Raises:
        ValueError: The error is raised if string_library is not one of the
        following 'scipy_sklearn', 'npeet'
        ValueError: The error is raised if string_entropy_method is not one of
        the following 'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'

    Returns:
        pandas.DataFrame: This dataframe contains model names as indices and
        information theory properties as columns.
    """

    list_valid_entropy_methods = ['vasicek', 'van es', 'ebrahimi', 'correa',
                                  'auto']
    list_valid_libraries = ['scipy_sklearn', 'npeet']
    list_valid_discrete_features = [True, False]
    # dict_mi_parameters = dict(
    #    string_library='',
    #    string_entropy_method='',
    #    bool_discrete_features=bool)

    if dict_mi_parameters['string_library'] not in list_valid_libraries:
        raise ValueError('string_library is not one of the following:' +
                         str(list_valid_libraries))
    if dict_mi_parameters['string_library'] == 'scipy_sklearn':
        if dict_mi_parameters[
                'string_entropy_method'] not in list_valid_entropy_methods:
            raise ValueError('string_entropy_method is not one of the' +
                             ' following:' +
                             str(list_valid_entropy_methods))
    if dict_mi_parameters[
            'bool_discrete_features'] not in list_valid_discrete_features:
        raise ValueError('bool_discrete_features is not one of the following:'
                         + str(list_valid_discrete_features))

    list_all_features = df_input.columns.to_list()

    list_adapted_npeet_reference = list_adapt_to_npeet(
        df_input[string_reference_model])

    # Initialize dict
    dict_result = {}
    for string_one_model in list_all_features:
        dict_result[string_one_model] = []

    # TODO: Entropies are often negative when using default parameters
    # That is causing an error when calculating angles
    # Try to find better default parameters so it doesn't happen
    for string_one_model in list_all_features:
        list_adapted_npeet_one = list_adapt_to_npeet(
            df_input[string_one_model])

        if dict_mi_parameters['string_library'] == 'scipy_sklearn':
            # Calculate entropies
            # 0 in the list
            if dict_mi_parameters['bool_discrete_features'] is True:
                dict_result[string_one_model].append(
                    float_calculate_discrete_entropy(
                        df_input[string_one_model], int_base=2))
            else:
                dict_result[string_one_model].append(
                    differential_entropy(df_input[string_one_model], base=2))
            # dict_result[string_one_model].append(
            #    mutual_info_regression(
            #        df_input[string_one_model].to_numpy().reshape(-1, 1),
            #        df_input[string_one_model],
            #        discrete_features=False)[0])

            # Calculate mutual informations against the reference feature
            # 1 in the liststring_angular_column
            dict_result[string_one_model].append(
                mutual_info_regression(
                    df_input[string_reference_model].to_numpy().reshape(
                        -1, 1),
                    df_input[string_one_model],
                    discrete_features=dict_mi_parameters[
                        'bool_discrete_features'])[0])

        else:
            if dict_mi_parameters['bool_discrete_features'] is True:
                # Calculate entropies
                # 0 in the list
                dict_result[string_one_model].append(
                    entropy_estimators.entropyd(list_adapted_npeet_one))
                # Calculate mutual informations against the reference feature
                # 1 in the liststring_angular_column
                dict_result[string_one_model].append(
                    entropy_estimators.midd(list_adapted_npeet_reference,
                                            list_adapted_npeet_one))
            else:
                dict_result[string_one_model].append(
                    entropy_estimators.entropy(list_adapted_npeet_one))
                # Calculate mutual informations against the reference feature
                # 1 in the liststring_angular_column
                dict_result[string_one_model].append(
                    entropy_estimators.mi(list_adapted_npeet_reference,
                                          list_adapted_npeet_one))

    for string_one_model in list_all_features:
        # Calculating fixed MI from equation 17 from the paper
        # I(X,Y) = I~(X,Y) * (H(X) / I~(X,X)) where I~ is MI calculated using
        # some estimation method. This MI is used for every calculation
        # afterwards
        # 2 in the list
        dict_result[string_one_model].append(
            dict_result[string_one_model][1] *
            (dict_result[string_reference_model][0] /
             dict_result[string_reference_model][1])
        )

        # Calculate scaled entropies
        # 3 in the list
        dict_result[string_one_model].append(
            dict_result[string_one_model][0] /
            dict_result[string_reference_model][0]
        )

        # Calculate normalized mutual information according to the paper
        # NMI(X,Y) = I(X,Y) / sqrt(H(X) * H(Y))
        # This has to be modified because H(X)*H(Y) can be negative for
        # differential entropies. NMI is not used at all for the MI chart
        # that spans two quadrants
        # 4 in the list
        float_product = dict_result[string_reference_model][0] *\
            dict_result[string_one_model][0]

        if float_product < 0:
            dict_result[string_one_model].append(1)

        else:
            dict_result[string_one_model].append(
                dict_result[string_one_model][2] / math.sqrt(float_product))

        # Calculate arccos of normalized mutual information according to the
        # paper arccos(NMI(X,Y))
        # 5 in the list
        dict_result[string_one_model].append(
            math.degrees(math.acos(dict_result[string_one_model][4])))

    for string_one_model in list_all_features:
        ######################################################################
        # This part is for the chart that spans two quadrants

        # First calculate joint entropies by using equation 10 from the paper
        # I(X,Y) = H(X) + H(Y) - H(X,Y) => H(X,Y) = H(X) + H(Y) - I(X,Y)
        # 6 in the list
        dict_result[string_one_model].append(
            dict_result[string_reference_model][0] +
            dict_result[string_one_model][0] -
            dict_result[string_one_model][2]
        )

        # Calculate scaled mutual information according to the paper
        # SMI(X,Y) = I(X,Y) * ((H(X,Y) / (H(X)*H(Y)))) (equation 15)
        # 7 in the list
        smi_x_y = dict_result[string_one_model][2] * (
            dict_result[string_one_model][6] /
            (dict_result[string_reference_model][0]
             * dict_result[string_one_model][0]))

        dict_result[string_one_model].append(smi_x_y)

        # Calculate arccos of biased scaled mutual information according to the
        # paper arccos(c(X,Y)) where c(X,Y) = 2*SMI(X,Y) - 1
        # 8 in the list
        float_c_x_y = 2*smi_x_y - 1

        if float_c_x_y > 1:
            dict_result[string_one_model].append(
                math.degrees(math.acos(1)))
        elif float_c_x_y < -1:
            dict_result[string_one_model].append(
                math.degrees(math.acos(-1)))
        else:
            dict_result[string_one_model].append(
                math.degrees(math.acos(float_c_x_y)))

        # Calculate root entropy
        # 9 in the list
        if dict_result[string_one_model][0] >= 0:
            float_root_entropy = math.sqrt(dict_result[string_one_model][0])
        else:
            float_root_entropy = -1

        dict_result[string_one_model].append(float_root_entropy)
        ######################################################################

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        columns=['Entropy', 'Mutual Information', 'Fixed_MI', 'Scaled_entropy',
                 'Normalized MI', 'Angle_NMI', 'Joint_entropies', 'Scaled MI',
                 'Angle_SMI', 'Root Entropy'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def df_calculate_all_properties(df_input, string_reference_model,
                                dict_mi_parameters, string_corr_method):
    """
    df_calculate_all_properties caclulates all necessary statistical and
    information theory properties for the Taylor and Mutual Information diagram
    from the input data set.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant statistical
        information and information theory properties.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        dict_mi_parameters (dict, optional): This dictionary contains
        configuration parameters for the calculation of entropy and mutual
        information. Defaults to
        dict(string_library='scipy_sklearn', string_entropy_method='auto',
             bool_discrete_features=False).
        string_corr_method (str, optional): This string contains the name of
        the method to be used when calculating the correlation. Defaults to
        'pearson'.

    Returns:
        pandas.DataFrame: This dataframe contains model names as indices and
        statistical and information theory properties as columns.
    """

    df_td = df_calculate_td_properties(
        df_input, string_reference_model, string_corr_method)
    df_mid = df_calculate_mid_properties(
        df_input, string_reference_model, dict_mi_parameters)

    return df_td.merge(df_mid, on='Model', how='inner')


def chart_create_diagram(df_input, string_reference_model,
                         string_mid_type='scaled', bool_flag_as_subplot=False,
                         chart_result_upper=None,
                         string_diagram_type='taylor'):
    """
    chart_create_diagram is a general function that creates both the Taylor and
    the Mutual Information diagrams according to the passed argument.

    Args:
        df_input (pandas.DataFrame): This dataframe contains model names as
        indices and statistical and information theory properties as columns.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        string_mid_type (str, optional): This string contains the type of the
        Mutual Information diagram. If it is 'scaled' then it will span both
        quadrants. If it is 'normalized', it will span only the first
        quadrant of the circle. Defaults to 'scaled'.
        bool_flag_as_subplot (bool, optional): This boolean parameter is used
        to determine if the passed chart is a subplot or not. If it is False,
        then only one diagram is created. If it is True, both Taylor and Mutual
        Information diagrams are created and placed side-by-side. Defaults to
        False.
        chart_result_upper (plotly.graph_objects.Figure, optional): This chart
        is not None only if both diagrams have to be created. It contains the
        blank canvas (1 row, 2 columns) for both objects. Defaults to None.
        string_diagram_type (str, optional): This string contains the type of
        the diagram that has to be created. Defaults to 'taylor'.

    Raises:
        ValueError: The error is raised if string_diagram_type is not one of
        the following 'taylor', 'mid'
        ValueError: The error is raised if string_mid_type is not one of
        the following 'scaled', 'normalized'

    Returns:
        plotly.graph_objects.Figure: This chart contains the resulting Taylor
        or Mutual Information diagram.
    """

    list_valid_diagram_types = ['taylor', 'mid']
    list_valid_mid_types = ['scaled', 'normalized']

    if string_diagram_type not in list_valid_diagram_types:
        raise ValueError('string_diagram_type not in ' +
                         str(list_valid_diagram_types))
    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    # General properties
    list_color_scheme = None
    string_tooltip_label_0 = 'Model'
    int_number_of_models = len(df_input[string_tooltip_label_0].to_list())

    if int_number_of_models <= 9:
        list_color_scheme = LIST_TABLEAU_10
    else:
        list_color_scheme = LIST_TABLEAU_20

    np_tmp = np.array(
        [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

    if string_diagram_type == 'taylor':
        bool_show_legend = True
        string_radial_column = 'Standard Deviation'
        string_angular_column = 'Angle'
        string_angular_column_label = 'Correlation'
        string_tooltip_label_1 = string_radial_column
        string_tooltip_label_2 = string_angular_column_label
        bool_only_half = True if (
            df_input[string_angular_column] <= 90).all() else False
        int_subplot_column_number = 1
        np_angular_labels = np_tmp if bool_only_half else np.concatenate(
            (-np_tmp[:0:-1], np_tmp))
        np_angular_ticks = np.degrees(np.arccos(np_angular_labels))

    else:
        bool_show_legend = False
        int_subplot_column_number = 2

        if string_mid_type == 'scaled':
            string_angular_column = 'Angle_SMI'
            string_radial_column = 'Entropy'
            string_angular_column_label = 'Scaled Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Scaled MI'
            bool_only_half = False
            np_angular_labels = np.concatenate((-np_tmp[:0:-1], np_tmp))
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))
            np_angular_labels = np.round((np_angular_labels + 1) / 2, 3)

        else:
            string_angular_column = 'Angle_NMI'
            string_radial_column = 'Root Entropy'
            string_angular_column_label = 'Normalized Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Normalized MI'
            bool_only_half = True
            np_angular_labels = np_tmp
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))

    int_max_angle = 90 if bool_only_half else 180
    float_max_r = df_input[string_radial_column].max() +\
        df_input[string_radial_column].mean()

    if bool_flag_as_subplot is True:
        chart_result = chart_result_upper
    else:
        chart_result = go.Figure()

    np_tooltip_data = list(
        df_input[[string_tooltip_label_0, string_tooltip_label_1,
                  string_tooltip_label_2]].to_numpy())
    string_tooltip_hovertemplate = (
        string_tooltip_label_0 + ': %{customdata[0]}<br>' +
        string_tooltip_label_1 + ': %{customdata[1]:.3f}<br>' +
        string_tooltip_label_2 + ': %{customdata[2]:.3f}<br>' +
        '<extra></extra>')

    dict_polar_chart = dict(
        sector=[0, int_max_angle],
        bgcolor=STRING_BACKGROUND_COLOR,
        radialaxis=dict(
            range=[0, float_max_r],
            griddash='dot',
            gridcolor=STRING_GRID_COLOR,
            tickcolor=STRING_LABEL_TITLE_COLOR,
            tickfont=dict(color=STRING_LABEL_TITLE_COLOR),
            layer='below traces',
            title=dict(
                text=string_radial_column,
                font=dict(
                    color=STRING_LABEL_TITLE_COLOR))),
        angularaxis=dict(
            direction="counterclockwise",
            tickvals=np_angular_ticks,
            ticktext=np_angular_labels,
            tickcolor=STRING_LABEL_TITLE_COLOR,
            tickfont=dict(color=STRING_LABEL_TITLE_COLOR),
            griddash='dot',
            gridcolor=STRING_GRID_COLOR,
            layer='below traces'))
    dict_legend = dict(
        title=dict(text=string_tooltip_label_0),
        font=dict(
            color=STRING_LABEL_TITLE_COLOR),
        bgcolor=STRING_BACKGROUND_COLOR,
        bordercolor=STRING_GRID_COLOR,
        borderwidth=0.2)

    for tmp_r, tmp_angle, tmp_model_int, tmp_model in zip(
            df_input[string_radial_column], df_input[string_angular_column],
            pd.factorize(df_input[string_tooltip_label_0])[0],
            df_input[string_tooltip_label_0]):

        if bool_flag_as_subplot is True:
            chart_result.add_trace(
                go.Scatterpolar(
                    name=tmp_model,
                    r=[tmp_r],
                    theta=[tmp_angle],
                    mode='markers',
                    legendgroup=tmp_model,
                    showlegend=bool_show_legend,
                    customdata=[
                        np_tooltip_data[tmp_model_int]] * int_number_of_models,  # This has to be done like this, since we add traces one by one # noqa
                    hovertemplate=string_tooltip_hovertemplate,
                    marker=dict(
                        color=list_color_scheme[tmp_model_int])),
                row=1,
                col=int_subplot_column_number)

        else:
            chart_result.add_trace(
                go.Scatterpolar(
                    name=tmp_model,
                    r=[tmp_r],
                    theta=[tmp_angle],
                    mode='markers',
                    customdata=[
                        np_tooltip_data[tmp_model_int]] * int_number_of_models,  # This has to be done like this, since we add traces one by one # noqa
                    hovertemplate=string_tooltip_hovertemplate,
                    marker=dict(
                        color=list_color_scheme[tmp_model_int])))

    if bool_flag_as_subplot is True:
        if string_diagram_type == 'taylor':
            chart_result.update_layout(
                polar=dict_polar_chart,
                legend=dict_legend,
                height=600,
                showlegend=True)

        else:
            chart_result.update_layout(
                polar2=dict_polar_chart,
                legend=dict_legend,
                height=600,
                showlegend=True)

    else:
        chart_result.update_layout(
            polar=dict_polar_chart,
            height=600,
            legend=dict_legend,
            title=dict(
                text=string_angular_column_label,
                x=0.5,
                y=0.9,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color=STRING_LABEL_TITLE_COLOR)))

    return chart_result


def chart_create_taylor_diagram(df_input, string_reference_model,
                                string_corr_method):
    """
    chart_create_taylor_diagram creates the Taylor diagram according to the
    df_input argument where models are placed in columns and rows contain
    model predictions.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant statistical
        information.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        string_corr_method (str, optional): This string contains the name of
        the method to be used when calculating the correlation. Defaults to
        'pearson'.

    Returns:
       plotly.graph_objects.Figure: This chart contains the resulting Taylor
       diagram.
    """

    df_td = df_calculate_td_properties(
        df_input, string_reference_model, string_corr_method)
    chart_result = chart_create_diagram(
        df_td, string_reference_model=string_reference_model,
        bool_flag_as_subplot=False, string_diagram_type='taylor')

    return chart_result


def chart_create_mi_diagram(df_input, string_reference_model,
                            string_mid_type, dict_mi_parameters):
    """
    chart_create_mi_diagram creates the Mutual Information diagram according
    to the df_input argument where models are placed in columns and rows
    contain model predictions.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant information
        theory information.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        string_mid_type (str, optional): This string contains the type of the
        Mutual Information diagram. If it is 'scaled' then it will span both
        quadrants. If it is 'normalized', it will span only the first
        quadrant of the circle. Defaults to 'scaled'.
        dict_mi_parameters (dict, optional): This dictionary contains
        configuration parameters for the calculation of entropy and mutual
        information. Defaults to
        dict(string_library='scipy_sklearn', string_entropy_method='auto',
             bool_discrete_features=False).

    Raises:
        ValueError: The error is raised if string_mid_type is not one of
        the following 'scaled', 'normalized'.

    Returns:
        plotly.graph_objects.Figure: This chart contains the resulting Mutual
        Information diagram.
    """

    list_valid_mid_types = ['normalized', 'scaled']

    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    df_mid = df_calculate_mid_properties(
        df_input, string_reference_model, dict_mi_parameters)

    chart_result = chart_create_diagram(
        df_mid, string_reference_model=string_reference_model,
        string_mid_type=string_mid_type, bool_flag_as_subplot=False,
        string_diagram_type='mid')

    return chart_result


def chart_create_all_diagrams(df_input, string_reference_model,
                              string_corr_method, string_mid_type,
                              dict_mi_parameters):
    """
    chart_create_all_diagrams creates both the Taylor and the Mutual
    Information diagrams (side-by-side) according to the df_input argument
    where models are placed in columns and rows contain model predictions.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
        model prediction in rows. It is used to calculate relevant statistical
        information and information theory properties.
        string_reference_model (str): This string contains the name of the
        model present in the df_input argument (as a column) which can be
        considered as a reference point in the final diagram. This is often
        the ground truth.
        string_corr_method (str, optional): This string contains the name of
        the method to be used when calculating the correlation. Defaults to
        'pearson'.
        string_mid_type (str, optional): This string contains the type of the
        Mutual Information diagram. If it is 'scaled' then it will span both
        quadrants. If it is 'normalized', it will span only the first
        quadrant of the circle. Defaults to 'scaled'.
        dict_mi_parameters (dict, optional): This dictionary contains
        configuration parameters for the calculation of entropy and mutual
        information. Defaults to
        dict(string_library='scipy_sklearn', string_entropy_method='auto',
             bool_discrete_features=False).

    Raises:
        ValueError: The error is raised if string_mid_type is not one of
        the following 'scaled', 'normalized'.

    Returns:
        plotly.graph_objects.Figure: This chart contains the both the Taylor
        and the Mutual Information diagrams side-by-side.
    """

    list_valid_mid_types = ['normalized', 'scaled']

    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    string_combined_chart_title = "Taylor Diagram and Mutual Information Diagram" # noqa
    string_angular_title_td = 'Correlation'

    if string_mid_type == 'scaled':
        string_angular_title_mid = 'Scaled Mutual Information'
    else:
        string_angular_title_mid = 'Normalized Mutual Information'

    df_all = df_calculate_all_properties(
        df_input=df_input, string_reference_model=string_reference_model,
        string_corr_method=string_corr_method,
        dict_mi_parameters=dict_mi_parameters)

    chart_result = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'polar'}]*2],
        subplot_titles=(string_angular_title_td,  string_angular_title_mid))

    chart_result = chart_create_diagram(
        df_all, string_reference_model=string_reference_model,
        bool_flag_as_subplot=True, chart_result_upper=chart_result,
        string_diagram_type='taylor')

    chart_result = chart_create_diagram(
        df_all, string_reference_model=string_reference_model,
        string_mid_type=string_mid_type, bool_flag_as_subplot=True,
        chart_result_upper=chart_result, string_diagram_type='mid')

    chart_result.update_annotations(
        yshift=10, font_color=STRING_LABEL_TITLE_COLOR)
    chart_result.update_layout(
        title=dict(
            text=string_combined_chart_title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=18,
                color=STRING_LABEL_TITLE_COLOR)))

    return chart_result
