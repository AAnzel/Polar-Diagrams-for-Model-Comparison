import math
import pandas as pd
import numpy as np

from npeet import entropy_estimators
from sklearn.metrics import mean_squared_error
from scipy.stats import differential_entropy
from sklearn.feature_selection import mutual_info_regression

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html?highlight=entropy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html
# https://github.com/paulbrodersen/entropy_estimators/
# https://github.com/gregversteeg/NPEET <- USE THIS ONE FIRST !!!!!!!!!!!!!!!!
# https://github.com/BiuBiuBiLL/NPEET_LNC <- And this one second (an improvement) # noqa


string_background_color = '#F8F8F8'
string_grid_color = '#C0C0C0'
string_label_title_color = '#404040'


def calculate_td_properties(df_input, string_reference_feature,
                            string_method='pearson'):

    list_all_features = df_input.columns.to_list()

    # Initialize dict
    dict_result = {}
    for string_one_feature in list_all_features:
        dict_result[string_one_feature] = []

    for string_one_feature in list_all_features:
        # Calculating standard deviations
        dict_result[string_one_feature].append(
            df_input[string_one_feature].std(ddof=0))

        # Calculating Pearson's correlation
        dict_result[string_one_feature].append(
            df_input[string_reference_feature].corr(
                df_input[string_one_feature], method=string_method))

        # Calculate arccos of Pearson's correlation
        dict_result[string_one_feature].append(math.degrees(
            math.acos(dict_result[string_one_feature][-1])))

        # Calculating RMS
        dict_result[string_one_feature].append(
            mean_squared_error(
                df_input[string_reference_feature],
                df_input[string_one_feature], squared=False))

    for string_one_feature in list_all_features:
        # Calculating correlation using RMS formula
        # dict_result[string_one_feature].append(
        #    (dict_result[string_one_feature][0]**2 + dict_result[string_reference_feature][0]**2 - dict_result[string_one_feature][3]**2) / (2 * dict_result[string_one_feature][0] * dict_result[string_reference_feature][0]) # noqa
        # )

        # Calculating the angle using calculated correlation
        # dict_result[string_one_feature].append(math.degrees(
        #    math.acos(dict_result[string_one_feature][-1])))

        # Normalizing the RMS as in the paper
        dict_result[string_one_feature].append(
            dict_result[string_one_feature][3] /
            dict_result[string_reference_feature][0])

        # Calculating normalized standard deviation
        dict_result[string_one_feature].append(
            dict_result[string_one_feature][0] / dict_result[
                string_reference_feature][0])

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        # columns=['STD', 'Correlation', 'Angle', 'RMS', 'Calculated_Corr',
        #         'Calculated_Angle', 'Normalized_RMS', 'Normalized_STD']
        columns=['STD', 'Correlation', 'Angle', 'RMS', 'Normalized_RMS',
                 'Normalized_STD'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def list_adapt_to_npeet(list_input):
    return [[i] for i in list_input]


def calculate_mid_properties(df_input, string_reference_feature,
                             string_library):

    list_all_features = df_input.columns.to_list()

    list_adapted_npeet_reference = list_adapt_to_npeet(
        df_input[string_reference_feature])

    # Initialize dict
    dict_result = {}
    for string_one_feature in list_all_features:
        dict_result[string_one_feature] = []

    # TODO: Entropies are negative often when using default parameters
    # That is causing an error when calculating angles
    # Try to find better default parameters so it doesn't happen
    for string_one_feature in list_all_features:
        list_adapted_npeet_one = list_adapt_to_npeet(
            df_input[string_one_feature])

        if string_library == 'scipy_sklearn':
            # Calculate entropies
            # 0 in the list
            dict_result[string_one_feature].append(
                differential_entropy(df_input[string_one_feature], base=2))
            # dict_result[string_one_feature].append(
            #    mutual_info_regression(
            #        df_input[string_one_feature].to_numpy().reshape(-1, 1),
            #        df_input[string_one_feature],
            #        discrete_features=False)[0])

            # Calculate mutual informations against the reference feature
            # 1 in the liststring_angular_column
            dict_result[string_one_feature].append(
                mutual_info_regression(
                    df_input[string_reference_feature].to_numpy().reshape(
                        -1, 1),
                    df_input[string_one_feature],
                    discrete_features=False)[0])

        elif string_library == 'npeet':
            # Calculate entropies
            # 0 in the list
            dict_result[string_one_feature].append(
                entropy_estimators.entropy(list_adapted_npeet_one))

            # Calculate mutual informations against the reference feature
            # 1 in the liststring_angular_column
            dict_result[string_one_feature].append(
                entropy_estimators.mi(list_adapted_npeet_reference,
                                      list_adapted_npeet_one))

        else:
            print('BAD LIB ARGUMENT')
            return None

    for string_one_feature in list_all_features:
        # Calculating fixed MI from equation 17 from the paper
        # I(X,Y) = I~(X,Y) * (H(X) / I~(X,X)) where I~ is MI calculated using
        # some estimation method. This MI is used for every calculation
        # afterwards
        # 2 in the list
        dict_result[string_one_feature].append(
            dict_result[string_one_feature][1] *
            (dict_result[string_reference_feature][0] /
             dict_result[string_reference_feature][1])
        )

        # Calculate scaled entropies
        # 3 in the list
        dict_result[string_one_feature].append(
            dict_result[string_one_feature][0] /
            dict_result[string_reference_feature][0]
        )

        # Calculate normalized mutual information according to the paper
        # NMI(X,Y) = I(X,Y) / sqrt(H(X) * H(Y))
        # This has to be modified because H(X)*H(Y) can be negative for
        # differential entropies. NMI is not used at all for the MI chart
        # that spans two quadrants
        # 4 in the list
        float_product = dict_result[string_reference_feature][0] *\
            dict_result[string_one_feature][0]

        if float_product < 0:
            dict_result[string_one_feature].append(1)

        else:
            dict_result[string_one_feature].append(
                dict_result[string_one_feature][2] / math.sqrt(float_product))

        # Calculate arccos of normalized mutual information according to the
        # paper arccos(NMI(X,Y))
        # 5 in the list
        dict_result[string_one_feature].append(
            math.degrees(math.acos(dict_result[string_one_feature][4])))

    for string_one_feature in list_all_features:
        ######################################################################
        # This part is for the chart that spans two quadrants

        # First calculate joint entropies by using equation 10 from the paper
        # I(X,Y) = H(X) + H(Y) - H(X,Y) => H(X,Y) = H(X) + H(Y) - I(X,Y)
        # 6 in the list
        dict_result[string_one_feature].append(
            dict_result[string_reference_feature][0] +
            dict_result[string_one_feature][0] -
            dict_result[string_one_feature][2]
        )

        # Calculate scaled mutual information according to the paper
        # SMI(X,Y) = I(X,Y) * ((H(X,Y) / (H(X)*H(Y)))) (equation 15)
        # 7 in the list
        smi_x_y = dict_result[string_one_feature][2] * (
            dict_result[string_one_feature][6] /
            (dict_result[string_reference_feature][0]
             * dict_result[string_one_feature][0]))

        dict_result[string_one_feature].append(smi_x_y)

        # Calculate arccos of biased scaled mutual information according to the
        # paper arccos(c(X,Y)) where c(X,Y) = 2*SMI(X,Y) - 1
        # 8 in the list
        float_c_x_y = 2*smi_x_y - 1

        if float_c_x_y > 1:
            dict_result[string_one_feature].append(
                math.degrees(math.acos(1)))
        elif float_c_x_y < -1:
            dict_result[string_one_feature].append(
                math.degrees(math.acos(-1)))
        else:
            dict_result[string_one_feature].append(
                math.degrees(math.acos(float_c_x_y)))

        # Calculate root entropy
        # 9 in the list
        if dict_result[string_one_feature][0] >= 0:
            float_root_entropy = math.sqrt(dict_result[string_one_feature][0])
        else:
            float_root_entropy = -1

        dict_result[string_one_feature].append(float_root_entropy)
        ######################################################################

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        columns=['Entropy', 'Mutual_information', 'Fixed_MI', 'Scaled_entropy',
                 'Normalized_MI', 'Angle_NMI', 'Joint_entropies', 'Scaled_MI',
                 'Angle_SMI', 'Root_Entropy'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def df_calculate_all_properties(df_input, string_reference_feature,
                                string_library, string_method):
    string_method = 'pearson'
    string_library = 'scipy_sklearn'

    df_td = calculate_td_properties(df_input, string_reference_feature,
                                    string_method=string_method)
    df_mid = calculate_mid_properties(df_input, string_reference_feature,
                                      string_library=string_library)

    return df_td.merge(df_mid, on='Model', how='inner')


def chart_create_diagram(df_input, string_reference_feature,
                         string_mid_type='scaled', bool_flag_as_subplot=False,
                         chart_result_upper=None,
                         string_diagram_type='taylor'):

    # General properties
    list_color_scheme = None
    string_tooltip_label_0 = 'Model'
    int_number_of_models = len(df_input[string_tooltip_label_0].to_list())

    if int_number_of_models <= 9:
        list_color_scheme = px.colors.qualitative.Set1
    else:
        list_color_scheme = px.colors.qualitative.Light24

    np_tmp = np.array(
        [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

    if string_diagram_type == 'taylor':
        bool_show_legend = True
        string_radial_column = 'STD'
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

    elif string_diagram_type == 'mid':
        bool_show_legend = False
        int_subplot_column_number = 2

        if string_mid_type == 'scaled':
            string_angular_column = 'Angle_SMI'
            string_radial_column = 'Entropy'
            string_angular_column_label = 'Scaled Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Scaled_MI'
            bool_only_half = False
            np_angular_labels = np.concatenate((-np_tmp[:0:-1], np_tmp))
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))
            np_angular_labels = np.round((np_angular_labels + 1) / 2, 3)

        elif string_mid_type == 'normalized':
            string_angular_column = 'Angle_NMI'
            string_radial_column = 'Root_Entropy'
            string_angular_column_label = 'Normalized Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Normalized_MI'
            bool_only_half = True
            np_angular_labels = np_tmp
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))

        else:
            # TODO: Raise an error
            print('Type has to be either "scaled" or "normalized"')
            return None

    else:
        # TODO: Raise an error
        return None

    int_max_angle = 90 if bool_only_half else 180
    float_max_r = df_input[string_radial_column].max() +\
        df_input[string_radial_column].mean()

    if bool_flag_as_subplot is True:
        chart_result = chart_result_upper
    else:
        chart_result = go.Figure()

    # TODO: Change tooltip information to show more meaningful info
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
        bgcolor=string_background_color,
        radialaxis=dict(
            range=[0, float_max_r],
            griddash='dot',
            gridcolor=string_grid_color,
            tickcolor=string_label_title_color,
            tickfont=dict(color=string_label_title_color),
            layer='below traces',
            title=dict(
                text=string_radial_column,
                font=dict(
                    color=string_label_title_color))),
        angularaxis=dict(
            direction="counterclockwise",
            tickvals=np_angular_ticks,
            ticktext=np_angular_labels,
            tickcolor=string_label_title_color,
            tickfont=dict(color=string_label_title_color),
            griddash='dot',
            gridcolor=string_grid_color,
            layer='below traces'))

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
                        np_tooltip_data[tmp_model_int]] * int_number_of_models,
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
                        np_tooltip_data[tmp_model_int]] * int_number_of_models,
                    hovertemplate=string_tooltip_hovertemplate,
                    marker=dict(
                        color=list_color_scheme[tmp_model_int])))

    if bool_flag_as_subplot is True:
        if string_diagram_type == 'taylor':
            chart_result.update_layout(
                polar=dict_polar_chart,
                height=600,
                showlegend=True)

        else:
            chart_result.update_layout(
                polar2=dict_polar_chart,
                height=600,
                showlegend=True)

    else:
        chart_result.update_layout(
            polar=dict_polar_chart,
            height=600,
            title=dict(
                text=string_angular_column_label,
                x=0.494,
                y=0.9,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color=string_label_title_color)))

    return chart_result


def chart_create_all_diagrams(df_input, string_reference_feature,
                              string_td_method, string_mid_type,
                              string_mid_library='scipy_sklearn'):

    string_combined_chart_title = "Taylor Diagram and Mutual Information Diagram" # noqa

    string_angular_title_td = 'Correlation'
    if string_mid_type == 'scaled':
        string_angular_title_mid = 'Scaled Mutual Information'
    elif string_mid_type == 'normalized':
        string_angular_title_mid = 'Normalized Mutual Information'
    else:
        # TODO: Raise an error
        print('Type has to be either "scaled" or "normalized"')
        return None

    df_all = df_calculate_all_properties(
        df_input=df_input, string_reference_feature=string_reference_feature,
        string_method=string_td_method, string_library=string_mid_library)

    chart_result = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'polar'}]*2],
        subplot_titles=(string_angular_title_td,  string_angular_title_mid))

    chart_result = chart_create_diagram(
        df_all, string_reference_feature=string_reference_feature,
        bool_flag_as_subplot=True, chart_result_upper=chart_result,
        string_diagram_type='taylor')

    chart_result = chart_create_diagram(
        df_all, string_reference_feature=string_reference_feature,
        string_mid_type=string_mid_type, bool_flag_as_subplot=True,
        chart_result_upper=chart_result, string_diagram_type='mid')

    chart_result.update_annotations(
        yshift=10, font_color=string_label_title_color)
    chart_result.update_layout(
        title=dict(
            text=string_combined_chart_title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=18,
                color=string_label_title_color)))

    return chart_result
