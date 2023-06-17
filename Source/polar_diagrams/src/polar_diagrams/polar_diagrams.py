import math
import warnings
import pandas as pd
import numpy as np

from scipy.sparse import issparse
from scipy.stats import differential_entropy
from sklearn.utils.validation import check_array, check_X_y
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif

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


_STRING_BACKGROUND_COLOR = '#FFFFFF'
_STRING_GRID_COLOR = '#C0C0C0'
_STRING_LABEL_TITLE_COLOR = '#404040'
_STRING_TICK_COLOR = '#000000'
_STRING_GRID_STYLE = 'solid'
_INT_TICK_WIDTH = 2
_INT_RANDOM_SEED = None
_INT_MARKER_SIZE = 10
_INT_MARKER_LINE_WIDTH = 2
_INT_CHART_WIDTH = 1400
_INT_CHART_HEIGHT = 500
_INT_LEGEND_X = 0
_INT_LEGEND_Y = -0.2
_FLOAT_MARKER_OPACITY = 0.60
_FLOAT_LEGEND_BORDER_WIDTH = 0.2

# Note: Color acquired from: https://public.tableau.com/views/TableauColors/ColorsbyHexCode?%3Aembed=y&%3AshowVizHome=no&%3Adisplay_count=y&%3Adisplay_static_image=y # noqa
_LIST_TABLEAU_10 = ['#1f77b4', '#2ca02c', '#7f7f7f', '#8c564b', '#17becf',
                    '#9467bd', '#bcbd22', '#d62728', '#e377c2', '#ff7f0e']
_LIST_TABLEAU_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
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
          model prediction in rows. It is used to calculate relevant
          statistical information.
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

    df_input = _df_check_df_raw_input(df_input)

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

    for string_one_model in list_all_features:
        # Calculating CRMS
        dict_result[string_one_model].append(
            math.sqrt(mean_squared_error(
                df_input[string_reference_model],
                df_input[string_one_model], squared=True) -
                      (np.mean(df_input[string_reference_model]) -
                       np.mean(df_input[string_one_model]))**2))

        # Normalizing the CRMS as in the paper
        dict_result[string_one_model].append(
            dict_result[string_one_model][3] /
            dict_result[string_reference_model][0])

        # Calculating normalized standard deviation
        dict_result[string_one_model].append(
            dict_result[string_one_model][0] / dict_result[
                string_reference_model][0])

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        columns=['Standard Deviation', 'Correlation', 'Angle', 'CRMS',
                 'Normalized CRMS', 'Normalized Standard Deviation'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def _float_calculate_discrete_entropy(list_labels, int_base=2):
    """
    _float_calculate_discrete_entropy calculates Shannon (discrete) entropy of
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


def _dict_check_discrete_models(df_input, string_reference_model,
                                dict_mi_parameters):
    """
    _dict_check_discrete_models checks if
    dict_mi_parameters['discrete_models'] is valid argument as defined in
    scikit-learn library.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
          model prediction in rows. It is used to calculate relevant
          information theory properties
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        dict_mi_parameters (dict): This dictionary contains configuration
          parameters for the calculation of entropy and mutual information.

    Raises:
        ValueError: Raises error if dict_mi_parameters['discrete_models']
        is not valid.

    Returns:
        dict: The dictionary where keys are column names that are different
          than string_reference_model and keys that are boolean and say if that
          model has discrete or continous values. Those keys are validated
          using this function.
    """

    # This check was acquired from https://github.com/scikit-learn/scikit-learn/blob/d949e7f731c99db8a88d16532f5476b52033bf8f/sklearn/feature_selection/_mutual_info.py#L5 # noqa

    X = df_input.drop(string_reference_model, axis=1)
    y = df_input[string_reference_model]
    list_X_columns = X.columns

    if dict_mi_parameters['bool_discrete_reference_model'] is False:
        X, y = check_X_y(
            X, y, accept_sparse="csc",
            y_numeric=not dict_mi_parameters['bool_discrete_reference_model'])
    n_samples, n_features = X.shape

    if isinstance(dict_mi_parameters['discrete_models'], (str, bool)):
        if isinstance(dict_mi_parameters['discrete_models'], str):
            if dict_mi_parameters['discrete_models'] == "auto":
                discrete_models = issparse(X)
            else:
                raise ValueError("Invalid string value for discrete_models.")
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(dict_mi_parameters['discrete_models'])
    else:
        discrete_models = check_array(
            dict_mi_parameters['discrete_models'], ensure_2d=False)
        if discrete_models.dtype != "bool":
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_models] = True
        else:
            discrete_mask = discrete_models

    dict_feature_discrete_mask = dict()
    for int_i, string_one_column_name in enumerate(list_X_columns):
        dict_feature_discrete_mask[
            string_one_column_name] = discrete_mask[int_i]

    return dict_feature_discrete_mask


def df_calculate_mid_properties(df_input, string_reference_model,
                                dict_mi_parameters=dict(
                                    string_entropy_method='auto',
                                    int_mi_n_neighbors=3,
                                    discrete_models='auto',
                                    bool_discrete_reference_model=False,
                                    int_random_state=_INT_RANDOM_SEED)):
    """
    df_calculate_mid_properties caclulates all necessary information theory
    properties for the Mutual Information diagram from the input data set.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
          model prediction in rows. It is used to calculate relevan
          information theory propertiesdict_mi_parameters['discrete_models']
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        dict_mi_parameters (dict, optional): This dictionary contains
          configuration parameters for the calculation of entropy and mutual
          information. Defaults to
          dict(int_mi_n_neighbors=3, string_entropy_method='auto',
               bool_discrete_reference_model=False, discrete_models='auto',
               int_random_state=_INT_RANDOM_SEED).

    Raises:
        ValueError: The error is raised if int_mi_n_neighbors is less or equal
          than zero.
        ValueError: The error is raised if string_entropy_method is not one of
          the following 'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'

    Returns:
        pandas.DataFrame: This dataframe contains model names as indices and
          information theory properties as columns.
    """

    df_input = _df_check_df_raw_input(df_input)

    dict_feature_discrete_mask = _dict_check_discrete_models(
        df_input, string_reference_model, dict_mi_parameters)

    list_valid_parameters = ['int_mi_n_neighbors', 'string_entropy_method',
                             'bool_discrete_reference_model',
                             'int_random_state']
    list_valid_entropy_methods = ['vasicek', 'van es', 'ebrahimi', 'correa',
                                  'auto']
    list_valid_discrete_target = [True, False]
    list_valid_float_types = [np.float64, np.float32, np.double]

    if not all(string_parameter in dict_mi_parameters
               for string_parameter in list_valid_parameters):
        raise ValueError('dict_mi_parameters must contain all of the keys' +
                         ' from this list ' + str(list_valid_parameters))

    if dict_mi_parameters['int_mi_n_neighbors'] <= 0:
        raise ValueError('int_mi_n_neighbors has to be greater than 0.')

    if dict_mi_parameters[
            'string_entropy_method'] not in list_valid_entropy_methods:
        raise ValueError('string_entropy_method is not one of the' +
                         ' following:' + str(list_valid_entropy_methods))

    if dict_mi_parameters[
            'bool_discrete_reference_model'] not in list_valid_discrete_target:
        raise ValueError('bool_discrete_reference_model is not one of the' +
                         ' following:' + str(list_valid_discrete_target))

    list_all_features = df_input.columns.to_list()

    # Initialize dict
    dict_result = {}
    for string_one_model in list_all_features:
        dict_result[string_one_model] = []

    # That is causing an error when calculating angles
    # Try to find better default parameters so it doesn't happen
    for string_one_model in list_all_features:
        # Calculate entropies
        # 0 in the list
        if string_one_model == string_reference_model:
            if dict_mi_parameters['bool_discrete_reference_model']:
                dict_result[string_one_model].append(
                    _float_calculate_discrete_entropy(
                        df_input[string_one_model], int_base=2))
            else:
                if df_input[string_one_model].dtype in list_valid_float_types:
                    dict_result[string_one_model].append(
                        differential_entropy(df_input[string_one_model],
                                             base=2))
                else:
                    raise RuntimeError(
                        'Model named ' + string_one_model +
                        ' is said to be contionus but has values' +
                        ' that are not one of the following type ' +
                        str(list_valid_float_types))
        else:
            if dict_feature_discrete_mask[string_one_model]:
                dict_result[string_one_model].append(
                    _float_calculate_discrete_entropy(
                        df_input[string_one_model], int_base=2))
            else:
                if df_input[string_one_model].dtype in list_valid_float_types:
                    dict_result[string_one_model].append(
                        differential_entropy(df_input[string_one_model],
                                             base=2))
                else:
                    raise RuntimeError(
                        'Model named ' + string_one_model +
                        ' is said to be contionus but has values' +
                        ' that are not one of the following type ' +
                        str(list_valid_float_types))

        if dict_mi_parameters['bool_discrete_reference_model']:
            # Calculate mutual informations against the reference feature
            # 1 in the list
            dict_result[string_one_model].append(
                mutual_info_classif(
                    df_input[string_one_model].to_numpy().reshape(-1, 1),
                    df_input[string_reference_model],
                    random_state=dict_mi_parameters['int_random_state'],
                    discrete_features=dict_mi_parameters['discrete_models']
                    )[0])

        else:
            # Calculate mutual informations against the reference feature
            # 1 in the list
            dict_result[string_one_model].append(
                mutual_info_regression(
                    df_input[string_one_model].to_numpy().reshape(-1, 1),
                    df_input[string_reference_model],
                    random_state=dict_mi_parameters['int_random_state'],
                    discrete_features=dict_mi_parameters['discrete_models']
                    )[0])

    for string_one_model in list_all_features:
        # Calculating fixed MI from equation 17 from the paper
        # I(X,Y) = I~(X,Y) * (H(X) / I~(X,X)) where I~ is MI calculated using
        # some estimation method. This MI is used for every calculation
        # afterwards
        # 2 in the list
        dict_result[string_one_model].append(
            dict_result[string_one_model][1] *
            (dict_result[string_reference_model][0] /
             dict_result[string_reference_model][1]))

        # Calculate normalized (centered) entropies
        # 3 in the list
        dict_result[string_one_model].append(
            dict_result[string_one_model][0] /
            dict_result[string_reference_model][0])

        # Calculate normalized mutual information according to the paper
        # NMI(X,Y) = I(X,Y) / sqrt(H(X) * H(Y))
        # This has to be modified because H(X)*H(Y) can be negative for
        # differential entropies. NMI is not used at all for the MI chart
        # that spans two quadrants
        # 4 in the list
        float_product = dict_result[string_reference_model][0] *\
            dict_result[string_one_model][0]

        if float_product < 0:
            float_product = 1

        float_divide = dict_result[string_one_model][2] /\
            math.sqrt(float_product)

        dict_result[string_one_model].append(float_divide)

        # Calculate arccos of normalized mutual information according to the
        # paper arccos(NMI(X,Y))
        # 5 in the list
        if float_divide > 1:
            dict_result[string_one_model].append(
                math.degrees(math.acos(1)))
        elif float_divide < -1:
            dict_result[string_one_model].append(
                math.degrees(math.acos(-1)))
        else:
            dict_result[string_one_model].append(
                math.degrees(math.acos(float_divide)))

        # Calculate root entropy
        # 6 in the list
        if dict_result[string_one_model][0] >= 0:
            float_root_entropy = math.sqrt(dict_result[string_one_model][0])
        else:
            float_root_entropy = -1

        dict_result[string_one_model].append(float_root_entropy)

    for string_one_model in list_all_features:
        ######################################################################
        # This part is for the chart that spans two quadrants

        # First calculate joint entropies by using equation 10 from the paper
        # I(X,Y) = H(X) + H(Y) - H(X,Y) => H(X,Y) = H(X) + H(Y) - I(X,Y)
        # 7 in the list
        dict_result[string_one_model].append(
            dict_result[string_reference_model][0] +
            dict_result[string_one_model][0] -
            dict_result[string_one_model][2])

        # Calculate scaled mutual information according to the paper
        # SMI(X,Y) = I(X,Y) * ((H(X,Y) / (H(X)*H(Y)))) (equation 15)
        # 8 in the list
        smi_x_y = dict_result[string_one_model][2] * (
            dict_result[string_one_model][-1] /
            (dict_result[string_reference_model][0]
             * dict_result[string_one_model][0]))

        dict_result[string_one_model].append(smi_x_y)

        # Calculate arccos of biased scaled mutual information according to the
        # paper arccos(c(X,Y)) where c(X,Y) = 2*SMI(X,Y) - 1
        # 9 in the list
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

        # Calculate normalized root entropy
        # 10 in the list
        dict_result[string_one_model].append(
            dict_result[string_one_model][6] /
            dict_result[string_reference_model][6])

        ######################################################################

        # Calculate the variation of information (VI) using the equation 11
        # from the paper: VI(X,Y) = H(X) + H(Y) - 2I(X;Y)
        # 11 in the list
        dict_result[string_one_model].append(
            dict_result[string_reference_model][0] +
            dict_result[string_one_model][0] -
            2 * dict_result[string_one_model][2])

        # Calculate the root variation of information (RVI)
        # 12 in the list
        if dict_result[string_one_model][-1] < 0:
            dict_result[string_one_model].append(-1)
        else:
            dict_result[string_one_model].append(
                math.sqrt(dict_result[string_one_model][-1]))

    df_result = pd.DataFrame().from_dict(
        dict_result, orient='index',
        columns=['Entropy', 'Mutual Information', 'Fixed_MI',
                 'Normalized Entropy', 'Normalized MI', 'Angle_NMI',
                 'Root Entropy', 'Joint_entropies', 'Scaled MI',
                 'Angle_SMI', 'Normalized Root Entropy', 'VI', 'RVI'])

    df_result = df_result.reset_index().rename(columns={'index': 'Model'})

    return df_result


def df_calculate_all_properties(df_input, string_reference_model,
                                dict_mi_parameters=dict(
                                    string_entropy_method='auto',
                                    int_mi_n_neighbors=3,
                                    discrete_models='auto',
                                    bool_discrete_reference_model=False,
                                    int_random_state=_INT_RANDOM_SEED),
                                string_corr_method='pearson'):
    """
    df_calculate_all_properties caclulates all necessary statistical and
    information theory properties for the Taylor and Mutual Information diagram
    from the input data set.

    Args:
        df_input (pandas.DataFrame): This dataframe has models in columns and
          model prediction in rows. It is used to calculate relevant
          statistical information and information theory properties.
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        dict_mi_parameters (dict, optional): This dictionary contains
          configuration parameters for the calculation of entropy and mutual
          information. Defaults to
          dict(int_mi_n_neighbors=3, string_entropy_method='auto',
               bool_discrete_reference_model=False, discrete_fetures='auto',
               int_random_state=_INT_RANDOM_SEED).
        string_corr_method (str, optional): This string contains the name of
          the method to be used when calculating the correlation. Defaults to
          'pearson'.

    Returns:
        pandas.DataFrame: This dataframe contains model names as indices and
          statistical and information theory properties as columns.
    """
    df_input = _df_check_df_raw_input(df_input)

    df_td = df_calculate_td_properties(
        df_input, string_reference_model, string_corr_method)
    df_mid = df_calculate_mid_properties(
        df_input, string_reference_model, dict_mi_parameters)

    df_result = df_td.merge(df_mid, on='Model', how='inner')

    _warning_check_identical_model_values(df_result)

    return df_result


def _tuple_hex_to_rgb(string_hex_color):
    """
    _tuple_hex_to_rgb converts color value given in hex format to rgba format
    with float_alpha_opacity opacity (transperancy) value

    Args:
        string_hex_color (str): This argument contains the hex value of the
          color.

    Returns:
        tuple: The return value is a tuple (R, G, B) where R, G, B are
          integer values from 0 to 255.
    """
    return tuple(
        [int(string_hex_color.lstrip('#')[int_i:int_i+2], 16)
         for int_i in (0, 2, 4)])


def _dict_calculate_model_colors(list_model_names, string_reference_model,
                                 int_number_of_datasets):
    """
    _dict_calculate_model_colors defines an RGBA color for each model parsed in
    the list_model_names. The reference model is always black.

    Args:
        list_model_names (list): This list contains the strings of model names.
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        int_number_of_datasets (int): This argument contains the number of
          datasets. This argument is important if we have two datasets where
          the second one is another version of the first dataset.

    Returns:
        dict: The function returns a dictionary where model strings are keys,
          and values are RGBA tuples.
    """
    int_number_of_models = len(list_model_names)

    if int_number_of_models <= 9:
        list_color_scheme = _LIST_TABLEAU_10
    else:
        list_color_scheme = _LIST_TABLEAU_20
    int_num_discrete_colors = len(list_color_scheme)

    dict_result = dict()

    for int_i in range(int_number_of_datasets):
        for int_j, string_model_name in enumerate(list_model_names):
            if string_model_name == string_reference_model:
                string_hex_color = '#000000'
            else:
                string_hex_color = list_color_scheme[
                    int_j % int_num_discrete_colors]

            if string_model_name not in dict_result:
                dict_result[string_model_name] = dict()

            tuple_rgb_color = _tuple_hex_to_rgb(string_hex_color)
            dict_result[string_model_name][int_i] = (
                int(tuple_rgb_color[0]), int(tuple_rgb_color[1]),
                int(tuple_rgb_color[2]))

    return dict_result


def _warning_check_identical_model_values(df_input):
    """
    _warning_check_identical_model_values checks if there are models that have
    identical statistical or information theory properties. This check is
    performed so that the user is warned if the diagrams contain circles that
    fully overlap.

    Args:
        df_input (pandas.DataFrame): This dataframe contains both statistical
          and information theory properties as calculated in
          df_calculate_all_properties. The dataframe contains columns like:
          Model, Entropy, Standard Deviation, etc.

    Returns:
        None: None
    """

    df_taylor_result = df_input[
        df_input.duplicated(
            subset=['Standard Deviation', 'Correlation'], keep=False)]

    df_mi_result = df_input[
        df_input.duplicated(subset=['Entropy', 'Scaled MI'], keep=False)]

    if df_taylor_result.shape[0] != 0:
        warnings.warn('WARNING: The following models have the same Standard' +
                      ' Deviation and Correlation: ' +
                      str(df_taylor_result['Model'].to_list()) +
                      '.\nThe resulting Taylor diagram will have overlapping' +
                      ' marks (circles).', RuntimeWarning)
    if df_mi_result.shape[0] != 0:
        warnings.warn('WARNING: The following models have the same Entropy' +
                      ' and Scaled Mutual Information: ' +
                      str(df_mi_result['Model'].to_list()) +
                      '.\nThe resulting Mutual Information diagram will have' +
                      ' overlapping marks (circles).', RuntimeWarning)

    return None


def _chart_create_diagram(list_df_input, string_reference_model,
                          string_mid_type, string_diagram_type,
                          bool_normalized_measures, bool_flag_as_subplot=False,
                          chart_result_upper=None):
    """
    _chart_create_diagram is a general function that creates both the Taylor
    and the Mutual Information diagrams according to the passed argument.

    Args:
        list_df_input (list): This list contains one or two dataframes which
          have models in columns and model prediction in rows. If parsed as a
          pd.DataFrame() object, it is considered as a first and only element
          of the list. Each one of these dataframes is used to calculate
          relevant statistical information and information theory properties.
          If the list contains two elements, both dataframes need to have the
          same set of columns. If the second dataframe contains only one row,
          then this dataframe is considered to contain a property that is
          encoded as using size of the marker of the resulting diagrams. If the
          second dataframe contains multiple rows, it is then considered to be
          a second version of the first dataframe in the list. This is then
          encoded using solid borders around circle marks in the resulting
          diagrams.
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        string_mid_type (str): This string contains the type of the
          Mutual Information diagram. If it is 'scaled' then it will span both
          quadrants. If it is 'normalized', it will span only the first
          quadrant of the circle.
        string_diagram_type (str): This string contains the type of the diagram
          that has to be created.
        bool_normalized_measures (bool, optional): This boolean parameter is
          used to determine if the passed chart should have normalized entropy
          and STD values or not. If it is False, then real entropy and STD
          values are used for the radial axis. If it is True, normalized values
          are used. Defaults to False.
        bool_flag_as_subplot (bool, optional): This boolean parameter is used
          to determine if the passed chart is a subplot or not. If it is False,
          then only one diagram is created. If it is True, both Taylor and
          Mutual Information diagrams are created and placed side-by-side.
          Defaults to False.
        chart_result_upper (plotly.graph_objects.Figure, optional): This chart
          is not None only if both diagrams have to be created. It contains the
          blank canvas (1 row, 2 columns) for both objects. Defaults to None.

    Raises:
        TypeError: The error is raised if bool_notmalized_measures is not one
          of bool type.
        ValueError: The error is raised if string_diagram_type is not one of
          the following 'taylor', 'mid'
        ValueError: The error is raised if string_mid_type is not one of
          the following 'scaled', 'normalized'

    Returns:
        plotly.graph_objects.Figure: This chart contains the resulting Taylor
          or Mutual Information diagram.
    """

    list_valid_diagram_types = ['taylor', 'mid']
    list_valid_mid_types = ['scaled', 'normalized', None]

    if not isinstance(bool_normalized_measures, bool):
        raise TypeError('bool_normalized_measures must be of bool type')

    if string_diagram_type not in list_valid_diagram_types:
        raise ValueError('string_diagram_type not in ' +
                         str(list_valid_diagram_types))
    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    # General properties
    string_tooltip_label_0 = 'Model'
    int_number_of_models = len(
        list_df_input[0][string_tooltip_label_0].to_list())
    int_number_of_datasets = len(list_df_input)

    np_tmp = np.array(
        [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

    if string_diagram_type == 'taylor':
        bool_show_legend = True
        if bool_normalized_measures:
            string_radial_column = 'Normalized Standard Deviation'
        else:
            string_radial_column = 'Standard Deviation'
        string_angular_column = 'Angle'
        string_angular_column_label = 'Correlation'
        string_tooltip_label_1 = string_radial_column
        string_tooltip_label_2 = string_angular_column_label
        string_tooltip_label_3 = 'CRMS'
        bool_only_half = True if (
            list_df_input[0][string_angular_column] <= 90).all() else False
        int_subplot_column_number = 1
        np_angular_labels = np_tmp if bool_only_half else np.concatenate(
            (-np_tmp[:0:-1], np_tmp))
        np_angular_ticks = np.degrees(np.arccos(np_angular_labels))

    else:
        bool_show_legend = False
        int_subplot_column_number = 2

        if string_mid_type == 'scaled':
            string_angular_column = 'Angle_SMI'
            if bool_normalized_measures:
                string_radial_column = 'Normalized Entropy'
            else:
                string_radial_column = 'Entropy'
            string_angular_column_label = 'Scaled Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Scaled MI'
            string_tooltip_label_3 = 'VI'
            # TODO: Improve this below. We might have to check the second
            # TODO: version data set as well, since those models might go
            # TODO: on the other quarter of the circle
            bool_only_half = bool_only_half = True if (
                list_df_input[0][string_angular_column] <= 90).all() else False
            np_angular_labels = np.concatenate((-np_tmp[:0:-1], np_tmp))
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))
            np_angular_labels = np.round((np_angular_labels + 1) / 2, 3)

        else:
            string_angular_column = 'Angle_NMI'
            if bool_normalized_measures:
                string_radial_column = 'Normalized Root Entropy'
            else:
                string_radial_column = 'Root Entropy'
            string_angular_column_label = 'Normalized Mutual Information'
            string_tooltip_label_1 = string_radial_column
            string_tooltip_label_2 = 'Normalized MI'
            string_tooltip_label_3 = 'RVI'
            bool_only_half = True
            np_angular_labels = np_tmp
            np_angular_ticks = np.degrees(np.arccos(np_angular_labels))

    string_scalar_column = 'Scalar'
    int_max_angle = 90 if bool_only_half else 180
    float_max_r = list_df_input[0][string_radial_column].max() +\
        list_df_input[0][string_radial_column].mean()

    if int_number_of_datasets == 2 and list_df_input[1].shape[1] != 2:
        # We check if it is NOT a scenario where user inputed dataframe with
        # one row (with scalar information). That dataframe is then modified to
        # look like this:
        # Model    Scalar
        # 0    y1  0.388677
        # 1    y2  0.271349
        float_max_r = max(
            float_max_r,
            list_df_input[1][string_radial_column].max() +
            list_df_input[1][string_radial_column].mean())

    if bool_flag_as_subplot:
        chart_result = chart_result_upper
    else:
        chart_result = go.Figure()

    string_tooltip_suffix = ''

    dict_polar_chart = dict(
        sector=[0, int_max_angle],
        bgcolor=_STRING_BACKGROUND_COLOR,
        radialaxis=dict(
            range=[0, float_max_r],
            griddash=_STRING_GRID_STYLE,
            gridcolor=_STRING_GRID_COLOR,
            ticks='outside',
            tickwidth=_INT_TICK_WIDTH,
            tickcolor=_STRING_TICK_COLOR,
            tickfont=dict(color=_STRING_LABEL_TITLE_COLOR),
            showline=True,
            linecolor=_STRING_TICK_COLOR,
            layer='below traces',
            title=dict(
                text='<br>' + string_radial_column,
                font=dict(
                    color=_STRING_LABEL_TITLE_COLOR,
                    size=16))),
        angularaxis=dict(
            direction="counterclockwise",
            ticks='outside',
            tickwidth=_INT_TICK_WIDTH,
            tickvals=np_angular_ticks,
            ticktext=np_angular_labels,
            tickcolor=_STRING_TICK_COLOR,
            tickfont=dict(color=_STRING_LABEL_TITLE_COLOR),
            griddash=_STRING_GRID_STYLE,
            gridcolor=_STRING_GRID_COLOR,
            showline=True,
            linecolor=_STRING_TICK_COLOR,
            layer='below traces'))
    dict_legend = dict(
        title=dict(text=string_tooltip_label_0),
        font=dict(
            color=_STRING_LABEL_TITLE_COLOR),
        bgcolor=_STRING_BACKGROUND_COLOR,
        bordercolor=_STRING_GRID_COLOR,
        borderwidth=_FLOAT_LEGEND_BORDER_WIDTH,
        orientation='h',
        x=_INT_LEGEND_X,
        y=_INT_LEGEND_Y)

    dict_model_colors = _dict_calculate_model_colors(
        list_df_input[0][string_tooltip_label_0].to_list(),
        string_reference_model,
        int_number_of_datasets)

    # Calculate marker sizes if we have a scalar dataset
    # And also check if we have a second version dataset so that we have a
    # nicer tooltip information
    # int_dataset_option can be one of [0, 1, 2] where:
    # 0 = we have one dataset, 1 = two datasets with the second one holding
    # scalar values, 2 = two datasets with the second one being the second
    # version dataset
    if int_number_of_datasets == 2 and list_df_input[1].shape[1] == 2:
        np_first_row = list_df_input[1][string_scalar_column].to_numpy()
        np_scaled_values = (np_first_row - np.min(np_first_row)) /\
            (np.max(np_first_row) - np.min(np_first_row))

        dict_model_marker_sizes = dict(zip(
            list_df_input[0][string_tooltip_label_0],
            np_scaled_values + 1))

        int_dataset_option = 1

    elif int_number_of_datasets == 1:
        int_dataset_option = 0
    else:
        int_dataset_option = 2

    for int_i, df_input in enumerate(list_df_input):
        list_tooltip_columns = [
            string_tooltip_label_0, string_tooltip_label_1,
            string_tooltip_label_2, string_tooltip_label_3]

        if int_dataset_option == 0:
            string_tooltip_suffix = '<br>'
        elif int_dataset_option == 1:
            df_input = list_df_input[0].merge(
                list_df_input[1], on=string_tooltip_label_0, how='inner')
            list_tooltip_columns.append(string_scalar_column)
            string_tooltip_suffix = '<br>' + string_scalar_column +\
                ': %{customdata[4]:.3f}<br>'
        else:
            string_tooltip_suffix = ' - <b>Version ' + str(int_i) +\
                '</b><br>'

        np_tooltip_data = list(
            df_input[list_tooltip_columns].to_numpy())

        string_tooltip_hovertemplate = (
            string_tooltip_label_0 + ': %{customdata[0]}' +
            string_tooltip_suffix +
            string_tooltip_label_1 + ': %{customdata[1]:.3f}<br>' +
            string_tooltip_label_2 + ': %{customdata[2]:.3f}<br>' +
            'Reference distance (' + string_tooltip_label_3 +
            '): %{customdata[3]:.3f}<br>' + '<extra></extra>')

        for tmp_r, tmp_angle, tmp_model_int, tmp_model in zip(
                df_input[string_radial_column],
                df_input[string_angular_column],
                pd.factorize(df_input[string_tooltip_label_0])[0],
                df_input[string_tooltip_label_0]):

            string_marker_color = dict_model_colors[tmp_model][int_i]
            # Do not show the legend for the scalar values
            bool_show_legend = False if int_i == 1 else bool_show_legend

            if int_i == 1 and int_dataset_option == 1:
                # The marker type for the scalar second dataset
                # We add aditional marker with only border
                string_marker_color = dict_model_colors[tmp_model][0]
                dict_marker = dict(
                    line=dict(
                        color='rgba' + str(string_marker_color + (1,)),
                        width=_INT_MARKER_LINE_WIDTH),
                    color='rgba' + str(string_marker_color + (0,)),
                    size=_INT_MARKER_SIZE * dict_model_marker_sizes[tmp_model])
            elif int_i == 0 and int_dataset_option == 2:
                dict_marker = dict(
                    color='rgba' + str(
                        string_marker_color + (_FLOAT_MARKER_OPACITY,)),
                    size=_INT_MARKER_SIZE + _INT_MARKER_LINE_WIDTH)
            else:
                dict_marker = dict(
                    line=dict(
                        color='rgba' + str(string_marker_color + (1,)),
                        width=_INT_MARKER_LINE_WIDTH),
                    color='rgba' + str(
                        string_marker_color + (_FLOAT_MARKER_OPACITY,)),
                    size=_INT_MARKER_SIZE)

            if bool_flag_as_subplot:
                chart_result.add_trace(
                    go.Scatterpolar(
                        name=tmp_model,
                        r=[tmp_r],
                        theta=[tmp_angle],
                        mode='markers',
                        legendgroup=tmp_model,
                        showlegend=bool_show_legend,
                        # This below has to be done like this, since we add
                        # traces one by one
                        customdata=[np_tooltip_data[tmp_model_int]] *\
                        int_number_of_models,

                        hovertemplate=string_tooltip_hovertemplate,
                        hoverlabel=dict(
                            bgcolor=_STRING_BACKGROUND_COLOR,
                            bordercolor='rgba' + str(
                                string_marker_color + (1,)),
                            font=dict(
                                color=_STRING_TICK_COLOR)),
                        marker=dict_marker),
                    row=1,
                    col=int_subplot_column_number)

            else:
                chart_result.add_trace(
                    go.Scatterpolar(
                        name=tmp_model,
                        r=[tmp_r],
                        theta=[tmp_angle],
                        mode='markers',
                        # This below has to be done like this, since we add
                        # traces one by one
                        customdata=[np_tooltip_data[tmp_model_int]] *\
                        int_number_of_models,

                        hovertemplate=string_tooltip_hovertemplate,
                        hoverlabel=dict(
                            bgcolor=_STRING_BACKGROUND_COLOR,
                            bordercolor='rgba' + str(
                                string_marker_color + (1,)),
                            font=dict(
                                color=_STRING_TICK_COLOR)),
                        marker=dict_marker))

    if bool_flag_as_subplot:
        if string_diagram_type == 'taylor':
            chart_result.update_layout(
                polar=dict_polar_chart,
                legend=dict_legend,
                width=_INT_CHART_WIDTH,
                height=_INT_CHART_HEIGHT + 120*round(int_number_of_models/10),
                showlegend=True)

        else:
            chart_result.update_layout(
                polar2=dict_polar_chart,
                legend=dict_legend,
                width=_INT_CHART_WIDTH,
                height=_INT_CHART_HEIGHT + 120*round(int_number_of_models/10),
                showlegend=True)

    else:
        chart_result.update_layout(
            polar=dict_polar_chart,
            legend=dict_legend,
            width=_INT_CHART_WIDTH/2 + 100,
            height=_INT_CHART_HEIGHT + 80*round(int_number_of_models/4),
            title=dict(
                text=string_angular_column_label,
                x=0.5,
                xref='paper',
                yref='paper',
                xanchor='auto',
                yanchor='auto',
                font=dict(
                    size=16,
                    color=_STRING_LABEL_TITLE_COLOR)))

    return chart_result


def chart_create_taylor_diagram(list_df_input, string_reference_model,
                                string_corr_method='pearson',
                                bool_normalized_measures=False):
    """
    chart_create_taylor_diagram creates the Taylor diagram according to the
    list_df_input argument where models are placed in columns and rows contain
    model predictions.

    Args:
        list_df_input (list): This list contains one or two dataframes which
          have models in columns and model prediction in rows. If parsed as a
          pd.DataFrame() object, it is considered as a first and only element
          of the list. Each one of these dataframes is used to calculate
          relevant statistical information and information theory properties.
          If the list contains two elements, both dataframes need to have the
          same set of columns. If the second dataframe contains only one row,
          then this dataframe is considered to contain a property that is
          encoded as using size of the marker of the resulting diagrams. If the
          second dataframe contains multiple rows, it is then considered to be
          a second version of the first dataframe in the list. This is then
          encoded using solid borders around circle marks in the resulting
          diagrams.
        string_reference_model (str): This string contains the name of the
          model present in the df_input argument (as a column) which can be
          considered as a reference point in the final diagram. This is often
          the ground truth.
        string_corr_method (str, optional): This string contains the name of
          the method to be used when calculating the correlation. Defaults to
          'pearson'.
        bool_normalized_measures (bool, optional): This boolean parameter is
          used to determine if the passed chart should have normalized entropy
          and STD values or not. If it is False, then real entropy and STD
          values are used for the radial axis. If it is True, normalized values
          are used. Defaults to False.

    Raises:
        ValueError: The error is raised if the string_corr_method is not one of
          the following 'pearson', 'kendall', 'spearman'.
        TypeError: The error is raised if bool_notmalized_measures is not one
          of bool type.

    Returns:
       plotly.graph_objects.Figure: This chart contains the resulting Taylor
         diagram.
    """
    list_df_input = _list_check_list_df_input(list_df_input)
    list_df_td = []
    list_valid_corr_methods = ['pearson', 'kendall', 'spearman']

    if string_corr_method not in list_valid_corr_methods:
        raise ValueError('string_corr_method is not one of the following:' +
                         str(list_valid_corr_methods))

    if not isinstance(bool_normalized_measures, bool):
        raise TypeError('bool_normalized_measures must be of bool type')

    for int_i, df_input in enumerate(list_df_input):
        # We have to check if the secons pandas.DataFrame has one or multiple
        # rows. If it has one, we encode that property using the size of the
        # mark of the resulting diagram. If it has multiple rows, we need to
        # calculate all information for that dataframe and we visualize both
        # using arrows in the resulting diagram
        if int_i == 1 and df_input.shape[0] == 1:
            # We don't calculate properties when we have one row
            # We consider that a scalar value which we encode using mark size
            list_df_td.append(
                df_input.melt().rename(
                    columns={'variable': 'Model', 'value': 'Scalar'}))
            continue

        df_td = df_calculate_td_properties(
            df_input, string_reference_model, string_corr_method)
        list_df_td.append(df_td)

    chart_result = _chart_create_diagram(
        list_df_td, string_reference_model=string_reference_model,
        string_mid_type=None, bool_flag_as_subplot=False,
        string_diagram_type='taylor',
        bool_normalized_measures=bool_normalized_measures)

    return chart_result


def chart_create_mi_diagram(list_df_input, string_reference_model,
                            string_mid_type='scaled',
                            dict_mi_parameters=dict(
                                string_entropy_method='auto',
                                int_mi_n_neighbors=3,
                                discrete_models='auto',
                                bool_discrete_reference_model=False,
                                int_random_state=_INT_RANDOM_SEED),
                            bool_normalized_measures=False):
    """
    chart_create_mi_diagram creates the Mutual Information diagram according
    to the list_df_input argument where models are placed in columns and rows
    contain model predictions.

    Args:
        list_df_input (list): This list contains one or two dataframes which
          have models in columns and model prediction in rows. If parsed as a
          pd.DataFrame() object, it is considered as a first and only element
          of the list. Each one of these dataframes is used to calculate
          relevant statistical information and information theory properties.
          If the list contains two elements, both dataframes need to have the
          same set of columns. If the second dataframe contains only one row,
          then this dataframe is considered to contain a property that is
          encoded as using size of the marker of the resulting diagrams. If the
          second dataframe contains multiple rows, it is then considered to be
          a second version of the first dataframe in the list. This is then
          encoded using solid borders around circle marks in the resulting
          diagrams.
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
          dict(int_mi_n_neighbors=3, string_entropy_method='auto',
               bool_discrete_reference_model=False, discrete_fetures='auto',
               int_random_state=_INT_RANDOM_SEED).
        bool_normalized_measures (bool, optional): This boolean parameter is
          used to determine if the passed chart should have normalized entropy
          and STD values or not. If it is False, then real entropy and STD
          values are used for the radial axis. If it is True, normalized values
          are used. Defaults to False.

    Raises:
        TypeError: The error is raised if bool_notmalized_measures is not one
          of bool type.
        ValueError: The error is raised if string_mid_type is not one of
          the following 'scaled', 'normalized'.

    Returns:
        plotly.graph_objects.Figure: This chart contains the resulting Mutual
          Information diagram.
    """

    list_valid_mid_types = ['normalized', 'scaled']

    if not isinstance(bool_normalized_measures, bool):
        raise TypeError('bool_normalized_measures must be of bool type')

    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    list_df_input = _list_check_list_df_input(list_df_input)
    list_df_mid = []

    for int_i, df_input in enumerate(list_df_input):
        # We have to check if the secons pandas.DataFrame has one or multiple
        # rows. If it has one, we encode that property using the size of the
        # mark of the resulting diagram. If it has multiple rows, we need to
        # calculate all information for that dataframe and we visualize both
        # using arrows in the resulting diagram
        if int_i == 1 and df_input.shape[0] == 1:
            # We don't calculate properties when we have one row
            # We consider that a scalar value which we encode using mark size
            list_df_mid.append(
                df_input.melt().rename(
                    columns={'variable': 'Model', 'value': 'Scalar'}))
            continue

        df_mid = df_calculate_mid_properties(
            df_input, string_reference_model, dict_mi_parameters)
        list_df_mid.append(df_mid)

    chart_result = _chart_create_diagram(
        list_df_mid, string_reference_model=string_reference_model,
        string_mid_type=string_mid_type, bool_flag_as_subplot=False,
        string_diagram_type='mid',
        bool_normalized_measures=bool_normalized_measures)

    return chart_result


def _df_check_df_raw_input(df_input):
    """
    _df_check_df_raw_input performs a sanity checks on the df_input argument.

    Args:
        df_input (pandas.DataFrame): This DataFrame have models in columns and
          model prediction in rows. The DataFrames is used to calculate
          relevant statistical information and information theory properties.

    Raises:
        TypeError: This error is raised if the argument is not a
          pandas.DataFrame object.

    Returns:
        pandas.DataFrame: This function returns a validated DataFrame.
    """

    if not isinstance(df_input, pd.DataFrame):
        raise TypeError('df_input must be a pandas.DataFrame')

    return df_input


def _list_check_list_df_input(list_df_input):
    """
    _list_check_list_df_input performs a sanity checks on the list_df_input
    argument.

    Args:
        list_df_input (list): This list contains one or two dataframes which
          have models in columns and model prediction in rows. If parsed as a
          pd.DataFrame() object, it is considered as a first and only element
          of the list. Each one of these dataframes is used to calculate
          relevant statistical information and information theory properties.
          If the list contains two elements, both dataframes need to have the
          same set of columns. If the second dataframe contains only one row,
          then this dataframe is considered to contain a property that is
          encoded as using size of the marker of the resulting diagrams. If the
          second dataframe contains multiple rows, it is then considered to be
          a second version of the first dataframe in the list. This is then
          encoded using solid borders around circle marks in the resulting
          diagrams.

    Raises:
        TypeError: This error is raised if the argument is neither a list nor
          a pandas.DataFrame object.
        TypeError: This error is raised if the argument contains elements that
          are not a pandas.DataFrame object.
        ValueError: This error is raised if the argument contains more than 2
          elements.
        AttributeError: This error is raised if the pandas.DataFrame elements
          do not contain the same set of columns. This happens if the column
          names are different between two pandas.DataFrame elements.

    Returns:
        list: This function returns a validated list. If the parsed argument
          was a pandas.DataFrame object, it is now the only element of the
          resulting list.
    """
    list_valid_list_df_input_types = (list, pd.DataFrame)
    list_valid_list_df_input_lenghts = [1, 2]

    if isinstance(list_df_input, list_valid_list_df_input_types):
        if isinstance(list_df_input, pd.DataFrame):
            list_df_input = [list_df_input]
    else:
        raise TypeError('list_df_input is not a list nor a pandas.DataFrame')

    if not all(isinstance(df_input, pd.DataFrame)
               for df_input in list_df_input):
        raise TypeError('list_df_input can contain only pandas.DataFrames' +
                        ' elements')

    if len(list_df_input) not in list_valid_list_df_input_lenghts:
        raise ValueError('list_df_input can contain only one or two' +
                         ' pandas.DataFrames')

    if len(list_df_input) == 2:
        if set(list_df_input[0].columns.to_list()) != set(
                list_df_input[1].columns.to_list()):
            raise AttributeError('list_df_input dataframes need to have the ' +
                                 'same set of columns')
    return list_df_input


def chart_create_all_diagrams(list_df_input, string_reference_model,
                              string_mid_type='scaled',
                              string_corr_method='pearson',
                              dict_mi_parameters=dict(
                                  string_entropy_method='auto',
                                  int_mi_n_neighbors=3,
                                  discrete_models='auto',
                                  bool_discrete_reference_model=False,
                                  int_random_state=_INT_RANDOM_SEED),
                              bool_normalized_measures=False):
    """
    chart_create_all_diagrams creates both the Taylor and the Mutual
    Information diagrams (side-by-side) according to the list_df_input argument
    where models are placed in columns and rows contain model predictions.

    Args:
        list_df_input (list): This list contains one or two dataframes which
          have models in columns and model prediction in rows. If parsed as a
          pd.DataFrame() object, it is considered as a first and only element
          of the list. Each one of these dataframes is used to calculate
          relevant statistical information and information theory properties.
          If the list contains two elements, both dataframes need to have the
          same set of columns. If the second dataframe contains only one row,
          then this dataframe is considered to contain a property that is
          encoded as using size of the marker of the resulting diagrams. If the
          second dataframe contains multiple rows, it is then considered to be
          a second version of the first dataframe in the list. This is then
          encoded using solid borders around circle marks in the resulting
          diagrams.
        string_reference_model (str): This string contains the name of the
          model present in one or both elements of the list_df_input argument
          (as a column) which can be considered as a reference point in the
          final diagram. This is often the ground truth.
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
          dict(int_mi_n_neighbors=3, string_entropy_method='auto',
               bool_discrete_reference_model=False, discrete_fetures='auto',
               int_random_state=_INT_RANDOM_SEED).
        bool_normalized_measures (bool, optional): This boolean parameter is
          used to determine if the passed chart should have normalized entropy
          and STD values or not. If it is False, then real entropy and STD
          values are used for the radial axis. If it is True, normalized values
          are used. Defaults to False.

    Raises:
        TypeError: The error is raised if bool_notmalized_measures is not one
          of bool type.
        ValueError: The error is raised if string_mid_type is not one of
          the following 'scaled', 'normalized'.

    Returns:
        plotly.graph_objects.Figure: This chart contains the both the Taylor
          and the Mutual Information diagrams side-by-side.
    """

    list_valid_mid_types = ['normalized', 'scaled']

    if not isinstance(bool_normalized_measures, bool):
        raise TypeError('bool_normalized_measures must be of bool type')

    if string_mid_type not in list_valid_mid_types:
        raise ValueError('string_mid_type not in ' + str(list_valid_mid_types))

    list_df_input = _list_check_list_df_input(list_df_input)

    string_combined_chart_title = "Taylor Diagram and Mutual Information Diagram" # noqa
    string_angular_title_td = 'Correlation'

    if string_mid_type == 'scaled':
        string_angular_title_mid = 'Scaled Mutual Information'
    else:
        string_angular_title_mid = 'Normalized Mutual Information'

    list_df_all = []

    for int_i, df_input in enumerate(list_df_input):
        # We have to check if the secons pandas.DataFrame has one or multiple
        # rows. If it has one, we encode that property using the concentric
        # circle around mark of the resulting diagram. If it has multiple rows,
        # we need to calculate all information for that dataframe and we
        # visualize both using circles with and without the border
        if int_i == 1 and df_input.shape[0] == 1:
            # We don't calculate properties when we have one row
            # We consider that a scalar value which we encode using mark size
            list_df_all.append(df_input.melt().rename(
                    columns={'variable': 'Model', 'value': 'Scalar'}))
            continue

        df_all = df_calculate_all_properties(
            df_input=df_input, string_reference_model=string_reference_model,
            string_corr_method=string_corr_method,
            dict_mi_parameters=dict_mi_parameters)
        list_df_all.append(df_all)

    chart_result = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'polar'}]*2],
        subplot_titles=(string_angular_title_td,  string_angular_title_mid))

    chart_result = _chart_create_diagram(
        list_df_all, string_reference_model=string_reference_model,
        bool_flag_as_subplot=True, chart_result_upper=chart_result,
        string_diagram_type='taylor', string_mid_type=None,
        bool_normalized_measures=bool_normalized_measures)

    chart_result = _chart_create_diagram(
        list_df_all, string_reference_model=string_reference_model,
        string_mid_type=string_mid_type, bool_flag_as_subplot=True,
        chart_result_upper=chart_result, string_diagram_type='mid',
        bool_normalized_measures=bool_normalized_measures)

    chart_result.update_annotations(
        yshift=25, font_color=_STRING_LABEL_TITLE_COLOR)
    chart_result.update_layout(
        title=dict(
            text=string_combined_chart_title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=20,
                color=_STRING_LABEL_TITLE_COLOR)))

    return chart_result
