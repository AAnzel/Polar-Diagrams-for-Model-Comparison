# Table of Contents

* [\_\_init\_\_](#__init__)
* [polar\_diagrams](#polar_diagrams)
  * [df\_calculate\_td\_properties](#polar_diagrams.df_calculate_td_properties)
  * [df\_calculate\_mid\_properties](#polar_diagrams.df_calculate_mid_properties)
  * [df\_calculate\_all\_properties](#polar_diagrams.df_calculate_all_properties)
  * [chart\_create\_taylor\_diagram](#polar_diagrams.chart_create_taylor_diagram)
  * [chart\_create\_mi\_diagram](#polar_diagrams.chart_create_mi_diagram)
  * [chart\_create\_all\_diagrams](#polar_diagrams.chart_create_all_diagrams)

<a id="__init__"></a>

# \_\_init\_\_

.. include:: ../README.md

<a id="polar_diagrams"></a>

# polar\_diagrams

<a id="polar_diagrams.df_calculate_td_properties"></a>

## df\_calculate\_td\_properties

```python
def df_calculate_td_properties(df_input,
                               string_reference_model,
                               string_corr_method='pearson')
```

df_calculate_td_properties caclulates all necessary statistical information
for the Taylor diagram from the input data set.

**Arguments**:

- `df_input` _pandas.DataFrame_ - This dataframe has models in columns and
  model prediction in rows. It is used to calculate relevant
  statistical information.
- `string_reference_model` _str_ - This string contains the name of the
  model present in the df_input argument (as a column) which can be
  considered as a reference point in the final diagram. This is often
  the ground truth.
- `string_corr_method` _str, optional_ - This string contains the name of
  the method to be used when calculating the correlation. Defaults to
  'pearson'.
  

**Raises**:

- `ValueError` - The error is raised if the string_corr_method is not one of
  the following 'pearson', 'kendall', 'spearman'.
  

**Returns**:

- `pandas.DataFrame` - This dataframe contains model names as indices and
  statistical properties as columns.

<a id="polar_diagrams.df_calculate_mid_properties"></a>

## df\_calculate\_mid\_properties

```python
def df_calculate_mid_properties(df_input,
                                string_reference_model,
                                dict_mi_parameters=dict(
                                    string_entropy_method='auto',
                                    int_mi_n_neighbors=3,
                                    discrete_models='auto',
                                    bool_discrete_reference_model=False,
                                    int_random_state=_INT_RANDOM_SEED))
```

df_calculate_mid_properties caclulates all necessary information theory
properties for the Mutual Information diagram from the input data set.

**Arguments**:

- `df_input` _pandas.DataFrame_ - This dataframe has models in columns and
  model prediction in rows. It is used to calculate relevan
  information theory propertiesdict_mi_parameters['discrete_models']
- `string_reference_model` _str_ - This string contains the name of the
  model present in the df_input argument (as a column) which can be
  considered as a reference point in the final diagram. This is often
  the ground truth.
- `dict_mi_parameters` _dict, optional_ - This dictionary contains
  configuration parameters for the calculation of entropy and mutual
  information. Defaults to
  dict(int_mi_n_neighbors=3, string_entropy_method='auto',
  bool_discrete_reference_model=False, discrete_models='auto',
  int_random_state=_INT_RANDOM_SEED).
  

**Raises**:

- `ValueError` - The error is raised if int_mi_n_neighbors is less or equal
  than zero.
- `ValueError` - The error is raised if string_entropy_method is not one of
  the following 'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'
  

**Returns**:

- `pandas.DataFrame` - This dataframe contains model names as indices and
  information theory properties as columns.

<a id="polar_diagrams.df_calculate_all_properties"></a>

## df\_calculate\_all\_properties

```python
def df_calculate_all_properties(df_input,
                                string_reference_model,
                                dict_mi_parameters=dict(
                                    string_entropy_method='auto',
                                    int_mi_n_neighbors=3,
                                    discrete_models='auto',
                                    bool_discrete_reference_model=False,
                                    int_random_state=_INT_RANDOM_SEED),
                                string_corr_method='pearson')
```

df_calculate_all_properties caclulates all necessary statistical and
information theory properties for the Taylor and Mutual Information diagram
from the input data set.

**Arguments**:

- `df_input` _pandas.DataFrame_ - This dataframe has models in columns and
  model prediction in rows. It is used to calculate relevant
  statistical information and information theory properties.
- `string_reference_model` _str_ - This string contains the name of the
  model present in the df_input argument (as a column) which can be
  considered as a reference point in the final diagram. This is often
  the ground truth.
- `dict_mi_parameters` _dict, optional_ - This dictionary contains
  configuration parameters for the calculation of entropy and mutual
  information. Defaults to
  dict(int_mi_n_neighbors=3, string_entropy_method='auto',
  bool_discrete_reference_model=False, discrete_fetures='auto',
  int_random_state=_INT_RANDOM_SEED).
- `string_corr_method` _str, optional_ - This string contains the name of
  the method to be used when calculating the correlation. Defaults to
  'pearson'.
  

**Returns**:

- `pandas.DataFrame` - This dataframe contains model names as indices and
  statistical and information theory properties as columns.

<a id="polar_diagrams.chart_create_taylor_diagram"></a>

## chart\_create\_taylor\_diagram

```python
def chart_create_taylor_diagram(list_df_input,
                                string_reference_model,
                                string_corr_method='pearson',
                                bool_normalized_measures=False)
```

chart_create_taylor_diagram creates the Taylor diagram according to the
list_df_input argument where models are placed in columns and rows contain
model predictions.

**Arguments**:

- `list_df_input` _list_ - This list contains one or two dataframes which
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
- `string_reference_model` _str_ - This string contains the name of the
  model present in the df_input argument (as a column) which can be
  considered as a reference point in the final diagram. This is often
  the ground truth.
- `string_corr_method` _str, optional_ - This string contains the name of
  the method to be used when calculating the correlation. Defaults to
  'pearson'.
- `bool_normalized_measures` _bool, optional_ - This boolean parameter is
  used to determine if the passed chart should have normalized entropy
  and STD values or not. If it is False, then real entropy and STD
  values are used for the radial axis. If it is True, normalized values
  are used. Defaults to False.
  

**Raises**:

- `ValueError` - The error is raised if the string_corr_method is not one of
  the following 'pearson', 'kendall', 'spearman'.
- `TypeError` - The error is raised if bool_notmalized_measures is not one
  of bool type.
  

**Returns**:

- `plotly.graph_objects.Figure` - This chart contains the resulting Taylor
  diagram.

<a id="polar_diagrams.chart_create_mi_diagram"></a>

## chart\_create\_mi\_diagram

```python
def chart_create_mi_diagram(list_df_input,
                            string_reference_model,
                            string_mid_type='scaled',
                            dict_mi_parameters=dict(
                                string_entropy_method='auto',
                                int_mi_n_neighbors=3,
                                discrete_models='auto',
                                bool_discrete_reference_model=False,
                                int_random_state=_INT_RANDOM_SEED),
                            bool_normalized_measures=False)
```

chart_create_mi_diagram creates the Mutual Information diagram according
to the list_df_input argument where models are placed in columns and rows
contain model predictions.

**Arguments**:

- `list_df_input` _list_ - This list contains one or two dataframes which
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
- `string_reference_model` _str_ - This string contains the name of the
  model present in the df_input argument (as a column) which can be
  considered as a reference point in the final diagram. This is often
  the ground truth.
- `string_mid_type` _str, optional_ - This string contains the type of the
  Mutual Information diagram. If it is 'scaled' then it will span both
  quadrants. If it is 'normalized', it will span only the first
  quadrant of the circle. Defaults to 'scaled'.
- `dict_mi_parameters` _dict, optional_ - This dictionary contains
  configuration parameters for the calculation of entropy and mutual
  information. Defaults to
  dict(int_mi_n_neighbors=3, string_entropy_method='auto',
  bool_discrete_reference_model=False, discrete_fetures='auto',
  int_random_state=_INT_RANDOM_SEED).
- `bool_normalized_measures` _bool, optional_ - This boolean parameter is
  used to determine if the passed chart should have normalized entropy
  and STD values or not. If it is False, then real entropy and STD
  values are used for the radial axis. If it is True, normalized values
  are used. Defaults to False.
  

**Raises**:

- `TypeError` - The error is raised if bool_notmalized_measures is not one
  of bool type.
- `ValueError` - The error is raised if string_mid_type is not one of
  the following 'scaled', 'normalized'.
  

**Returns**:

- `plotly.graph_objects.Figure` - This chart contains the resulting Mutual
  Information diagram.

<a id="polar_diagrams.chart_create_all_diagrams"></a>

## chart\_create\_all\_diagrams

```python
def chart_create_all_diagrams(list_df_input,
                              string_reference_model,
                              string_mid_type='scaled',
                              string_corr_method='pearson',
                              dict_mi_parameters=dict(
                                  string_entropy_method='auto',
                                  int_mi_n_neighbors=3,
                                  discrete_models='auto',
                                  bool_discrete_reference_model=False,
                                  int_random_state=_INT_RANDOM_SEED),
                              bool_normalized_measures=False)
```

chart_create_all_diagrams creates both the Taylor and the Mutual
Information diagrams (side-by-side) according to the list_df_input argument
where models are placed in columns and rows contain model predictions.

**Arguments**:

- `list_df_input` _list_ - This list contains one or two dataframes which
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
- `string_reference_model` _str_ - This string contains the name of the
  model present in one or both elements of the list_df_input argument
  (as a column) which can be considered as a reference point in the
  final diagram. This is often the ground truth.
- `string_corr_method` _str, optional_ - This string contains the name of
  the method to be used when calculating the correlation. Defaults to
  'pearson'.
- `string_mid_type` _str, optional_ - This string contains the type of the
  Mutual Information diagram. If it is 'scaled' then it will span both
  quadrants. If it is 'normalized', it will span only the first
  quadrant of the circle. Defaults to 'scaled'.
- `dict_mi_parameters` _dict, optional_ - This dictionary contains
  configuration parameters for the calculation of entropy and mutual
  information. Defaults to
  dict(int_mi_n_neighbors=3, string_entropy_method='auto',
  bool_discrete_reference_model=False, discrete_fetures='auto',
  int_random_state=_INT_RANDOM_SEED).
- `bool_normalized_measures` _bool, optional_ - This boolean parameter is
  used to determine if the passed chart should have normalized entropy
  and STD values or not. If it is False, then real entropy and STD
  values are used for the radial axis. If it is True, normalized values
  are used. Defaults to False.
  

**Raises**:

- `TypeError` - The error is raised if bool_notmalized_measures is not one
  of bool type.
- `ValueError` - The error is raised if string_mid_type is not one of
  the following 'scaled', 'normalized'.
  

**Returns**:

- `plotly.graph_objects.Figure` - This chart contains the both the Taylor
  and the Mutual Information diagrams side-by-side.

