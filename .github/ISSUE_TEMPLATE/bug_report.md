---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: AAnzel

---

#### Describe the bug
A clear and concise description of what the bug is.

#### Steps/Code to Reproduce
<!--
Example:
```python
import polar-diagrams as diag

dict_mi_parameters_features_continous_target_continous = dict(
    string_entropy_method='auto',
    int_mi_n_neighbors=3,
    bool_discrete_reference_model=False,
    discrete_models=False,
    int_random_state=42)

chart_taylor_res = diag.chart_create_taylor_diagram(df_anscombes_quartet_modified, string_reference_model=string_ref_model, string_corr_method=string_corr_method)

df_taylor_res = diag.df_calculate_all_properties(df_anscombes_quartet_modified, string_reference_model=string_ref_model, string_corr_method=string_corr_method, dict_mi_parameters=dict_mi_parameters_features_continous_target_continous)

diag.chart_create_all_diagrams(df_anscombes_quartet_modified_with_noise, string_reference_model=string_ref_model, string_corr_method=string_corr_method, string_mid_type=string_mid_type, dict_mi_parameters=dict_mi_parameters_features_continous_target_continous)

```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

```
Sample code to reproduce the problem
```

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Versions
<!--
Please put the version of the library by running:
`pip show polar-diagrams`
-->


<!-- Thanks for contributing! -->
