# Polar Diagrams for Model Comparison

![GitHub Test Status](https://img.shields.io/github/actions/workflow/status/AAnzel/Polar-Diagrams-for-Model-Comparison/python-package.yml?label=test&style=flat-square)
![GitHub Build Status](https://img.shields.io/github/actions/workflow/status/AAnzel/Polar-Diagrams-for-Model-Comparison/python-package.yml?label=build&style=flat-square)
![GitHub Publish](https://img.shields.io/github/actions/workflow/status/AAnzel/Polar-Diagrams-for-Model-Comparison/python-publish.yml?label=publish&style=flat-square)
![GitHub Issues](https://img.shields.io/github/issues-raw/AAnzel/Polar-Diagrams-for-Model-Comparison?style=flat-square)
![PyPI - License](https://img.shields.io/pypi/l/polar-diagrams?style=flat-square)
![PyPI](https://img.shields.io/pypi/v/polar-diagrams?style=flat-square)
![PyPI - Status](https://img.shields.io/pypi/status/polar-diagrams?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/polar-diagrams?style=flat-square)

## Manuscript

This library is created for the following paper:

***"Interactive Polar Diagrams for Model Comparison"*** by Aleksandar Anžel, Dominik Heider, and Georges Hattab

Please cite the paper as:
```latex
@article{ANZEL2023107843,
title = {Interactive polar diagrams for model comparison},
journal = {Computer Methods and Programs in Biomedicine},
volume = {242},
pages = {107843},
year = {2023},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2023.107843},
url = {https://www.sciencedirect.com/science/article/pii/S0169260723005096},
author = {Aleksandar Anžel and Dominik Heider and Georges Hattab},
keywords = {Bioinformatics, Machine-learning, Visualization, Evaluation, Climate, Comparison, Ai, Data-visualization, Information-visualization, Predictive-analysis, Model-comparison, Climate-model-visualization, Ml-model-evaluation, Taylor-diagram, Mutual-information-diagram, Entropy, Mutual-information, Variation-of-information, Correlation, Medical-data},
abstract = {Objective
Evaluating the performance of multiple complex models, such as those found in biology, medicine, climatology, and machine learning, using conventional approaches is often challenging when using various evaluation metrics simultaneously. The traditional approach, which relies on presenting multi-model evaluation scores in the table, presents an obstacle when determining the similarities between the models and the order of performance.
Methods
By combining statistics, information theory, and data visualization, juxtaposed Taylor and Mutual Information Diagrams permit users to track and summarize the performance of one model or a collection of different models. To uncover linear and nonlinear relationships between models, users may visualize one or both charts.
Results
Our library presents the first publicly available implementation of the Mutual Information Diagram and its new interactive capabilities, as well as the first publicly available implementation of an interactive Taylor Diagram. Extensions have been implemented so that both diagrams can display temporality, multimodality, and multivariate data sets, and feature one scalar model property such as uncertainty. Our library, named polar-diagrams, supports both continuous and categorical attributes.
Conclusion
The library can be used to quickly and easily assess the performances of complex models, such as those found in machine learning, climate, or biomedical domains.}
}
```

---
Abstract:

> __Objective__: 
Evaluating the performance of multiple complex models, such as those found in biology, medicine, climatology, and machine learning, using conventional approaches is often challenging when using various evaluation metrics simultaneously. The traditional approach, which relies on presenting multi-model evaluation scores in the table, presents an obstacle when determining the similarities between the models and the order of performance.

> __Methods__:
By combining statistics, information theory, and data visualization, juxtaposed Taylor and Mutual Information Diagrams permit users to track and summarize the performance of one model or a collection of different models. To uncover linear and nonlinear relationships between models, users may visualize one or both charts.

>__Results__:
Our library presents the first publicly available implementation of the Mutual Information Diagram and its new interactive capabilities, as well as the first publicly available implementation of an interactive Taylor Diagram. 
Extensions have been implemented so that both diagrams can display temporality, multimodality, and multivariate data sets, and feature one scalar model property such as uncertainty.
Our library, named \emph{polar-diagrams}, supports both continuous and categorical attributes.

>__Conclusion__:
The library can be used to quickly and easily assess the performances of complex models, such as those found in machine learning, climate, or biomedical domains.



## Dependencies

The code is written in Python 3.9.15 and tested on Linux with the following libraries installed:

|Library|Version|
|---|---|
|numpy|1.23.5|
|pandas|1.5.2|
|scikit-learn|1.2.0|
|scipy|1.9.3|
|plotly|5.9.0|
|kaleido|0.2.1|

The dependencies can also be found in [requirements.txt](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/requirements.txt).

## Data
|Location|Description|
|---|---|
|[Data/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data)|contains all datasets used in [Source/main.ipynb](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/main.ipynb).
|[Data/Dataset_0/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data/Dataset_0/)|contains the Anscombe's quartet data set and the Datasaurus data set.
|[Data/Dataset_1/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data/Dataset_1/)|contains the official, automatically generated script for downloading the CMIP3 data from the [https://esgf-node.llnl.gov/projects/cmip3/](https://esgf-node.llnl.gov/projects/cmip3/). To generate the whole data set, the user should first place itself into this directory and then run the following command from the terminal `sh generate_dataset_1.sh`. [1]
|[Data/Dataset_2/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data/Dataset_2/)|contains the data set from the subsection *3.2 Example 2 — Machine Learning Model Evaluation* of our paper.
|[Data/Dataset_3/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data/Dataset_3/)|contains the data set from the subsection *3.3 Example 3 — Biomedical Similarity Assertion* of our paper.


**[1]** The script used for downloading the [Dataset_1/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Data/Dataset_1/) was generated using the tutorial found here [https://esgf.github.io/esgf-user-support/faq.html#how-to-preserve-the-directory-structure](https://esgf.github.io/esgf-user-support/faq.html#how-to-preserve-the-directory-structure).
Script can be automatically generated and downloaded again from here [https://esgf-data.dkrz.de/esg-search/wget?download_structure=model&project=CMIP3&experiment=historical&ensemble=run1&variable=ts](https://esgf-data.dkrz.de/esg-search/wget?download_structure=model&project=CMIP3&experiment=historical&ensemble=run1&variable=ts).

## Code
|Source Code|Description|
|---|---|
|[Source/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source)|contains all source scripts.
|[Source/main.ipynb](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/main.ipynb)|contains the IPython (jupyter) notebook that demonstrates the library using multiple datasets. This notebook reproduces all of the results we presented in our paper.
|[Source/polar_diagrams/](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/polar_diagrams/)|contains the root source code directory of our library.
|[Source/polar_diagrams/src/polar_diagrams/polar_diagrams.py](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/polar_diagrams/src/polar_diagrams/polar_diagrams.py)|contains the source code that imports the data, modifies it, calculates statistical and information theory properties, and builds diagrams.
|[Source/polar_diagrams/tests/test.py](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/polar_diagrams/tests/test.py)|contains the source code for all unit tests.


## Installation
### Stable
We recommend installing the library using pip:
```bash
pip install polar-diagrams
```

### Unstable

```bash
git clone https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison.git
cd Polar-Diagrams-for-Model-Comparison/Source/polar_diagrams/
pip install --editable .
```

## Running
Please check the API documentation of our library at [Source/polar_diagrams/docs/polar_diagrams.md](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/polar_diagrams/docs/polar_diagrams.md) or the IPython (jupyter) notebook that demonstrates the library at [Source/main.ipynb](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/Source/main.ipynb).

## License

Licensed under the GNU General Public License, Version 3.0 ([LICENSE](https://github.com/AAnzel/Polar-Diagrams-for-Model-Comparison/tree/master/LICENSE) or https://www.gnu.org/licenses/gpl-3.0.en.html)

### Contribution

Any contribution intentionally submitted for inclusion in the work by you, shall be licensed under the GNU GPLv3.
