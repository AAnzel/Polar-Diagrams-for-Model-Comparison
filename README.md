# Model Comparison Polar Diagrams

## Manuscript

This library is created for the following paper:

***"Name of the paper"*** by Aleksandar Anžel, Dominik Heider, and Georges Hattab

Please cite the paper as:
```latex
Bibtex citation placeholder
```

---
Abstract:

> Abstract placeholder


## Dependencies

The code is written in Python 3.9.15 and tested on Linux with the following libraries installed:

|Library|Version|
|---|---|
|numpy|1.23.4|
|pandas|1.5.1|
|scikit-learn|1.1.3|
|scipy|1.9.3|
|plotly|5.9.0|

The dependencies can also be found in [requirements.txt](requirements.txt).

## Data
|Location|Description|
|---|---|
|[Data/](./Data/)|contains all datasets used in [Source/main.ipynb](./Source/main.ipynb).
|[Data/Dataset_0/](./Data/Dataset_0/)|contains the Anscombe's quartet dataset.
|[Data/Dataset_1/](./Data/Dataset_1/)|contains the official, automatically generated script for downloading the CMIP3 data from the [https://esgf-node.llnl.gov/projects/cmip3/](https://esgf-node.llnl.gov/projects/cmip3/). To generate the whole dataset, the user should first place itself into this directory and then run the following command from the terminal `sh generate_dataset_1.sh`. [1]
|[Data/Dataset_2/](./Data/Dataset_2/)|contains the dataset from the following paper [Filippone, Maurizio & Masulli, Francesco & Rovetta, Stefano. (2008). An Experimental Comparison of Kernel Clustering Methods.. Frontiers in Artificial Intelligence and Applications. 193. 118-126. 10.3233/978-1-58603-984-4-118.](https://www.researchgate.net/publication/220956693_An_Experimental_Comparison_of_Kernel_Clustering_Methods).


**[1]** The script used for downloading the [Dataset_1/](./Data/Dataset_1/) was generated using the tutorial found here [https://esgf.github.io/esgf-user-support/faq.html#how-to-preserve-the-directory-structure](https://esgf.github.io/esgf-user-support/faq.html#how-to-preserve-the-directory-structure).
Script can be automatically generated and downloaded again from here [https://esgf-data.dkrz.de/esg-search/wget?download_structure=model&project=CMIP3&experiment=historical&ensemble=run1&variable=ts](https://esgf-data.dkrz.de/esg-search/wget?download_structure=model&project=CMIP3&experiment=historical&ensemble=run1&variable=ts).

## Code
|Source Code|Description|
|---|---|
|[Source/](./Source/)|contains all scripts.
|[Source/main.ipynb](./Source/main.ipynb)|contains the IPython (jupyter) notebook that demonstrates the library using multiple datasets.
|[Source/diagrams.py](./Source/diagrams.py)|contains the source code that imports the data, modifies it, calculates statistical and information theory properties, and builds diagrams.

## Getting started


## Installation & Running
### Stable

### Unstable

## License

Licensed under the GNU General Public License, Version 3.0 ([LICENSE](./LICENSE) or https://www.gnu.org/licenses/gpl-3.0.en.html)

### Contribution

Any contribution intentionally submitted for inclusion in the work by you, shall be licensed under the GNU GPLv3.
