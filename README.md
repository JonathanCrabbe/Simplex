# SimplEx
## Preliminaries
Clone the repository and install the required packages by running
```shell
pip install requirements.txt
```
## Reproducing MNIST Quality Experiment
1. Run 
```shell
python -m experiments.mnist -experiment "approximation_quality" -cv 0
python -m experiments.mnist -experiment "approximation_quality" -cv 1
python -m experiments.mnist -experiment "approximation_quality" -cv 2
python -m experiments.mnist -experiment "approximation_quality" -cv 3
python -m experiments.mnist -experiment "approximation_quality" -cv 4
python -m experiments.mnist -experiment "approximation_quality" -cv 5
python -m experiments.mnist -experiment "approximation_quality" -cv 6
python -m experiments.mnist -experiment "approximation_quality" -cv 7
python -m experiments.mnist -experiment "approximation_quality" -cv 8
python -m experiments.mnist -experiment "approximation_quality" -cv 9
```
2. Run 
```shell
python -m experiments.results.mnist.quality.plot_results

```
3. The resulting plots are saved in ``./experiments/results/mnist/quality``

## Reproducing Prostate Cancer Quality Experiment
1. Make sure that the files ``cutract_internal_all.csv`` and ``seer_external_imputed_new.csv`` are in the folder ``data/Prostate Cancer``
2. Run 
```shell
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 0
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 1
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 2
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 3
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 4
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 5
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 6
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 7
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 8
python -m experiments.prostate_cancer -experiment "approximation_quality" -cv 9

```
3. Run 
```shell
python -m experiments.results.prostate.quality.plot_results

```
4. The resulting plots are saved in ``./experiments/results/prostate/quality``

## Reproducing Prostate Cancer Outlier Experiment
1. Make sure that the files ``cutract_internal_all.csv`` and ``seer_external_imputed_new.csv`` are in the folder ``data/Prostate Cancer``
2. Run 
```shell
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 0
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 1
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 2
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 3
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 4
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 5
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 6
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 7
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 8
python -m experiments.prostate_cancer -experiment "outlier_detection" -cv 9


```
3. Run 
```shell
python -m experiments.results.prostate.outlier.plot_results

```
4. The resulting plots are saved in ``./experiments/results/prostate/outlier``
