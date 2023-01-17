# SimplEx - Explaining Latent Representations with a Corpus of Examples

[![Tests](https://github.com/vanderschaarlab/Simplex/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/Simplex/actions/workflows/test.yml)
[![pdf](https://img.shields.io/badge/PDF-%20NeurIPS%202021-red)](https://papers.nips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://github.com/JonathanCrabbe/Simplex/blob/main/LICENSE)


![image](https://github.com/vanderschaarlab/Simplex/raw/main/simplex_examples.png "Examples of SimplEx explanations")

Code Author: Jonathan Crabbé ([jc2133@cam.ac.uk](mailto:jc2133@cam.ac.uk))

This repository contains the implementation of SimplEx, a method to explain the latent
representations of black-box models with the help of a corpus of examples.
For more details, please read our [NeurIPS 2021 paper](https://papers.nips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html): 'Explaining Latent Representations with a Corpus of Examples'.

## :rocket: Installation

The library can be installed from PyPI using
```bash
$ pip install simplexai
```
or from source, using
```bash
$ pip install .
```

## Toy example

Bellow, you can find a toy demonstration where we make a corpus decomposition of test examples representations.
All the relevant code can be found in the file [simplex](explainers/simplex.py).

```python
from simplexai.explainers.simplex import Simplex
from simplexai.models.image_recognition import MnistClassifier
from simplexai.experiments.mnist import load_mnist

# Get a model
model = MnistClassifier() # Model should have the BlackBox interface

# Load corpus and test inputs
corpus_loader = load_mnist(subset_size=100, train=True, batch_size=100) # MNIST train loader
test_loader = load_mnist(subset_size=10, train=True, batch_size=10) # MNIST test loader
corpus_inputs, _ = next(iter(corpus_loader)) # A tensor of corpus inputs
test_inputs, _ = next(iter(test_loader)) # A set of inputs to explain

# Compute the corpus and test latent representations
corpus_latents = model.latent_representation(corpus_inputs).detach()
test_latents = model.latent_representation(test_inputs).detach()

# Initialize SimplEX, fit it on test examples
simplex = Simplex(corpus_examples=corpus_inputs,
                  corpus_latent_reps=corpus_latents)
simplex.fit(test_examples=test_inputs,
            test_latent_reps=test_latents,
            reg_factor=0)

# Get the weights of each corpus decomposition
weights = simplex.weights
```
We get a tensor weights that can be interpreted as follows:
``weights[i,c] = weight of corpus example c in the decomposition of example i``.

We can get the importance of each corpus feature for the decomposition
of a given example ``i`` in the following way:
```python
import torch

# Compute the Integrated Jacobian for a particular example
i = 4
input_baseline = torch.zeros(corpus_inputs.shape) # Baseline tensor of the same shape as corpus_inputs
simplex.jacobian_projection(test_id=i, model=model, input_baseline=input_baseline)
result = simplex.decompose(i)
```
We get a list ``result`` where each element of the list corresponds to a corpus example.
This list is sorted by decreasing order of importance in the corpus decomposition.
Each element of the list is a tuple structured as follows:
```python
w_c, x_c, proj_jacobian_c = result[c]
```
Where ``w_c`` corresponds to the weight ``weights[i,c]``, ``x_c`` corresponds to ``corpus_inputs[c]``
and ``proj_jacobian`` is a tensor such that ``proj_jacobian_c[k]`` is the Projected Jacobian
of feature ``k`` from corpus example ``c``.


## Reproducing the paper results
### Reproducing MNIST Approximation Quality Experiment
1. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 9)
```shell
python -m simplexai.experiments.mnist -experiment "approximation_quality" -cv CV

```

2. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.mnist.quality.plot_results -cv_list CV1 CV2 CV3 ...

```
3. The resulting plots and data are saved [here](experiments/results/mnist/quality).

### Reproducing Prostate Cancer Approximation Quality Experiment
This experiment requires the access to the private datasets CUTRACT
and SEER decribed in the paper.
1. Copy the files ``cutract_internal_all.csv`` and ``seer_external_imputed_new.csv`` are in the folder ``data/Prostate Cancer``
2. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 9)
```shell
python -m simplexai.experiments.prostate_cancer -experiment "approximation_quality" -cv CV

```
3. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.prostate.quality.plot_results -cv_list CV1 CV2 CV3 ...

```
4. The resulting plots are saved [here](experiments/results/prostate/quality).

### Reproducing Prostate Cancer Outlier Experiment
This experiment requires the access to the private datasets CUTRACT
and SEER decribed in the paper.
1. Make sure that the files ``cutract_internal_all.csv`` and ``seer_external_imputed_new.csv`` are in the folder ``data/Prostate Cancer``
2. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 9)
```shell
python -m simplexai.experiments.prostate_cancer -experiment "outlier_detection" -cv CV

```
3. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.prostate.outlier.plot_results -cv_list CV1 CV2 CV3 ...

```
4. The resulting plots are saved [here](experiments/results/prostate/outlier).

### Reproducing MNIST Jacobian Projection Significance Experiment
1. Run the following script
```shell
python -m simplexai.experiments.mnist -experiment "jacobian_corruption"

```

2.The resulting plots and data are saved [here](experiments/results/mnist/jacobian_corruption).

### Reproducing MNIST Outlier Detection Experiment
1. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 9)
```shell
python -m simplexai.experiments.mnist -experiment "outlier_detection" -cv CV

```

2. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.mnist.outlier.plot_results -cv_list CV1 CV2 CV3 ...

```
3. The resulting plots and data are saved [here](experiments/results/mnist/outlier).

### Reproducing MNIST Influence Function Experiment
1. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 4)
```shell
python -m simplexai.experiments.mnist -experiment "influence" -cv CV

```

2. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.mnist.influence.plot_results -cv_list CV1 CV2 CV3 ...

```
3. The resulting plots and data are saved [here](experiments/results/mnist/influence).

Note: some problems can appear with the package
[Pytorch Influence Functions](https://github.com/nimarb/pytorch_influence_functions).
If this is the case, please change ``calc_influence_function.py`` in the following way:

```python
343: influences.append(tmp_influence) ==> influences.append(tmp_influence.cpu())
438: influences_meta['test_sample_index_list'] = sample_list ==> #influences_meta['test_sample_index_list'] = sample_list
```

### Reproducing AR Approximation Quality Experiment
1. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 4)
```shell
python -m simplexai.experiments.time_series -experiment "approximation_quality" -cv CV

```

2. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.ar.quality.plot_results -cv_list CV1 CV2 CV3 ...

```
3. The resulting plots and data are saved [here](experiments/results/ar/quality).

### Reproducing AR Outlier Detection Experiment
1. Run the following script for different values of CV (the results from the paper
   were obtained by taking all integer CV between 0 and 4)
```shell
python -m simplexai.experiments.time_series -experiment "outlier_detection" -cv CV

```

2. Run the following script by adding all the values of CV from the previous step
```shell
python -m simplexai.experiments.results.ar.outlier.plot_results -cv_list CV1 CV2 CV3 ...

```
3. The resulting plots and data are saved [here](experiments/results/ar/outlier).


## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

## Citing

If you use this code, please cite the associated paper:

```
@inproceedings{Crabbe2021Simplex,
 author = {Crabbe, Jonathan and Qian, Zhaozhi and Imrie, Fergus and van der Schaar, Mihaela},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {12154--12166},
 publisher = {Curran Associates, Inc.},
 title = {Explaining Latent Representations with a Corpus of Examples},
 url = {https://proceedings.neurips.cc/paper/2021/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
