# synthetic-fine-tuning-experiments

Code for generating the results pertaining to pre-training and fine-tuning found in:

<div align="center">

> **[Synthetic data enable experiments in atomistic machine learning](https://arxiv.org/abs/2211.16443)**\
> _[John Gardner](https://twitter.com/jla_gardner), [Zoé Faure Beaulieu](https://twitter.com/ZFaureBeaulieu) and [Volker Deringer](http://deringer.chem.ox.ac.uk)_

</div>

_(see the [sister repo](https://github.com/jla-gardner/synthetic-data-experiments) for results pertaining to atomistic model comparisons)_

---

## Repo Overview

-   **[src/](src/)** contains the source code for the synthetic fine-tuning method.
-   **[scripts/](scripts/)** contains the scripts to run the experiments.
-   **[notebooks/](./plotting/analysis.ipynb)** contains the notebooks to generate the plots in the paper.
-   **[notebooks/pre-training-demo.ipynb](./notebooks/pre-training-demo.ipynb)** contains a demo of the synthetic fine-tuning method.
-   **[experiment-logs/](./experiment-logs)** contains the results for hyperparameter tuning of all models.

---

## Reproducing our results

### 1. Clone the repo

```bash
git clone https://github.com/jla-gardner/synthetic-fine-tuning-experiments
cd synthetic-fine-tuning
```

### 2. Install dependencies

We strongly recommend using a virtual environment. With `conda` installed, this is as simple as:

```bash
conda create -n fine-tuning python=3.8 -y
conda activate fine-tuning
```

All dependencies can then be installed with:

```bash
pip install -r requirements.txt
```

### 3. Download the data

The **D0** dataset already exists as `./data/bulk_amo.extxyz`. A small sample of the **D1** dataset is included in `./data/synthetic.extxyz`. The full **D1** (~1.5GB) exists at [this url](https://github.com/jla-gardner/carbon-data).


### 4. Run the code

To check that everything is working, run the [demo notebook](notebooks/pre-training-demo.ipynb).

The scripts for running the experiments are in `./scripts/`. To run one of these, do:
    
```bash
./run <script-name>
```

e.g. `./run synthetic_pretrain` or `./run scripts/train_on_dft.py`.
