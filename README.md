# AOPT\_MOONAS

**AOPT\_MOONAS** is a PyTorch-based framework for multi‐objective neural architecture search (MO‐NAS) on CIFAR‐10. It implements three search strategies:

* **MGDA** (Multi‐Gradient Descent Algorithm)
* **NSGA‐II** (Non‐Dominated Sorting Genetic Algorithm II)
* **Random Search** baseline

Once Pareto‐optimal architectures are found, the code also supports retraining each discovered architecture to obtain its true performance.

---

## Table of Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Configuration Files](#configuration-files)
4. [Usage](#usage)

   * [Training a Supernet (Search Phase)](#training-a-supernet-search-phase)
   * [Evaluating & Retrieving Pareto Fronts](#evaluating--retrieving-pareto-fronts)
   * [Retraining Selected Architectures](#retraining-selected-architectures)
   * [Visualizing Sample Predictions](#visualizing-sample-predictions)
5. [Example Commands](#example-commands)
6. [Dependencies](#dependencies)
7. [Contact & License](#contact--license)

---

## Example Outputs

### Approximated Pareto Fronts

![Approximated Pareto](plots/approximated_pareto.png)

### True Pareto Front (after retraining)

![True Pareto Front](plots/true_pareto.png)

### Qualitative Visualization of Best Architectures

#### MGDA Best Architecture
![MGDA Best Visualization](plots/mgda_best_vis.png)

#### NSGA Best Architecture
![NSGA Best Visualization](plots/nsga_best_vis.png)

#### Random Search Best Architecture
![Random Best Visualization](plots/random_best_vis.png)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Sivanirosh/AOPT_MOONAS.git
   cd AOPT_MOONAS
   ```

2. **Create and activate a Python virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

   > The primary dependencies are `torch`, `torchvision`, `numpy`, `PyYAML`, `tqdm`, and `matplotlib`.

4. **Prepare CIFAR‐10 data** (automatically downloaded by PyTorch when training/evaluating).

---

## Repository Structure

Below is a high‐level overview of the folder layout:

```
AOPT_MOONAS/
├── .gitignore
├── README.md
├── notebooks/                          # Jupyter notebooks for toy examples & experiments
│   ├── Toy_example.ipynb
│   ├── Toy_example_v2.ipynb
│   └── MOO_Project_final_version.ipynb
├── config/                             # YAML configuration files for different search strategies
│   ├── config_model_MGDA.yaml
│   ├── config_model_random.yaml
│   └── config_model_NSGA.yaml
├── data/                               # CIFAR‐10 utilities and raw data
│   ├── cifar-10-batches-py/            # Raw binary files for CIFAR‐10 (auto‐populated)
│   ├── cifar-10-python.tar.gz
│   ├── dataloader.py                   # Custom DataLoader (if needed)
│   ├── dataset.py                      # CIFAR‐10 Dataset wrapper
│   └── transforms.py                   # Data augmentation / normalization
├── eval/                               # Evaluation utilities
│   ├── __init__.py
│   └── evaluator.py                    # Pareto‐front utility functions
├── optimizer/                          # Search‐strategy implementations
│   ├── __init__.py
│   ├── msgda.py                        # MGDA‐based gradient solver
│   ├── nsga2.py                        # NSGA‐II (EA) implementation
│   └── ste.py                          # Straight‐Through Estimator (ReinMaxSTE)
├── models/                             # Model definitions
│   ├── __init__.py
│   ├── factory.py                      # Factory to instantiate SuperNetwork & HyperNetwork from config
│   ├── hypernet.py                     # HyperNetwork definition
│   └── supernet.py                     # SuperNetwork (search‐space) definition
├── parsing/                            # Argument & config parsing utilities
│   ├── __init__.py
│   ├── arg_parser.py                   # CLI argument definitions
│   └── config_parser.py                # YAML config parser
├── requirements.txt                    # pip requirements
├── search/                             # Search‐phase orchestrator
│   ├── __init__.py
│   └── updater.py                      # Unified interface for MGDA, NSGA‐II, Random Search updaters
├── training/                           # Training & retraining utilities
│   ├── __init__.py
│   ├── trainer.py                      # “search‐phase” Trainer that invokes weight & arch updates
│   ├── Checkpointer.py                 # Checkpoint‐saving helper
│   └── retrainer.py                    # Retraining a fixed discrete architecture to convergence
├── utils/                              # Miscellaneous I/O helpers
│   ├── __init__.py
│   └── io.py                           # Logging / file I/O utilities
├── evaluate.py                         # Script to approximate & true Pareto fronts, plus visualization
├── train.py                            # Main entry‐point for search‐phase training
└── readme_template.md                  # (This README file)
```

---

## Configuration Files

All of the search parameters (learning rates, batch sizes, number of epochs, choice of search strategy, etc.) are controlled via YAML files under the `config/` directory. There are three example configuration files provided:

* `config/config_model_MGDA.yaml`
* `config/config_model_NSGA.yaml`
* `config/config_model_random.yaml`

Each config contains two top‐level sections:

1. **`model`**:

   * `supernet`: parameters for building the SuperNetwork (e.g. channel width, number of cells, number of nodes per cell).
   * `hypernet`: parameters for building the HyperNetwork (e.g. hidden dimensions, depth).

2. **`training`**:

   * `epochs`: number of search‐phase epochs.
   * `lr_weights`: learning rate for supernet weights.
   * `lr_arch`: learning rate for hypernet parameters.
   * `batch_size`: training batch size.
   * `val_batch_size`: validation batch size (held‐out batch for architecture update).
   * `search_strategy`: one of `MGDA`, `NSGAII`, or `Random`.
   * Additional strategy‐specific parameters (e.g. `mgda_iters`, `nsga_population_size`, `random_samples`).

To modify these settings, you can copy one of the existing YAML files and edit accordingly.

---

## Usage

### 1. Training a Supernet (Search Phase)

The “search phase” jointly trains:

* **SuperNetwork weights** using **classification‐only** loss (λ = \[1.0, 0.0]).
* **HyperNetwork parameters** using a multi‐objective update (MGDA / NSGA‐II / Random Search).

All of that is driven by `train.py`, which reads your chosen config file, constructs models, and runs the training loop.

#### Command‐Line Arguments

```bash
python train.py \
    --config <path/to/config.yaml> \
    --device-id <GPU_ID> \
    --output-dir <output_directory_for_checkpoints>
```

* `--config`: Path to one of the YAML configuration files (e.g. `config/config_model_MGDA.yaml`).
* `--device-id`: GPU index to use (e.g. `0`).
* `--output-dir`: Directory where all search‐phase checkpoints and logs will be saved.

> **Output Files**
> After training completes, you will find in `output-dir/`:
>
> * `checkpoint.pth` (contains saved `supernet` and `hypernet` weights).
> * A small JSON or YAML file (depending on your config) that logs hyperparameters and training metrics.

### 2. Evaluating & Retrieving Pareto Fronts

Once the search phase finishes, use `evaluate.py` to sample λ‐values, compute the corresponding accuracy and size penalty on the CIFAR‐10 test set, and approximate the Pareto front for each search strategy (MGDA, NSGA‐II, Random).

```bash
python evaluate.py \
    --device-id <GPU_ID> \
    --config-file-path <path/to/main_config.yaml> \
    --mgda-checkpoint <path/to/mgda/checkpoint.pth> \
    --nsga-checkpoint <path/to/nsga/checkpoint.pth> \
    --random-checkpoint <path/to/random/checkpoint.pth> \
    --n-samples <number_of_random_lambdas> \
    --output-dir <output_directory_for_plots_and_models>
```

* `--config-file-path`: Should point to a “master” config that contains anything needed to reconstruct SuperNetwork / HyperNetwork classes (only for retraining).
* `--mgda-checkpoint`, `--nsga-checkpoint`, `--random-checkpoint`: Paths to the three saved checkpoint files from the search stage.
* `--n-samples`: Number of λ‐samples to draw per strategy.
* `--output-dir`: Directory where plots and discrete architectures will be saved.

The script does the following:

1. **Load each checkpoint** (MGDA / NSGA‐II / Random).
2. **Compute “approximate Pareto”** by sampling `n-samples` random λ vectors and measuring accuracy (test‐set) + size penalty.
3. **Plot the approximated Pareto curves** for each strategy (saved under `output-dir/plots/`).
4. **Select one best λ** per strategy (highest accuracy on the sampled Pareto front) and convert it to a **discrete architecture** (argmax over logits).
5. **Write each discrete architecture** (list of chosen operations) to `output-dir/models/{strategy}_best.yaml`.

### 3. Retraining Selected Architectures

Once you have a discrete architecture (e.g. `[0,2,3,1,…]` – representing which op was chosen on each edge), you can retrain it from scratch to obtain its *true* classification accuracy and size penalty.

Use the helper function `retrain_fixed_arch` (in `training/retrainer.py`) which:

* Builds a fresh `DiscreteSuperNetwork` using that architecture.
* Trains it for a fixed number of epochs (standard single‐objective classification on CIFAR‐10).
* Saves the best checkpoint (lowest validation‐set loss or highest validation accuracy).
* Returns a dictionary with the final test accuracy and size penalty.

#### Typical Usage in `evaluate.py`

```python
from training.retrainer import retrain_fixed_arch

out = retrain_fixed_arch(
    discrete_arch=arch,                  # list of operation‐indices
    model_kwargs=model_cfg['supernet'],  # same supernet config used in search
    epochs=args.retrain_epochs,
    device=device,
    output_dir=os.path.join(losses_dir, strategy),
    prefix=strategy
)
```

After retraining, results are saved to:

* `output-dir/models/{strategy}_best.pth` (trained weights).
* Optionally, a CSV of the training/val/test losses in `output-dir/losses/{strategy}/`.

### 4. Visualizing Sample Predictions

If you wish to see how the best discovered architecture performs qualitatively (sample images with predicted vs. true labels), `evaluate.py` also includes a small plotting block:

```python
from data.transforms import get_val_transform
from data.dataset import CIFAR10Dataset
from torch.utils.data import DataLoader

val_dataset = CIFAR10Dataset(train=False, transform=get_val_transform())
vis_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# For each strategy:
net = DiscreteSuperNetwork(**model_cfg['supernet'], discrete_arch=arch)
net.load_state_dict(torch.load(models_dir/{strategy}_best.pth))
```

It then draws 8 sample images, un‐normalizes them, and overlays “True: …” vs. “Pred: …” on the image grid, saving under `output-dir/plots/{strategy}_vis.png`.

---

## Example Commands

Below are several concrete examples—adjust paths/GPUs as needed.

1. **MGDA Search Phase**

   ```bash
   python train.py \
     --config config/config_model_MGDA.yaml \
     --device-id 0 \
     --output-dir outputs/search_mgda
   ```

2. **NSGA‐II Search Phase**

   ```bash
   python train.py \
     --config config/config_model_NSGA.yaml \
     --device-id 0 \
     --output-dir outputs/search_nsga
   ```

3. **Random Search Baseline**

   ```bash
   python train.py \
     --config config/config_model_random.yaml \
     --device-id 0 \
     --output-dir outputs/search_random
   ```

4. **Evaluate & Generate Pareto Fronts**

   ```bash
   python evaluate.py \
     --device-id 0 \
     --config-file-path config/config_model_MGDA.yaml \
     --mgda-checkpoint outputs/search_mgda/checkpoint.pth \
     --nsga-checkpoint outputs/search_nsga/checkpoint.pth \
     --random-checkpoint outputs/search_random/checkpoint.pth \
     --n-samples 30 \
     --output-dir outputs/pareto_results
   ```

5. **Retrain Selected Architectures**
   (This is executed internally by `evaluate.py`, but if you wish to call directly, use:)

   ```python
   from training.retrainer import retrain_fixed_arch

   arch = [2,0,3,1, …]  # e.g. load from YAML: outputs/pareto_results/models/mgda_best.yaml
   out = retrain_fixed_arch(
       discrete_arch=arch,
       model_kwargs=config['model']['supernet'],
       epochs=30,
       device=torch.device('cuda:0'),
       output_dir='outputs/retrain_mgda',
       prefix='mgda'
   )
   print("Final accuracy:", out['final_acc'])
   ```

---

## Dependencies

All dependencies are listed in `requirements.txt`. The core packages are:

* `torch>=1.10.0`
* `torchvision>=0.11.0`
* `numpy`
* `tqdm`
* `matplotlib`
* `PyYAML`

To install them, run:

```bash
pip install -r requirements.txt
```

---

## Contact & License

If you encounter any issues or have suggestions, feel free to open an issue on the GitHub repository:
[https://github.com/Sivanirosh/AOPT\_MOONAS](https://github.com/Sivanirosh/AOPT_MOONAS)

---

**Enjoy experimenting with multi‐objective NAS on CIFAR‐10!**
