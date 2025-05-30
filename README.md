# AOPT_MOONAS
A curated, end‑to‑end collection of all research papers, code examples, and hands‑on experiments developed and studied during the Applied Optimization seminar in spring 2025.

# Programming Project
End-to-end multi-objective NAS on NASBench-201 (and a dummy toy space).
Supports gradient-based (DARTS-MGDA) and evolutionary (NSGA-II) search.

# My NAS Pipeline

Multi-objective Neural Architecture Search (NAS) framework supporting both gradient-based (DARTS with MGDA) and evolutionary (NSGA-II) methods, applied to NAS-Bench-201 and a toy dummy search space.

---

## Repository Structure

```
my_nas_pipeline/
├── README.md
├── requirements.txt
├── config/
│   ├── spaces/             # YAML configs for search spaces
│   └── methods/            # YAML configs for NAS algorithms
├── datasets/               # Data download & DataLoader wrappers (CIFAR-10)
├── search_spaces/
│   ├── abstract_space.py   # Base classes: SearchSpace, Evaluator, Benchmark
│   ├── nb201/              # NASBench-201 implementation
│   └── dummy/              # Toy search space implementation
├── nas_methods/
│   ├── differentiable/     # DARTS + MGDA (MODNAS-derived)
│   └── evolutionary/       # NSGA-II implementation
├── multiobjective/         # Obj. normalization & Pareto utilities
├── utils/                  # Logging, plotting, data helpers
├── scripts/                # CLI entrypoints: search, eval_archs, plot_pareto
└── tests/                  # (To be added) unit tests for core components
```

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/my_nas_pipeline.git
cd my_nas_pipeline

# Create a conda environment (or virtualenv)
conda create -n nas_env python=3.8
conda activate nas_env

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All hyperparameters are specified via YAML:

- **Space configs** (`config/spaces/*.yml`):
  - `nb201.yml`: path to NAS-Bench-201 tfrecord, dataset name, etc.
  - `dummy.yml`: dimensions, primitives list

- **Method configs** (`config/methods/*.yml`):
  - `darts_mgd.yml`: learning rates, epochs, batch size
  - `nsga2.yml`: population size, generations, crossover/mutation rates

Example `config/spaces/nb201.yml`:
```yaml
data_dir: /path/to/nb201
```

Example `config/methods/nsga2.yml`:
```yaml
pop_size: 50
num_gens: 30
crossover_rate: 0.9
mutation_rate: 0.1
```

## Quick Start

### 1. Run NAS Search

```bash
python scripts/search.py \
  --space nb201 \
  --method nsga2 \
  --config_space config/spaces/nb201.yml \
  --config_method config/methods/nsga2.yml \
  --out_dir results
```

This will create a timestamped directory under `results/nb201_nsga2_YYYYMMDD_HHMMSS/` containing:
- `results.json`: final architectures and their objective values

### 2. Evaluate and Compare

Re-evaluate saved architectures and compute IGD & hypervolume:

```bash
python scripts/eval_archs.py \
  --space nb201 \
  --config_space config/spaces/nb201.yml \
  --results results/nb201_nsga2_20250424_101530/results.json
```

### 3. Plot Pareto Fronts

Compare multiple runs against the true front:

```bash
python scripts/plot_pareto.py \
  --space nb201 \
  --config_space config/spaces/nb201.yml \
  results/nb201_nsga2_20250424_101530 \
  results/nb201_darts_mgd_20250424_102010
```

![Pareto Comparison](docs/pareto_example.png)

## Next Steps

- Add unit tests under `tests/` for spaces, evaluators, pareto utils, and NAS methods.  
- Extend to other benchmarks by creating new subfolders in `search_spaces/`.  
- Implement multi-fidelity searches and predictor modules.

---

*Created on 2025-04-24*

