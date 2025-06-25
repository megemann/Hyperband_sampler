# Multi-Objective Hyperband Sampler for Optuna

## Features

### Core Hyperband Algorithm
The **Hyperband Algorithm** is a widely used **multi-fidelity** search strategy that enables early stopping and adaptive resource allocation. It balances exploration (evaluating many configurations with minimal resources) and exploitation (training the most promising configurations to completion). 

Hyperband achieves this by creating tournament-style brackets, where each bracket is has an aggressive, conservative, or balanced approach. At each rung, the top preforming trial(s) get promoted and the algorithm allocates more resources in training those to completion. 

> For a more detailed explanation of Hyperband, please refer to the following references
>- **[My Blog Post on Hyperband Implementation](your-blog-url)** (coming soon)
>- Original research paper: [Hyperband: A Novel Bandit-Based Approach](https://arxiv.org/abs/1603.06560)
>- Internal documentation: `docs/HP_Optimization_Manual.pdf` 

#### Implementation Design
This implementation reimagines Hyperband as an **Optuna Compatible Sampler** rather than a pruner, inspired by the Keras Tuner implementation of Hyperband.

To provide a less intrusive approach than *Keras Tuner* that follows the structure of a *Optuna Sampler*, Hyperband injects a custom parameter called `resource` into each trial via a `suggest_int` hyperparameter call. This allows the resource allocation to be **dynamic**, governed by the algorithm instead of a fixed training schedule. 

**Hyperparameters:**
- `min-resource`: Minimum number of resources allocated to a trial (min_epochs)
- `max-resource`: Maximum number of resources allocated (max_epochs)
- `reduction-factor`: Elimination factor at each rung, (higher = more elimination)
- `hyperband-iterations`: Number of independent Hyperband runs

### Multi-Objective Optimization
This sampler also supports native Multi-Objective Optimization. To enable it, pass the `hyperband_study` only a directions array (e.g. `['minimize','maximize']`) with no input to the parameter `direction`. 

In multi-objective optimizations, the algorithm implements a *Pareto Front* to provide a collection of all optimal trade offs between solutions. It provides the set of all non-dominating solutions, where no solution is better in all objectives.

Please refer to all specifications for multi-objective tuners and the implementation example in the [Optuna Documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html#multi-objective)

### Performance & Scalability
Due to limitations of bracket-based scheudling, each trial within a rung is not independent. This provides an issue with Optunas native parallelization, which assumes either full independence or dependence on **only** prior trials.

To provide some scaling support, we enabled parallelization through independent executions of the hyperband sampler, governed specifically by the **`hyperband-iterations`** hyperparameter. By assigning **`hyperband-iterations > 1`** and `n-jobs = -1 or > 1`, the study will assign parallel executions via **Python Threads**. This works for both single- and multi-objective optimization and supports GPU accelerated workflows. 

> **WARNING:** This is **NOT** Optuna's native parallel processing with multi-worker parallelism. All parallelism is managed by our native `hyperband_study` via internal threading

## Installation


### Prerequisites
- Python 3.7+
- PyTorch (for neural network examples)
- Optuna >= 3.0.0
- NumPy, scikit-learn (for examples)

### Installation Method

#### From Source (Recommended)
```bash
# Clone the repository
git clone https://github.com/megemann/Hyperband_sampler.git
cd hyperband_sampler

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

#### Direct Installation from GitHub
```bash
pip install git+https://github.com/megemann/Hyperband_sampler.git
```

### Verify Installation
```python
from hyperband_sampler import HyperbandStudy
```


## Quick Start

### Single-Objective Optimization

```python
from hyperband_sampler import HyperbandStudy

# Define objective function with resource parameter
def objective(trial):
    # Hyperparameters to optimize
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    
    # Resource parameter (epochs) - controlled by Hyperband
    epochs = trial.suggest_int("resource", 1, 100)
    
    # Simulate training with different epochs
    # More epochs = better performance but more cost
    loss = (x**2 + y**2) * (1.0 / epochs) + learning_rate * 0.1
    
    return loss  # Minimize this value

# Create and run Hyperband study
study = HyperbandStudy(
    min_resource=3,        # Minimum epochs
    max_resource=30,       # Maximum epochs  
    reduction_factor=3,    # Elimination factor
    directions="minimize"  # Single objective
)

# Optimize
result = study.optimize(objective, timeout=300)  # 5 minutes

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### Multi-Objective Optimization

```python
from hyperband_sampler import HyperbandStudy

# Define multi-objective function with resource parameter
def multi_objective(trial):
    # Model hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_size = trial.suggest_int("hidden_size", 32, 512)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Resource parameter (training epochs) - controlled by Hyperband
    epochs = trial.suggest_int("resource", 1, 100)
    
    # Simulate competing objectives
    # Objective 1: Model accuracy (maximize)
    accuracy = 0.8 + (epochs / 100) * 0.15 - dropout * 0.1
    
    # Objective 2: Model complexity/size (minimize) 
    complexity = n_layers * hidden_size * (1 + dropout)
    
    return accuracy, complexity  # (maximize, minimize)

# Create multi-objective Hyperband study
study = HyperbandStudy(
    min_resource=5,
    max_resource=50,
    reduction_factor=3,
    directions=["maximize", "minimize"]  # Multi-objective
)

# Optimize with parallel iterations
result = study.optimize(
    multi_objective, 
    hyperband_iterations=3,  # Run 3 independent Hyperband runs
    n_jobs=2,               # Parallel execution
    timeout=600             # 10 minutes
)

# Get Pareto front solutions
print(f"Pareto front size: {len(study.best_trials)}")
for i, trial in enumerate(study.best_trials[:3]):
    print(f"Solution {i+1}: accuracy={trial.values[0]:.3f}, "
          f"complexity={trial.values[1]:.1f}, params={trial.params}")
```

### Key Features

- **Resource allocation**: Hyperband automatically manages the `resource` parameter (typically epochs/iterations)
- **Early stopping**: Poor configurations get eliminated early with few resources
- **Multi-fidelity**: Good configurations get more resources for better evaluation
- **Parallel execution**: Multiple Hyperband iterations can run in parallel
- **GPU compatible**: Uses threading instead of multiprocessing for GPU safety

## API Reference

### `HyperbandSampler`
*HyperbandSampler(min_resource: int, max_resource: int, reduction_factor: int = 3, seed: int = 42)*

#### parameters
- `min_resource` (int)
  - minimum resource to be allocated in a trial.
  - set this to the minimum viable resource where trials can be eliminated with semi-confidence
- `max_resource` (int)
  - maximum resource to be allocated in a trial.
- `reduction_factor` (int)
  - factor of trials to be promoted at each rung
  - Larger reduction factor > more elimination
- `seed` (int)
  - random sampler seed
  - used for reproduction of trials
  
#### `.visualize_brackets()`:
- plots graph of bracket rungs

#### `.is_complete()`:
- returns completion status of the sampler

#### `.get_total_trials_needed()`:
- returns the total resources needed to run the entire hyperband iteration

#### `get_bracket_breakdown()`:
- prints the specific bracket breakdown for this hyperparameter sampler

#### `.infer_relative_search_space`(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
- Optuna required function (no function in hyperband)
  
#### `.sample_relative`(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
- Optuna required function (no function in hyperband)

#### `.sample_independent`(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, param_name: str, param_distribution: optuna.distributions.BaseDistribution) 
- sample next configuration based on next configuration queued in the bracket


#### `.before_trial`(study: optuna.study.Study, trial: optuna.trial.FrozenTrial)
- simple trial preprocessing

#### `.after_trial`(study: optuna.study.Study, trial: optuna.trial.FrozenTrial)
- possibly promotes configurations and starts the next bracket

For more specific information on how Optuna Tuners function, and the use case for each of the optuna required abstract methods, please refer to the [Base Tuner](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BaseSampler.html) documentation.

### `HyperbandStudy`

*`Hyperband_Study`(min_resource, max_resource, reduction_factor, hyperband_iterations=1, 
                 directions=None, direction=None, study_name=None, storage=None, 
                 load_if_exists=False, sampler_seed=None, ``study_kwargs)*

#### parameters
- sampler args: `min_resource, max_resource, reduction_factor`.
- `hyperband_iterations` (int)
  - complete iterations of the hyperband algorithm to be ran in this study
- `directions` (Array[string]) / `direction` (string)
  - directions: multiobjective optimization
  - direction: singleobjective optimization
  - Direction of each attribute returned by the `optimize` function. The number of directions should be exactly the same as the objectives returned
- for the rest of the arguments, please refer to the [Optuna Study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study) documentation

####  `.optimize`(objective, n_trials=None, timeout=None, catch=(), callbacks=None, gc_after_trial=False, show_progress_bar=False, n_jobs=1):
- Runs the study based on user defined parameters
- `n_jobs`: number of parallel hyperband algorithms to be run in parallel.
- `hyperband_iterations = 1:` `n_jobs` must be equal to 1
- `hyperband_iterations > 1:` `n_jobs` can be up to the number of hyperband iterations or -1 to be the maximum number of parallel threads 
- Similar to the function of a regular `study.optimize` from optuna, in the [Optuna Documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study)

#### Properties
| Property | Description |
|----------|-------------|
| `best_trials` | Returns the best trials (Pareto front for multi-objective optimization, single best trial for single-objective optimization) |
| `best_trial` | Returns the best trial (first Pareto solution for multi-objective, best trial for single-objective) |
| `best_value` | Returns the best value (first Pareto solution values for multi-objective, best value for single-objective) |
| `best_params` | Returns the best parameters (first Pareto solution params for multi-objective, best params for single-objective) |
| `iteration_studies` | Returns list of individual iteration studies (only available when hyperband_iterations > 1) |



## Examples

### UCI Letter Recognition Dataset

This example demonstrates hyperparameter optimization for a neural network on the UCI Letter Recognition dataset.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from hyperband_sampler import HyperbandStudy

# Load and prepare data
X, y = fetch_openml('letter', version=1, return_X_y=True, as_frame=False)
X = StandardScaler().fit_transform(X.astype('float32'))
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to PyTorch datasets
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define neural network model
class LetterNet(nn.Module):
    def __init__(self, trial):
        super().__init__()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = []
        
        for i in range(n_layers):
            size = trial.suggest_int(f"n_units_l{i}", 32, 256)
            hidden_sizes.append(size)
        
        # Build layers
        layers = []
        in_dim = 16  # Input features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_size
        
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 26)  # 26 letters
    
    def forward(self, x):
        x = self.layers(x)
        return self.output(x)

# Define objective function
def objective(trial):
    # Model hyperparameters
    model = LetterNet(trial).to(device)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    
    # Resource parameter controlled by Hyperband
    epochs = trial.suggest_int("resource", 1, 50)
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    return accuracy  # Maximize accuracy

# Create and run Hyperband study
study = HyperbandStudy(
    min_resource=3,
    max_resource=30,
    reduction_factor=3,
    directions="maximize"
)

print(f"Total trials needed: {study.n_trials}")

# Run optimization
result = study.optimize(objective, timeout=3600)  # 1 hour timeout

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

## Performance Considerations

### GPU Compatibility
This implementation uses **threading instead of multiprocessing** for parallel execution, making it fully compatible with GPU workflows. Unlike multiprocessing, threading allows multiple Hyperband iterations to share the same CUDA context without serialization issues.

### Memory Usage
Each Hyperband iteration runs independently with its own model instances. Memory usage scales linearly with `n_jobs` and model size. For large models, consider reducing `hyperband_iterations` or `n_jobs` to avoid memory errors.

### Parallel Processing
- **Threading-based**: Uses `concurrent.futures.ThreadPoolExecutor` for parallel Hyperband iterations
- **GPU-safe**: No CUDA context conflicts unlike multiprocessing approaches  
- **Scaling**: Set `n_jobs=-1` to use all CPU cores, or limit with specific values
- **Limitation**: Only parallelizes across Hyperband iterations, not individual trials within brackets

**Example:**
```python
# Run 4 Hyperband iterations in parallel using 4 threads
study = HyperbandStudy(hyperband_iterations=4, ...)
study.optimize(objective, n_jobs=4)  # or n_jobs=-1 for max cores
```

## Testing

### Test Suite Overview
The package includes comprehensive tests covering core functionality:

- **`test_comprehensive.py`** - Full Hyperband algorithm validation with bracket progression
- **`test_multi_objective_sampler.py`** - Multi-objective optimization and Pareto front generation  
- **`test_multi_objective_study.py`** - HyperbandStudy wrapper with multiple iterations
- **`test_visualization_debug.py`** - Bracket visualization and debugging utilities

### Running Tests
```bash
# Run all tests
python hyperband_sampler/run_tests.py

# Run specific test
python hyperband_sampler/test_comprehensive.py
```

### Test Coverage
- Single and multi-objective optimization
- Serial and parallel execution (`n_jobs > 1`)
- Bracket structure validation
- Resource allocation correctness
- GPU compatibility (threading vs multiprocessing)

## Contributing
If you would like to contribute, contact me at ajfairbanks2005@gmail.com. I will open up the repo for forks and PR's

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this package in academic work, please consider citing:
```
@software{hyperband_sampler,
  title={Multi-Objective Hyperband Sampler for Optuna},
  author={Austin Fairbanks},
  year={2025},
  url={https://github.com/megemann/Hyperband_sampler}
}
```
