# Multi-Objective Hyperband Sampler for Optuna

## Features

### Core Hyperband Algorithm
The **Hyperband Algorithm** is a widely used **multi-fidelity** search strategy that enables early stopping and adaptive resource allocation. It balances exploration (evaluating many configurations with minimal resources) and exploitation (training the most promising configurations to completion). 

Hyperband achieves this by creating tournament-style brackets, where each bracket is has an aggressive, conservative, or balanced approach. At each rung, the top preforming trial(s) get promoted and the algorithm allocates more resources in training those to completion. 

For a more detailed explanation of Hyperband, please refer to *docs/HP_Optimization_Manual* or the research paper in references.

#### Implementation Design
This implementation reimagines Hyperband as an **Optuna Compatible Sampler** rather than a pruner, inspired by the Keras Tuner implementation of Hyperband.

To provide a less intrusive approach than *Keras Tuner* that follows the structure of a *Optuna Sampler*, Hyperband injects a custom parameter called ```resource``` into each trial via a ```suggest_int``` hyperparameter call. This allows the resource allocation to be **dynamic**, governed by the algorithm instead of a fixed training schedule. 

**Hyperparameters:**
- ```min-resource```: Minimum number of resources allocated to a trial (min_epochs)
- ```max-resource```: Maximum number of resources allocated (max_epochs)
- ```reduction-factor```: Elimination factor at each rung, (higher = more elimination)
- ```hyperband-iterations```:``` Number of independent Hyperband runs

### Multi-Objective Optimization
This sampler also supports native Multi-Objective Optimization. To enable it, pass the ```hyperband_study``` only a directions array (e.g. ```['minimize','maximize']```) with no input to the parameter ```direction```. 

In multi-objective optimizations, the algorithm implements a *Pareto Front* to provide a collection of all optimal trade offs between solutions. It provides the set of all non-dominating solutions, where no solution is better in all objectives.

Please refer to all specifications for multi-objective tuners and the implementation example in the [Optuna Documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html#multi-objective)

### Performance & Scalability
Due to limitations of bracket-based scheudling, each trial within a rung is not independent. This provides an issue with Optunas native parallelization, which assumes either full independence or dependence on **only** prior trials.

To provide some scaling support, we enabled parallelization through independent executions of the hyperband sampler, governed specifically by the **```hyperband-iterations```** hyperparameter. By assigning **```hyperband-iterations > 1```** and ```n-jobs = -1 or > 1```, the study will assign parallel executions via **Python Threads**. This works for both single- and multi-objective optimization and supports GPU accelerated workflows. 

> **WARNING:** This is **NOT** Optuna's native parallel processing with multi-worker parallelism. All parallelism is managed by our native ```hyperband_study``` via internal threading

## Installation
- Prerequisites
- Installation methods
- Dependency information

## Quick Start

### Single-Objective Optimization
- Basic usage example
- Configuration options
- Results interpretation

### Multi-Objective Optimization
- Setting up multiple objectives
- Pareto front visualization
- Best practices

### Multiple Hyperband Iterations
- Configuring successive halving
- Resource allocation strategies
- Convergence considerations

### Parallel Execution
- Parallelization options
- Distributed computing setup
- Performance scaling

## API Reference

**LATER; USE GITHUB PAGES**
### HyperbandSampler
- Constructor parameters
- Methods and properties
- Usage patterns

### HyperbandStudy
- Study creation and management
- Configuration options
- Integration with Optuna ecosystem

## Examples

### Basic Usage
- Minimal working example
- Key components explained
- Common patterns

### Advanced Multi-Objective
- Complex optimization scenarios
- Handling competing objectives
- Visualization techniques

### UCI Letter Recognition Dataset
- Dataset description
- Hyperparameter optimization approach
- Performance results and analysis

## Performance Considerations

### GPU Compatibility
- GPU acceleration options
- Configuration for GPU environments
- Performance comparisons

### Memory Usage
- Memory optimization techniques
- Handling large search spaces
- Profiling and monitoring

### Parallel Processing
- Thread and process management
- Distributed execution patterns
- Scaling considerations

## Testing
- Test suite overview
- Running tests
- Contributing new tests

## Contributing
- Contribution guidelines
- Development setup
- Code style and standards

## License 
- License information
- Usage restrictions
- Attribution requirements