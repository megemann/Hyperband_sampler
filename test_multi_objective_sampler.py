#!/usr/bin/env python
"""
Test multi-objective HyperbandSampler.
"""
import optuna
from hyperband_sampler import HyperbandSampler

def multi_objective_function(trial):
    """Multi-objective test function."""
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    epochs = trial.suggest_int("resource", 1, 81)  # Hyperband will override this
    
    # Simulate training with different epochs
    # Objective 1: Distance from origin (minimize)
    obj1 = (x**2 + y**2) * (1.0 / epochs)  # Better with more epochs
    
    # Objective 2: Distance from point (3, 3) (minimize)
    obj2 = ((x - 3)**2 + (y - 3)**2) * (1.0 / epochs)  # Better with more epochs
    
    return obj1, obj2

def single_objective_function(trial):
    """Single objective test function."""
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    epochs = trial.suggest_int("resource", 1, 81)  # Hyperband will override this
    
    # Simple quadratic function
    return (x**2 + y**2) * (1.0 / epochs)

def test_multi_objective_sampler():
    """Test multi-objective HyperbandSampler."""
    print("=" * 80)
    print("MULTI-OBJECTIVE HYPERBAND SAMPLER TEST")
    print("=" * 80)
    
    # Create multi-objective study
    sampler = HyperbandSampler(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        seed=42
    )
    
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Multi-objective
        sampler=sampler
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Sampler is multi-objective: {sampler._is_multi_objective}")
    
    # Run optimization
    n_trials = sampler.get_total_trials_needed()
    print(f"Running {n_trials} trials...")
    
    try:
        study.optimize(multi_objective_function, n_trials=n_trials)
    except optuna.TrialPruned:
        print("Hyperband completed!")
    
    # Get results
    print(f"\nCompleted {len(study.trials)} trials")
    print(f"Best trials (Pareto front): {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials[:3]):  # Show first 3
        print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
    
    return study

def test_single_objective_sampler():
    """Test single-objective HyperbandSampler for comparison."""
    print("\n" + "=" * 80)
    print("SINGLE-OBJECTIVE HYPERBAND SAMPLER TEST")
    print("=" * 80)
    
    # Create single-objective study
    sampler = HyperbandSampler(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        seed=42
    )
    
    study = optuna.create_study(
        direction="minimize",  # Single objective
        sampler=sampler
    )
    
    print(f"Study direction: {study.direction}")
    print(f"Sampler is multi-objective: {sampler._is_multi_objective}")
    
    # Run optimization
    n_trials = sampler.get_total_trials_needed()
    print(f"Running {n_trials} trials...")
    
    try:
        study.optimize(single_objective_function, n_trials=n_trials)
    except optuna.TrialPruned:
        print("Hyperband completed!")
    
    # Get results
    print(f"\nCompleted {len(study.trials)} trials")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    return study

def run_sampler_tests():
    """Run all sampler tests."""
    try:
        print("Starting Multi-Objective HyperbandSampler Tests...")
        
        # Test 1: Multi-objective
        study1 = test_multi_objective_sampler()
        
        # Test 2: Single objective
        study2 = test_single_objective_sampler()
        
        print("\n" + "=" * 80)
        print("ALL SAMPLER TESTS PASSED!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_sampler_tests()
    exit(0 if success else 1) 