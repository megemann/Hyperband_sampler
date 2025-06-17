#!/usr/bin/env python
"""
Test multi-objective HyperbandStudy.
"""
from hyperband_study import HyperbandStudy

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

def test_multi_objective_study_single_iteration():
    """Test multi-objective HyperbandStudy with single iteration."""
    print("=" * 80)
    print("MULTI-OBJECTIVE HYPERBAND STUDY TEST (Single Iteration)")
    print("=" * 80)
    
    # Create multi-objective study
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=1,
        directions=["minimize", "minimize"],
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    
    # Run optimization
    result_study = study.optimize(multi_objective_function)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials")
    print(f"Best trials (Pareto front): {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials[:3]):  # Show first 3
        print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
    
    return study

def test_single_objective_study_single_iteration():
    """Test single-objective HyperbandStudy with single iteration."""
    print("\n" + "=" * 80)
    print("SINGLE-OBJECTIVE HYPERBAND STUDY TEST (Single Iteration)")
    print("=" * 80)
    
    # Create single-objective study
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=1,
        directions=["minimize"],  # Single objective using directions
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    
    # Run optimization
    result_study = study.optimize(single_objective_function)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    return study

def test_single_objective_study_backward_compatible():
    """Test single-objective HyperbandStudy with backward-compatible direction parameter."""
    print("\n" + "=" * 80)
    print("SINGLE-OBJECTIVE HYPERBAND STUDY TEST (Backward Compatible)")
    print("=" * 80)
    
    # Create single-objective study using old direction parameter
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=1,
        direction="minimize",  # Old-style direction parameter
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    
    # Run optimization
    result_study = study.optimize(single_objective_function)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    return study

def test_multi_objective_study_multiple_iterations():
    """Test multi-objective HyperbandStudy with multiple iterations."""
    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE HYPERBAND STUDY TEST (Multiple Iterations)")
    print("=" * 80)
    
    # Create multi-objective study with multiple iterations
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=3,
        directions=["minimize", "minimize"],
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    print(f"Hyperband iterations: {study.hyperband_iterations}")
    
    # Run optimization
    result_study = study.optimize(multi_objective_function)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials across {study.hyperband_iterations} iterations")
    print(f"Final Pareto front size: {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials[:5]):  # Show first 5
        print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
    
    return study

def test_multi_objective_study_parallel():
    """Test multi-objective HyperbandStudy with parallel execution."""
    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE HYPERBAND STUDY TEST (Parallel)")
    print("=" * 80)
    
    # Create multi-objective study with parallel execution
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=3,
        directions=["minimize", "minimize"],
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    print(f"Hyperband iterations: {study.hyperband_iterations}")
    
    # Run optimization with parallel execution
    result_study = study.optimize(multi_objective_function, n_jobs=2)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials across {study.hyperband_iterations} iterations")
    print(f"Final Pareto front size: {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials[:5]):  # Show first 5
        print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
    
    return study

def test_mixed_objectives():
    """Test mixed objectives (minimize and maximize)."""
    print("\n" + "=" * 80)
    print("MIXED OBJECTIVES TEST (Minimize + Maximize)")
    print("=" * 80)
    
    def mixed_objective_function(trial):
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        epochs = trial.suggest_int("resource", 1, 81)
        
        # Objective 1: Distance from origin (minimize)
        obj1 = (x**2 + y**2) * (1.0 / epochs)
        
        # Objective 2: Negative distance from point (3, 3) (maximize = minimize negative)
        obj2 = -((x - 3)**2 + (y - 3)**2) * (1.0 / epochs)
        
        return obj1, obj2
    
    # Create mixed-objective study
    study = HyperbandStudy(
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        hyperband_iterations=1,
        directions=["minimize", "maximize"],
        sampler_seed=42
    )
    
    print(f"Study directions: {study.directions}")
    print(f"Is multi-objective: {study.is_multi_objective}")
    
    # Run optimization
    result_study = study.optimize(mixed_objective_function)
    
    # Get results
    print(f"\nCompleted {len(result_study.trials)} trials")
    print(f"Pareto front size: {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials[:3]):  # Show first 3
        print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
    
    return study

def run_study_tests():
    """Run all study tests."""
    try:
        print("Starting Multi-Objective HyperbandStudy Tests...")
        
        # Test 1: Multi-objective single iteration
        study1 = test_multi_objective_study_single_iteration()
        
        # Test 2: Single-objective single iteration (new style)
        study2 = test_single_objective_study_single_iteration()
        
        # Test 3: Single-objective backward compatible
        study3 = test_single_objective_study_backward_compatible()
        
        # Test 4: Multi-objective multiple iterations
        study4 = test_multi_objective_study_multiple_iterations()
        
        # Test 5: Multi-objective parallel
        study5 = test_multi_objective_study_parallel()
        
        # Test 6: Mixed objectives
        study6 = test_mixed_objectives()
        
        print("\n" + "=" * 80)
        print("ALL STUDY TESTS PASSED!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_study_tests()
    exit(0 if success else 1) 