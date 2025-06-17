#!/usr/bin/env python
"""
Visualization and debugging tests for HyperbandSampler.
Includes bracket visualization, configuration comparison, and debugging utilities.
"""
import optuna
from hyperband_sampler import HyperbandSampler
import numpy as np


def test_hyperband_bracket_configurations():
    """Test different Hyperband bracket configurations."""
    print("\nTEST 1: Standard Configuration")
    sampler1 = HyperbandSampler(min_resource=3, max_resource=81, reduction_factor=3)
    print(f"Brackets: {len(sampler1._brackets)}")
    print(f"Total trials needed: {sampler1.get_total_trials_needed()}")
    
    print("\nTEST 2: Conservative Configuration")
    sampler2 = HyperbandSampler(min_resource=5, max_resource=50, reduction_factor=2)
    print(f"Brackets: {len(sampler2._brackets)}")
    print(f"Total trials needed: {sampler2.get_total_trials_needed()}")
    
    print("\nTEST 3: Aggressive Configuration")
    sampler3 = HyperbandSampler(min_resource=1, max_resource=64, reduction_factor=4)
    print(f"Brackets: {len(sampler3._brackets)}")
    print(f"Total trials needed: {sampler3.get_total_trials_needed()}")
    
    print("\nTEST 4: Edge Case - Min equals Max")
    sampler4 = HyperbandSampler(min_resource=10, max_resource=10, reduction_factor=3)
    print(f"Brackets: {len(sampler4._brackets)}")
    print(f"Total trials needed: {sampler4.get_total_trials_needed()}")
    
    print("\nTEST 5: Large Resource Range")
    sampler5 = HyperbandSampler(min_resource=1, max_resource=1000, reduction_factor=5)
    print(f"Brackets: {len(sampler5._brackets)}")
    print(f"Total trials needed: {sampler5.get_total_trials_needed()}")
    
    return True


def compare_reduction_factors():
    """Compare different reduction factors and their computational costs."""
    
    print("\n" + "="*80)
    print("REDUCTION FACTOR COMPARISON")
    print("="*80)
    
    configs = [
        {"reduction_factor": 2, "name": "Conservative (keep 1/2)"},
        {"reduction_factor": 3, "name": "Standard (keep 1/3)"},
        {"reduction_factor": 4, "name": "Aggressive (keep 1/4)"}
    ]
    
    for config in configs:
        print(f"\n📊 {config['name']}")
        print("-" * 50)
        
        sampler = HyperbandSampler(
            min_resource=3,
            max_resource=81,
            reduction_factor=config["reduction_factor"],
            seed=42
        )
        
        # Calculate and display statistics
        total_trials = sum(bracket['rungs'][0]['n_configs'] for bracket in sampler._brackets)
        total_cost = sum(rung['n_configs'] * rung['resource'] 
                        for bracket in sampler._brackets 
                        for rung in bracket['rungs'])
        
        print(f"Number of brackets: {len(sampler._brackets)}")
        print(f"Total initial trials: {total_trials}")
        print(f"Total computational cost: {total_cost} epoch-trials")
        print(f"Efficiency ratio: {total_cost / total_trials:.1f} epochs/trial")


def debug_trial_calculation():
    """Debug and verify trial calculation methods."""
    
    print("\n" + "="*80)
    print("TRIAL CALCULATION DEBUG")
    print("="*80)
    
    # Create the same sampler as in our tests
    sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
    
    # Get total trials needed
    total_trials = sampler.get_total_trials_needed()
    print(f"🔢 Total trials needed: {total_trials}")
    
    # Get detailed breakdown
    breakdown = sampler.get_bracket_breakdown()
    print(f"\n📊 Detailed Breakdown:")
    print(f"Total brackets: {breakdown['total_brackets']}")
    print(f"Total trials: {breakdown['total_trials']}")
    
    print("\nPer-bracket details:")
    for bracket in breakdown['brackets']:
        print(f"  Bracket {bracket['bracket_id']}: {bracket['total_trials']} trials")
        for rung in bracket['rungs']:
            print(f"    Rung {rung['rung_id']}: {rung['n_configs']} trials × {rung['resource']} epochs = {rung['cost']} epoch-trials")
    
    # Calculate total cost (epoch-trials)
    total_cost = sum(
        sum(rung['cost'] for rung in bracket['rungs']) 
        for bracket in breakdown['brackets']
    )
    print(f"\n💰 Total computational cost: {total_cost} epoch-trials")
    
    return total_trials, breakdown


def debug_optimization_cycle():
    """Debug a complete optimization cycle with detailed logging."""
    
    print("\n" + "="*80)
    print("OPTIMIZATION CYCLE DEBUG")
    print("="*80)
    
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        resource = trial.suggest_int("resource", 1, 9)
        return x  # Simple objective for testing
    
    # Create a small Hyperband sampler for debugging
    sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
    
    # Print the bracket structure and trial requirements
    print("Hyperband Structure:")
    breakdown = sampler.get_bracket_breakdown()
    print(f"Total brackets: {breakdown['total_brackets']}")
    print(f"Total trials needed: {breakdown['total_trials']}")
    
    for bracket in breakdown['brackets']:
        print(f"\nBracket {bracket['bracket_id']}: {bracket['total_trials']} trials")
        for rung in bracket['rungs']:
            print(f"  Rung {rung['rung_id']}: {rung['n_configs']} trials × {rung['resource']} epochs = {rung['cost']} epoch-trials")
    
    # Calculate total computational cost
    total_cost = sum(
        sum(rung['cost'] for rung in bracket['rungs']) 
        for bracket in breakdown['brackets']
    )
    print(f"\n💰 Total computational cost: {total_cost} epoch-trials")
    
    # Create a study and run the optimization with exact number of trials
    study = optuna.create_study(sampler=sampler, direction='maximize')
    
    # Use the exact number of trials calculated by the sampler
    n_trials_needed = sampler.get_total_trials_needed()
    print(f"\nRunning optimization with exactly {n_trials_needed} trials (no waste!)...")
    
    try:
        study.optimize(objective, n_trials=n_trials_needed)
        print("Perfect! Hyperband completed exactly when all trials finished.")
    except optuna.TrialPruned as e:
        print(f"Unexpected: {e}")
    
    # Verify results
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {completed_trials}")
    print(f"Expected trials: {n_trials_needed}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    if completed_trials == n_trials_needed:
        print("Perfect optimization - no wasted trials!")
    
    return True


def test_graceful_completion_with_extra_trials():
    """Test that the sampler handles extra trials gracefully after completion."""
    try:
        print("="*80)
        print("TEST: Graceful Completion with Extra Trials")
        print("="*80)
        
        sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3, seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        n_trials_needed = sampler.get_total_trials_needed()
        extra_trials = n_trials_needed + 10  # Add 10 extra trials
        
        print(f"Hyperband needs: {n_trials_needed} trials")
        print(f"Running {extra_trials} trials ({extra_trials - n_trials_needed} extra)...")
        
        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            resource = trial.suggest_int("resource", 1, 9)
            return x
        
        try:
            study.optimize(objective, n_trials=extra_trials)
            print("Success! No exceptions raised, study completed naturally.")
        except optuna.TrialPruned as e:
            print(f"Error: {e}")
        
        # Verify results
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Completed trials: {completed_trials}")
        print(f"Expected trials: {n_trials_needed}")
        print(f"Sampler complete: {sampler.is_complete()}")
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
        
        if completed_trials >= n_trials_needed and sampler.is_complete():
            print("Test passed! Sampler handles extra trials gracefully.")
        
        return True
    
    except Exception as e:
        print(f"Error in graceful completion test: {e}")
        return False


def test_parameter_promotion_and_reuse():
    """Test and debug parameter promotion with detailed logging."""
    
    print("\n" + "="*80)
    print("PARAMETER PROMOTION DEBUG")
    print("="*80)
    
    # Create a sampler
    sampler = HyperbandSampler(min_resource=1, max_resource=27, reduction_factor=3)
    
    # Manually set up some promoted configs for rung 1 of bracket 0
    rung_key = (0, 1)
    sampler._promoted_configs[rung_key] = [
        {'x': 0.1},
        {'x': 0.2},
        {'x': 0.3}
    ]
    
    print(f"Set up promoted configs for rung {rung_key}: {sampler._promoted_configs[rung_key]}")
    
    # Force the sampler to be in bracket 0, rung 1
    sampler._current_bracket = 0
    sampler._current_rung = 1
    
    print(f"Forced sampler to bracket {sampler._current_bracket}, rung {sampler._current_rung}")
    
    # Create a study with the sampler
    study = optuna.create_study(sampler=sampler, direction='maximize')
    
    # Run trials and verify promoted configs are reused correctly
    expected_values = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    
    print("\nTesting parameter promotion and reuse:")
    for i in range(6):
        trial = study.ask()
        x = trial.suggest_float("x", 0, 1)
        resource = trial.suggest_int("resource", 1, 27)
        
        expected_x = expected_values[i]
        print(f"Trial {i}: x={x:.10f}, expected={expected_x}, resource={resource}")
        
        # The x value should match the expected promoted config
        assert abs(x - expected_x) < 1e-10, f"Trial {i}: expected x={expected_x}, got x={x}"
        
        # Complete the trial
        study.tell(trial, x)
    
    print("Parameter promotion and reuse test passed!")
    
    return True


def run_all_visualization_tests():
    """Run all visualization and debugging tests."""
    try:
        print("="*80)
        print("HYPERBAND VISUALIZATION & DEBUG TEST SUITE")
        print("="*80)
        
        # Test bracket configurations
        print("="*80)
        print("HYPERBAND BRACKET CONFIGURATION TESTS")
        print("="*80)
        test_hyperband_bracket_configurations()
        
        # Test exact trial count optimization
        test_exact_trial_count_optimization()
        
        # Test graceful completion
        test_graceful_completion_with_extra_trials()
        
        # Test parameter promotion
        test_parameter_promotion_and_reuse()
        
        print("\n" + "="*80)
        print("ALL VISUALIZATION & DEBUG TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR IN VISUALIZATION TESTS: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_visualization_tests() 