#!/usr/bin/env python
"""
Comprehensive test suite for HyperbandSampler.
Combines all unit tests, integration tests, and validation tests.
"""
import unittest
import optuna
from hyperband_sampler import HyperbandSampler
import numpy as np


class TestHyperbandSampler(unittest.TestCase):
    """Unit tests for HyperbandSampler functionality."""
    
    def setUp(self):
        self.sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
    
    def test_bracket_generation(self):
        """Test that brackets are generated correctly."""
        brackets = self.sampler._brackets
        
        # Should have multiple brackets with different strategies
        self.assertGreater(len(brackets), 0)
        
        # Each bracket should have rungs with decreasing n_configs and increasing resources
        for bracket in brackets:
            rungs = bracket['rungs']
            self.assertGreater(len(rungs), 0)
            
            prev_n_configs = float('inf')
            prev_resource = 0
            
            for rung in rungs:
                self.assertGreater(rung['n_configs'], 0)
                self.assertGreater(rung['resource'], 0)
                self.assertLessEqual(rung['n_configs'], prev_n_configs)
                self.assertGreaterEqual(rung['resource'], prev_resource)
                
                prev_n_configs = rung['n_configs']
                prev_resource = rung['resource']
    
    def test_resource_parameter(self):
        """Test that resource parameter is handled correctly."""
        study = optuna.create_study(sampler=self.sampler)
        
        trial = study.ask()
        resource = trial.suggest_int("resource", 1, 9)
        
        # Resource should match the current rung's resource
        expected_resource = self.sampler._brackets[0]['rungs'][0]['resource']
        self.assertEqual(resource, expected_resource)
    
    def test_random_sampling_first_rung(self):
        """Test that first rung uses random sampling."""
        study = optuna.create_study(sampler=self.sampler)
        
        # In first rung, should use random sampling
        trial = study.ask()
        x = trial.suggest_float("x", 0, 1)
        
        # Should be within bounds
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 1)
    
    def test_parameter_promotion_setup(self):
        """Test that parameter promotion works correctly."""
        # Set up promoted configs for rung 1
        rung_key = (0, 1)
        self.sampler._promoted_configs[rung_key] = [
            {'x': 0.123},
            {'x': 0.456}
        ]
        
        # Force sampler to be in rung 1
        self.sampler._current_bracket = 0
        self.sampler._current_rung = 1
        
        study = optuna.create_study(sampler=self.sampler)
        
        # First trial should get first promoted config
        trial1 = study.ask()
        x1 = trial1.suggest_float("x", 0, 1)
        self.assertAlmostEqual(x1, 0.123, places=5)
        
        # Second trial should get second promoted config
        trial2 = study.ask()
        x2 = trial2.suggest_float("x", 0, 1)
        self.assertAlmostEqual(x2, 0.456, places=5)
    
    def test_trial_calculation_methods(self):
        """Test that trial calculation methods work correctly."""
        # Get total trials needed
        total_trials = self.sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)
        
        # Get detailed breakdown
        breakdown = self.sampler.get_bracket_breakdown()
        self.assertIn('total_brackets', breakdown)
        self.assertIn('total_trials', breakdown)
        self.assertIn('brackets', breakdown)
        
        # Verify breakdown consistency
        calculated_total = sum(bracket['total_trials'] for bracket in breakdown['brackets'])
        self.assertEqual(calculated_total, breakdown['total_trials'])
        self.assertEqual(calculated_total, total_trials)
    
    def test_direct_parameter_promotion(self):
        """Test direct parameter promotion functionality."""
        # Create a smaller sampler for testing
        sampler = HyperbandSampler(min_resource=1, max_resource=4, reduction_factor=2)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        
        # Set up promoted configs directly
        bracket_id = 0
        rung_id = 1
        
        # Define specific parameter values
        param1 = 0.123
        param2 = 0.456
        
        # Set promoted configs using the correct key format
        rung_key = (bracket_id, rung_id)
        sampler._promoted_configs[rung_key] = [
            {"x": param1},
            {"x": param2}
        ]
        
        # Set current rung to the one with promoted configs
        sampler._current_bracket = bracket_id
        sampler._current_rung = rung_id
        
        # Create and register first trial
        trial1 = study.ask()
        x1 = trial1.suggest_float("x", 0, 1)
        
        # Create and register second trial
        trial2 = study.ask()
        x2 = trial2.suggest_float("x", 0, 1)
        
        # Verify both parameters match expected values
        self.assertAlmostEqual(x1, param1, places=5)
        self.assertAlmostEqual(x2, param2, places=5)


class TestHyperbandIntegration(unittest.TestCase):
    """Integration tests for complete Hyperband optimization cycles."""
    
    def objective(self, trial):
        """Simple objective function for testing."""
        x = trial.suggest_float("x", 0, 1)
        resource = trial.suggest_int("resource", 1, 9)
        return x  # Simple objective for testing
    
    def test_full_optimization_cycle(self):
        """Test a complete optimization cycle."""
        sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        # Run optimization for multiple trials
        study.optimize(self.objective, n_trials=20)
        
        # Should have completed some trials
        self.assertGreater(len(study.trials), 0)
        
        # Should have found a reasonable best value
        self.assertIsNotNone(study.best_value)
        self.assertGreaterEqual(study.best_value, 0)
        self.assertLessEqual(study.best_value, 1)
    
    def test_exact_trial_count_optimization(self):
        """Test optimization with exact trial count needed."""
        sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
        n_trials_needed = sampler.get_total_trials_needed()
        
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        try:
            study.optimize(self.objective, n_trials=n_trials_needed)
        except optuna.TrialPruned:
            pass  # Expected when Hyperband completes
        
        # Verify we completed exactly the right number of trials
        self.assertEqual(len(study.trials), n_trials_needed)
        self.assertTrue(sampler.is_complete())
    
    def test_graceful_completion_with_extra_trials(self):
        """Test that sampler handles extra trials gracefully after completion."""
        sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
        n_trials_needed = sampler.get_total_trials_needed()
        
        study = optuna.create_study(sampler=sampler, direction='maximize')
        extra_trials = n_trials_needed + 5  # Add 5 extra trials
        
        # Should handle extra trials gracefully by raising TrialPruned
        try:
            study.optimize(self.objective, n_trials=extra_trials)
        except optuna.TrialPruned:
            # This is expected when Hyperband completes
            pass
        
        # Verify we completed at least the required number of trials
        self.assertGreaterEqual(len(study.trials), n_trials_needed)
        self.assertTrue(sampler.is_complete())
    
    def test_promotion_logic(self):
        """Test that promotion logic works correctly."""
        sampler = HyperbandSampler(min_resource=1, max_resource=9, reduction_factor=3)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        # Run enough trials to complete first rung and trigger promotion
        study.optimize(self.objective, n_trials=15)
        
        # Check that some promoted configs were created
        has_promoted_configs = any(len(configs) > 0 for configs in sampler._promoted_configs.values())
        self.assertTrue(has_promoted_configs, "Should have created some promoted configs")
        
        # Check that trial results were stored
        self.assertGreater(len(sampler._trial_results), 0)
    
    def test_parameter_promotion_reuse(self):
        """Test that promoted configurations are reused correctly."""
        sampler = HyperbandSampler(min_resource=1, max_resource=27, reduction_factor=3)
        
        # Manually set up some promoted configs for rung 1 of bracket 0
        rung_key = (0, 1)
        sampler._promoted_configs[rung_key] = [
            {'x': 0.1},
            {'x': 0.2},
            {'x': 0.3}
        ]
        
        # Force the sampler to be in bracket 0, rung 1
        sampler._current_bracket = 0
        sampler._current_rung = 1
        
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        # Run trials and check that promoted configs are reused correctly
        expected_values = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        
        for i in range(6):
            trial = study.ask()
            x = trial.suggest_float("x", 0, 1)
            resource = trial.suggest_int("resource", 1, 27)
            
            expected_x = expected_values[i]
            self.assertAlmostEqual(x, expected_x, places=10)
            
            # Complete the trial
            study.tell(trial, x)


class TestHyperbandConfigurations(unittest.TestCase):
    """Test different Hyperband configurations and edge cases."""
    
    def test_standard_configuration(self):
        """Test standard Hyperband configuration."""
        sampler = HyperbandSampler(
            min_resource=3,
            max_resource=81, 
            reduction_factor=3,
            seed=42
        )
        
        self.assertGreater(len(sampler._brackets), 0)
        total_trials = sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)
    
    def test_conservative_configuration(self):
        """Test conservative configuration with smaller reduction factor."""
        sampler = HyperbandSampler(
            min_resource=5,
            max_resource=50,
            reduction_factor=2,
            seed=42
        )
        
        self.assertGreater(len(sampler._brackets), 0)
        total_trials = sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)
    
    def test_aggressive_configuration(self):
        """Test aggressive configuration with larger reduction factor."""
        sampler = HyperbandSampler(
            min_resource=1,
            max_resource=64,
            reduction_factor=4,
            seed=42
        )
        
        self.assertGreater(len(sampler._brackets), 0)
        total_trials = sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)
    
    def test_edge_case_min_equals_max(self):
        """Test edge case where min_resource equals max_resource."""
        sampler = HyperbandSampler(
            min_resource=5,
            max_resource=5,
            reduction_factor=3,
            seed=2
        )
        
        # Should still create valid brackets
        self.assertGreater(len(sampler._brackets), 0)
        total_trials = sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)
    
    def test_edge_case_large_resource_range(self):
        """Test edge case with very large resource range."""
        sampler = HyperbandSampler(
            min_resource=1,
            max_resource=100,
            reduction_factor=2,
            seed=2
        )
        
        self.assertGreater(len(sampler._brackets), 0)
        total_trials = sampler.get_total_trials_needed()
        self.assertGreater(total_trials, 0)


def run_all_tests():
    """Run all test suites."""
    # Create test suites
    unit_suite = unittest.TestLoader().loadTestsFromTestCase(TestHyperbandSampler)
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestHyperbandIntegration)
    config_suite = unittest.TestLoader().loadTestsFromTestCase(TestHyperbandConfigurations)
    
    # Combine all suites
    all_tests = unittest.TestSuite([unit_suite, integration_suite, config_suite])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE HYPERBAND SAMPLER TEST SUITE")
    print("="*80)
    
    success = run_all_tests()
    
    if success:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
    
    print("="*80) 