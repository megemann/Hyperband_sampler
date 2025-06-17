import numpy as np
import matplotlib.pyplot as plt
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna._transform import _SearchSpaceTransform
import optuna
from optuna.exceptions import TrialPruned
from optuna.logging import get_verbosity

def _dominates_hyperband(values1, values2, directions):
    """Check if values1 dominates values2 in multi-objective optimization."""
    if len(values1) != len(values2) or len(values1) != len(directions):
        return False
    
    better_in_any = False
    for v1, v2, direction in zip(values1, values2, directions):
        if direction == "minimize":
            if v1 > v2:  # worse in this objective
                return False
            elif v1 < v2:  # better in this objective
                better_in_any = True
        else:  # maximize
            if v1 < v2:  # worse in this objective
                return False
            elif v1 > v2:  # better in this objective
                better_in_any = True
    
    return better_in_any

def _get_pareto_front_hyperband(trial_results, directions):
    """Get Pareto front from trial results for multi-objective optimization."""
    if not trial_results:
        return []
    
    pareto_trials = []
    for trial_id, values in trial_results:
        is_dominated = False
        for other_trial_id, other_values in trial_results:
            if trial_id == other_trial_id:
                continue
            if _dominates_hyperband(other_values, values, directions):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_trials.append((trial_id, values))
    
    return pareto_trials

class HyperbandSampler(BaseSampler):
    """Simplified Hyperband algorithm implementation as an Optuna sampler.
    
    Hyperband is a bandit-based approach to hyperparameter optimization that 
    dynamically allocates resources to promising configurations and quickly 
    eliminates poor ones. Supports both single and multi-objective optimization.
    
    Args:
        min_resource (int): Minimum resource to allocate to a trial (e.g., 1 epoch)
        max_resource (int): Maximum resource to allocate to a trial (e.g., 100 epochs)
        reduction_factor (int): Factor by which to reduce number of configurations
            between rungs. Typical values are 3 or 4.
        seed (int, optional): Random seed for sampling. Defaults to None.
        verbose (int, optional): Verbosity level. Defaults to 2.
    """
    
    def __init__(self, min_resource: int, max_resource: int, reduction_factor: int = 3, seed: int = 42):
        super().__init__()
        # Store parameters
        self._min_resource = min_resource
        self._max_resource = max_resource  
        self._reduction_factor = min(25, max(reduction_factor, 2))
        
        # Initialize random state
        self._rng = LazyRandomState(seed)
        
        # Pre-compute bracket structure
        self._brackets = self._generate_brackets()

        # Set verbosity level
        self._verbose = get_verbosity()
        
        # Simple state tracking
        self._current_bracket = 0
        self._current_rung = 0
        self._trial_results = {}  # trial_id -> (bracket_id, rung_id, params, values)
        self._rung_trials = {}    # (bracket_id, rung_id) -> [trial_ids]
        self._promoted_configs = {}  # (bracket_id, rung_id) -> [configs]
        self._hyperband_complete = False  # Flag to track if all brackets are done
        
        # Multi-objective support
        self._directions = None  # Will be set when first trial completes
        self._is_multi_objective = False

    def _generate_brackets(self):
        """Generate Hyperband bracket structure."""
        max_resource = self._max_resource
        min_resource = self._min_resource
        reduction_factor = self._reduction_factor

        s_max = int(np.floor(np.log(max_resource / min_resource) / np.log(reduction_factor)))
        
        brackets = []
        for s in range(s_max + 1):
            bracket = {'bracket_id': s, 'rungs': []}
            
            # Number of rungs in this bracket
            num_rungs = s_max - s + 1 
            
            # Calculate initial number of configurations for this bracket
            n_initial = int(np.ceil((s_max + 1) * (reduction_factor ** (s_max - s)) / (s_max - s + 1)))
            for i in range(num_rungs):
                resource = min(min_resource * (reduction_factor ** (s + i)), max_resource)
                n_configs = max(1, int(n_initial / (reduction_factor ** i)))
                
                bracket['rungs'].append({
                    'rung_id': i,
                    'resource': resource,
                    'n_configs': n_configs
                })

            bracket['rungs'].append({
                    'rung_id': i+1,
                    'resource': max_resource,
                    'n_configs': 1
            })
            
            brackets.append(bracket)
        
        return brackets
        
    def visualize_brackets(self):
        """Simple visualization of Hyperband bracket structure"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self._brackets)))
        
        print(f"Hyperband Structure: {len(self._brackets)} brackets")
        print(f"Min resource: {self._min_resource}, Max resource: {self._max_resource}")
        print(f"Reduction factor: {self._reduction_factor}")
        print("-" * 60)
        
        for bracket_idx, bracket in enumerate(self._brackets):
            color = colors[bracket_idx]
            
            resources = [rung['resource'] for rung in bracket['rungs']]
            n_configs = [rung['n_configs'] for rung in bracket['rungs']]
            
            # Calculate total cost for this bracket
            total_cost = sum(rung['n_configs'] * rung['resource'] for rung in bracket['rungs'])
            
            # Determine bracket type
            if len(bracket['rungs']) == 1:
                bracket_type = "Conservative"
            elif len(bracket['rungs']) == len(self._brackets):
                bracket_type = "Aggressive" 
            else:
                bracket_type = "Balanced"
            
            # Plot line for this bracket
            ax.plot(resources, n_configs, 'o-', linewidth=3, markersize=10, 
                   color=color, label=f'Bracket {bracket_idx} ({bracket_type})', alpha=0.8)
            
            # Add text annotations
            for rung in bracket['rungs']:
                ax.annotate(f'{rung["n_configs"]}', 
                           (rung['resource'], rung['n_configs']),
                           textcoords="offset points", xytext=(0,15), 
                           ha='center', fontsize=10, fontweight='bold')
            
            # Print bracket info with cost
            print(f"Bracket {bracket_idx} ({bracket_type}): {len(bracket['rungs'])} rungs, Cost: {total_cost} epoch-trials")
            for rung in bracket['rungs']:
                print(f"  Rung {rung['rung_id']}: {rung['n_configs']} trials × {rung['resource']} epochs")
            print()
        
        ax.set_xlabel('Resource (Epochs)', fontsize=12)
        ax.set_ylabel('Number of Trials', fontsize=12)
        ax.set_title('Hyperband Bracket Structure\n(Conservative → Aggressive)', fontsize=14)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def is_complete(self):
        """Check if the Hyperband algorithm has completed all brackets."""
        
        return self._hyperband_complete
    
    def get_total_trials_needed(self):
        """Calculate the exact number of trials needed to complete all brackets.
        
        This is useful for setting n_trials in study.optimize() to avoid 
        unnecessary pruned trials after Hyperband completes.
        
        Returns:
            int: Total number of trials needed across all brackets and rungs.
        """
        total_trials = 0
        
        for bracket in self._brackets:
            bracket_trials = 0
            for rung in bracket['rungs']:
                bracket_trials += rung['n_configs']
            total_trials += bracket_trials
            
        return total_trials
    
    def get_bracket_breakdown(self):
        """Get a detailed breakdown of trials needed per bracket.
        
        Returns:
            dict: Dictionary with bracket details and trial counts.
        """
        breakdown = {
            'total_trials': 0,
            'total_brackets': len(self._brackets),
            'brackets': []
        }
        
        for b_idx, bracket in enumerate(self._brackets):
            bracket_info = {
                'bracket_id': b_idx,
                'total_trials': 0,
                'rungs': []
            }
            
            for r_idx, rung in enumerate(bracket['rungs']):
                rung_info = {
                    'rung_id': r_idx,
                    'n_configs': rung['n_configs'],
                    'resource': rung['resource'],
                    'cost': rung['n_configs'] * rung['resource']  # Total epoch-trials
                }
                bracket_info['rungs'].append(rung_info)
                bracket_info['total_trials'] += rung['n_configs']
            
            breakdown['brackets'].append(bracket_info)
            breakdown['total_trials'] += bracket_info['total_trials']
        
        return breakdown
    
    def infer_relative_search_space(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        return {}
    
    def sample_relative(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial, search_space: dict):
        return {}
    
    def sample_independent(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial, param_name: str, param_distribution: optuna.distributions.BaseDistribution):
        trial_id = trial._trial_id
        
        # Handle resource parameter - return the resource for current rung
        if param_name == 'resource':
            # If Hyperband is complete or indices are out of bounds, use max resource
            if (self._hyperband_complete or 
                self._current_bracket >= len(self._brackets)):
                resource = self._max_resource
                if self._verbose <= optuna.logging.WARNING:
                    print(f"Post-completion: resource={resource} for trial {trial_id}")
                return resource
            
            # Check rung bounds after ensuring bracket is valid
            if self._current_rung >= len(self._brackets[self._current_bracket]['rungs']):
                resource = self._max_resource
                if self._verbose <= optuna.logging.WARNING:
                    print(f"Post-completion: resource={resource} for trial {trial_id}")
                return resource
            
            resource = self._brackets[self._current_bracket]['rungs'][self._current_rung]['resource']
            return resource
        
        # If Hyperband is complete, just do random sampling
        if self._hyperband_complete:
            search_space = {param_name: param_distribution}
            trans = _SearchSpaceTransform(search_space)
            trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
            param_value = trans.untransform(trans_params)[param_name]
            if self._verbose <= optuna.logging.WARNING:
                print(f"Post-completion: {param_name}={param_value} for trial {trial_id}")
            return param_value
        
        # Check if we have promoted configs for this rung
        rung_key = (self._current_bracket, self._current_rung)
        
        # If this is the first rung or no promoted configs, random sample
        if self._current_rung == 0 or rung_key not in self._promoted_configs:
            search_space = {param_name: param_distribution}
            trans = _SearchSpaceTransform(search_space)
            trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
            param_value = trans.untransform(trans_params)[param_name]
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Random: {param_name}={param_value} for trial {trial_id} in rung {self._current_rung}")
            return param_value
        
        # Use promoted config
        promoted_configs = self._promoted_configs[rung_key]
        if promoted_configs:
            # Get the count of trials already assigned to this rung
            if rung_key not in self._rung_trials:
                self._rung_trials[rung_key] = []
            
            # The current trial is already in _rung_trials, so subtract 1 for the index
            trial_index = len(self._rung_trials[rung_key]) - 1
            config_index = trial_index % len(promoted_configs)
            config = promoted_configs[config_index]
            
            if param_name in config:
                param_value = config[param_name]
                if self._verbose <= optuna.logging.DEBUG:
                    print(f"Promoted: {param_name}={param_value} for trial {trial_id} (config {config_index})")
                return param_value
        
        # Fallback to random if no promoted value for this param
        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])
        param_value = trans.untransform(trans_params)[param_name]
        if self._verbose <= optuna.logging.WARNING:
            print(f"Fallback: {param_name}={param_value} for trial {trial_id}")
        return param_value
        
    def before_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # If Hyperband algorithm is complete, raise TrialPruned to stop optimization
        if self._hyperband_complete:
            if self._verbose <= optuna.logging.ERROR:  # ERROR level, want to print algorithm completion
                print("Hyperband algorithm has completed all brackets.")
            raise TrialPruned("Hyperband algorithm completed")
        
        trial_id = trial._trial_id
        rung_key = (self._current_bracket, self._current_rung)
        
        # Add trial to current rung
        if rung_key not in self._rung_trials:
            self._rung_trials[rung_key] = []
        self._rung_trials[rung_key].append(trial_id)
        
        if self._verbose <= optuna.logging.DEBUG:
            print(f"Trial {trial_id} assigned to bracket {self._current_bracket}, rung {self._current_rung}")

    def after_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial, state: optuna.trial.TrialState, values: list[float]):
        """Core Hyperband logic: complete rung, promote best configs. Supports multi-objective."""
        trial_id = trial._trial_id
        
        # Skip failed trials
        if state == optuna.trial.TrialState.FAIL or not values:
            if self._verbose <= optuna.logging.ERROR:
                print(f"Trial {trial_id} failed or has no value")
            return
        
        # If Hyperband is complete, just record the result but don't do any processing
        if self._hyperband_complete:
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Post-completion trial {trial_id} completed with values: {values}")
            return
        
        # Initialize directions on first trial
        if self._directions is None:
            # Check if study has directions (multi-objective) or direction (single-objective)
            if hasattr(study, 'directions'):
                self._directions = study.directions
                self._is_multi_objective = len(self._directions) > 1
            elif hasattr(study, 'direction'):
                self._directions = [study.direction]
                self._is_multi_objective = False
            else:
                # Fallback: shouldn't happen in normal Optuna usage
                self._directions = ["minimize"]
                self._is_multi_objective = False
                
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Initialized directions: {self._directions}, multi-objective: {self._is_multi_objective}")
        
        # Store trial result (keep all values for multi-objective)
        params = {key: val for key, val in trial.params.items() if key != 'resource'}
        self._trial_results[trial_id] = (self._current_bracket, self._current_rung, params, values)
        
        # Check if we've completed all trials for this rung
        rung_key = (self._current_bracket, self._current_rung)
        current_rung = self._brackets[self._current_bracket]['rungs'][self._current_rung]
        expected_trials = current_rung['n_configs']
        completed_trials = len([tid for tid in self._rung_trials.get(rung_key, []) 
                               if tid in self._trial_results])
        
        if self._verbose <= optuna.logging.DEBUG:
            print(f"Completed {completed_trials}/{expected_trials} trials in bracket {self._current_bracket}, rung {self._current_rung}")
        
        # If rung is complete, promote best configs
        if completed_trials >= expected_trials:
            self._promote_configs(study)
            self._advance_to_next_rung()
    
    def _promote_configs(self, study):
        """Promote the best configs from current rung to next rung. Supports multi-objective."""
        rung_key = (self._current_bracket, self._current_rung)
        bracket = self._brackets[self._current_bracket]
        
        # If this is the last rung, nothing to promote
        if self._current_rung >= len(bracket['rungs']) - 1:
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Last rung in bracket {self._current_bracket} - no promotion needed")
            return
        
        # Get all trials from this rung
        trial_ids = self._rung_trials.get(rung_key, [])
        trial_results = [(tid, self._trial_results[tid][3]) for tid in trial_ids 
                        if tid in self._trial_results]
        
        if not trial_results:
            return
        
        # Determine how many to promote
        next_rung = bracket['rungs'][self._current_rung + 1]
        n_promote = next_rung['n_configs']
        
        if self._is_multi_objective:
            # Multi-objective: use Pareto front
            pareto_front = _get_pareto_front_hyperband(trial_results, self._directions)
            
            if len(pareto_front) <= n_promote:
                # All Pareto solutions can be promoted
                promoted_trials = pareto_front
            else:
                # Need to select subset of Pareto front
                # For simplicity, take first n_promote solutions
                # In practice, you might want more sophisticated selection
                promoted_trials = pareto_front[:n_promote]
                
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Multi-objective: Pareto front size {len(pareto_front)}, promoting {len(promoted_trials)}")
        else:
            # Single objective: sort by value
            # Get direction from study or use first direction
            if hasattr(study, 'direction'):
                direction = study.direction
            else:
                direction = self._directions[0] if self._directions else "minimize"
            
            reverse_sort = (direction == "maximize")
            trial_results.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            # Take top trials
            promoted_trials = trial_results[:n_promote]
            
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Single-objective: promoting top {len(promoted_trials)} trials")
        
        # Extract their configs
        promoted_configs = []
        for trial_id, values in promoted_trials:
            params = self._trial_results[trial_id][2]  # Get params without resource
            promoted_configs.append(params)
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Promoting trial {trial_id} (values={values}) to next rung")
        
        # Store promoted configs for next rung
        next_rung_key = (self._current_bracket, self._current_rung + 1)
        self._promoted_configs[next_rung_key] = promoted_configs
        
        if self._verbose <= optuna.logging.DEBUG:
            print(f"Promoted {len(promoted_configs)} configs to rung {self._current_rung + 1}")
    
    def _advance_to_next_rung(self):
        """
        Move to the next rung or bracket.
        
        Handles all modular variables.
        """
        bracket = self._brackets[self._current_bracket]
        
        #  Case we have more rungs in the current bracket
        if self._current_rung < len(bracket['rungs']) - 1:
            self._current_rung += 1
            if self._verbose <= optuna.logging.DEBUG:
                print(f"Advanced to rung {self._current_rung} in bracket {self._current_bracket}")

        # Case we have no more rungs in the current bracket
        else:
            # Move to next bracket
            self._current_bracket += 1
            self._current_rung = 0
            
            if self._current_bracket >= len(self._brackets):
                # All brackets completed - Hyperband is done!
                self._hyperband_complete = True
                if self._verbose <= optuna.logging.DEBUG:
                    print("Hyperband algorithm completed! All brackets finished.")
            else:
                if self._verbose <= optuna.logging.DEBUG:
                    print(f"Advanced to bracket {self._current_bracket}, rung 0")