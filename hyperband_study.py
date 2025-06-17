import optuna
from hyperband_sampler import HyperbandSampler
import os
import concurrent.futures
import threading
import numpy as np
import time

'''
    This file is the main file for the HyperbandStudy class.
    It is a wrapper for an Optuna Study

    Core Functionality: (33% of the code)
    __init__: Intialize all variables and create the study.
    optimize: Run the optimization.
    _serial_optimize: Run single iteration of Hyperband serially. Made for hyperband_iterations > 1.
    For information on the parameters, see the documentation for the optimize function.
    For single iteration, the study is ran with the sampler and simply returned with some values frozen.
    For multiple iterations, the study is ran with the sampler and the results are stored in the iteration_studies list.
    The best trials are then returned.

    Expanded Functionality: (66% of the code)
    _parallel_optimize_threading: Run multiple iterations of Hyperband in parallel using threading.
    _get_pareto_front: Get the Pareto front for multi-objective optimization.
    _run_single_hyperband_iteration: Run a single iteration of Hyperband in parallel.
    _dominates: Check if one trial dominates another.
'''

def _run_single_hyperband_iteration(min_resource, max_resource, reduction_factor, 
                                   directions, sampler_seed, objective, n_trials, 
                                   timeout, catch, callbacks, gc_after_trial, 
                                   show_progress_bar, iteration_idx):
    """Helper function to run a single Hyperband iteration - thread-safe."""
    try:
        # Create HyperbandStudy for this iteration
        hyperband_study = HyperbandStudy(
            min_resource=min_resource,
            max_resource=max_resource, 
            reduction_factor=reduction_factor,
            hyperband_iterations=1,  # Force single iteration
            directions=directions,
            sampler_seed=sampler_seed
        )
        
        # Run optimization
        result_study = hyperband_study.optimize(
            objective=objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
            n_jobs=1  # Each iteration runs single-threaded
        )
        
        # Get best trials from the HyperbandStudy wrapper (not the underlying Optuna study)
        best_trials = hyperband_study.best_trials
        
        return result_study, iteration_idx, best_trials
        
    except Exception as e:
        print(f"Error in iteration {iteration_idx}: {e}")
        return None, iteration_idx, None

def _dominates(values1, values2, directions):
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

def _get_pareto_front(trials, directions):
    """Get Pareto front from a list of trials."""
    if not trials:
        return []
    
    pareto_trials = []
    for trial in trials:
        if trial.values is None:
            continue
            
        is_dominated = False
        for other_trial in trials:
            if other_trial.values is None or trial == other_trial:
                continue
            if _dominates(other_trial.values, trial.values, directions):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_trials.append(trial)
    
    return pareto_trials

class HyperbandStudy:
    
    def __init__(self, min_resource, max_resource, reduction_factor, hyperband_iterations=1, 
                 directions=None, direction=None, study_name=None, storage=None, 
                 load_if_exists=False, sampler_seed=None, **study_kwargs):
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.sampler_seed = sampler_seed
        
        # Handle both single and multi-objective
        if directions is not None and direction is not None:
            raise ValueError("Cannot specify both 'directions' and 'direction'. Use 'directions' for multi-objective.")
        
        if directions is not None:
            if isinstance(directions, str):
                directions = [directions]
            self.directions = directions
            self.is_multi_objective = len(directions) > 1
        elif direction is not None:
            self.directions = [direction]
            self.is_multi_objective = False
        else:
            self.directions = ["minimize"]
            self.is_multi_objective = False
        
        if (hyperband_iterations > 1):
            if (study_name is not None):
                raise ValueError("WARNING: study_name must be None when hyperband_iterations > 1")
            if (storage is not None):
                raise ValueError("WARNING: storage must be None when hyperband_iterations > 1")
            if (load_if_exists):
                raise ValueError("WARNING: load_if_exists must be False when hyperband_iterations > 1")
                
        self.hyperband_iterations = max(1, hyperband_iterations)
        self.sampler = HyperbandSampler(self.min_resource, self.max_resource, self.reduction_factor, seed=sampler_seed)
        
        # Store individual iteration studies
        self._iteration_studies = []
        
        if self.hyperband_iterations == 1:
            if self.is_multi_objective:
                self.study = optuna.create_study(
                    sampler=self.sampler,
                    directions=self.directions,
                    study_name=study_name,
                    storage=storage,
                    load_if_exists=load_if_exists,
                    **study_kwargs
                )
            else:
                self.study = optuna.create_study(
                    sampler=self.sampler,
                    direction=self.directions[0],
                    study_name=study_name,
                    storage=storage,
                    load_if_exists=load_if_exists,
                    **study_kwargs
                )
            self.n_trials = self.sampler.get_total_trials_needed()
        else:
            self.study = None
            self.n_trials = hyperband_iterations * self.sampler.get_total_trials_needed()
           
    def optimize(self, objective, n_trials=None, timeout=None, catch=(), callbacks=None, 
                 gc_after_trial=False, show_progress_bar=False, n_jobs=1):
        """Optimize an objective function.
        
        Parameters:
        -----------
        objective : callable
            Objective function to optimize. For multi-objective, should return tuple/list of values.
        n_trials : int | None
            The number of trials. If None, will use the total trials needed by Hyperband.
        timeout : float | None
            Stop study after the given number of second(s). None means no time limit.
        catch : Iterable[type[Exception]] | type[Exception]
            A study continues even when a trial raises one of these exceptions.
        callbacks : Iterable[Callable[[Study, FrozenTrial], None]] | None
            List of callback functions invoked at the end of each trial.
        gc_after_trial : bool
            Flag to run garbage collection after each trial.
        show_progress_bar : bool
            Flag to show progress bars or not.
        n_jobs : int
            Number of parallel threads. Only used when hyperband_iterations > 1.
            -1 means use all available CPUs.
        
        Returns:
        --------
        optuna.study.Study
            The completed study object.
        """
        if n_trials is None:
            n_trials = self.n_trials
        
        if self.hyperband_iterations == 1:
            if n_jobs > 1 or n_jobs == -1:
                raise ValueError("WARNING: n_jobs > 1 or n_jobs == -1 is not supported for single-iteration Hyperband.")

            try:
                self.study.optimize(
                    objective, 
                    n_trials=n_trials, 
                    timeout=timeout,
                    n_jobs=1,
                    catch=catch, 
                    callbacks=callbacks, 
                    gc_after_trial=gc_after_trial, 
                    show_progress_bar=show_progress_bar
                )
            except optuna.exceptions.TrialPruned:
                    pass  # Handle our custom algorithm stopping

        else:
            if n_jobs > 1 or n_jobs == -1:
                return self._parallel_optimize_threading(
                    objective, n_trials=n_trials, timeout=timeout, catch=catch, 
                    callbacks=callbacks, gc_after_trial=gc_after_trial, 
                    show_progress_bar=show_progress_bar, n_jobs=n_jobs
                )
            else:
                return self._serial_optimize(
                    objective, n_trials=n_trials, timeout=timeout, catch=catch,
                    callbacks=callbacks, gc_after_trial=gc_after_trial,
                    show_progress_bar=show_progress_bar
                )
        
        return self.study

    def _serial_optimize(self, objective, n_trials=None, timeout=None, catch=(), 
                        callbacks=None, gc_after_trial=False, show_progress_bar=False):
        """Run multiple Hyperband iterations serially."""
        all_trials = []
        start_time = time.time()
        
        for i in range(self.hyperband_iterations):
            # Check if timeout has been exceeded before starting new iteration
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                        print(f"Timeout reached ({timeout}s). Stopping after {i} iterations.")
                    break
            
            # Use different seed for each iteration
            iteration_seed = (self.sampler_seed + i) if self.sampler_seed is not None else None
            
            hyperband_study = HyperbandStudy(
                self.min_resource, 
                self.max_resource, 
                self.reduction_factor,
                hyperband_iterations=1,  # Force single iteration
                directions=self.directions,
                sampler_seed=iteration_seed
            )
            
            study = hyperband_study.optimize(
                objective, n_trials=n_trials, timeout=timeout, catch=catch, 
                callbacks=callbacks, gc_after_trial=gc_after_trial, 
                show_progress_bar=show_progress_bar
            )
            
            # Store the individual iteration study
            self._iteration_studies.append(study)
            
            # Collect trials from this iteration
            all_trials.extend(study.trials)
            
            if self.is_multi_objective:
                current_pareto = _get_pareto_front(study.trials, self.directions)
                if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                    print(f"Hyperband iteration {i+1}/{self.hyperband_iterations} completed")
                    print(f"Current iteration Pareto front size: {len(current_pareto)}")
            else:
                try:
                    current_value = study.best_value
                    if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                        print(f"Hyperband iteration {i+1}/{self.hyperband_iterations} completed")
                        print(f"Current iteration best: {current_value}")
                except AttributeError:
                    # Fallback for multi-objective studies that somehow got here
                    if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                        print(f"Hyperband iteration {i+1}/{self.hyperband_iterations} completed")
        
        # Create final study with all trials
        if self.is_multi_objective:
            self.study = optuna.create_study(directions=self.directions)
        else:
            self.study = optuna.create_study(direction=self.directions[0])
        
        # Add all trials to the final study
        for trial in all_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.study.add_trial(trial)
        
        if self.is_multi_objective:
            pareto_front = _get_pareto_front(self.study.trials, self.directions)
            if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                print(f"Final Pareto front size: {len(pareto_front)}")
        else:
            try:
                if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                    print(f"Overall best value: {self.study.best_value}")
            except AttributeError:
                if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                    print("Overall optimization completed (multi-objective study)")
        
        return self.study

    def _parallel_optimize_threading(self, objective, n_trials=None, timeout=None, catch=(), 
                                   callbacks=None, gc_after_trial=False, show_progress_bar=False, n_jobs=-1):
        """Run multiple Hyperband iterations in parallel using threading (GPU-safe)."""
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs > os.cpu_count():
            if (optuna.logging.get_verbosity() <= optuna.logging.WARNING):
                print(f"WARNING: n_jobs ({n_jobs}) > os.cpu_count() ({os.cpu_count()}). Limiting to {os.cpu_count()}.")
            n_jobs = os.cpu_count()
    
        if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
            print(f"Running {self.hyperband_iterations} Hyperband iterations in parallel using {n_jobs} threads...")
            print("Using threading (GPU-safe) instead of multiprocessing.")
        
        start_time = time.time()
        
        # Generate seeds for each iteration
        seeds = []
        for i in range(self.hyperband_iterations):
            if self.sampler_seed is not None:
                seeds.append(self.sampler_seed + i)
            else:
                seeds.append(None)
        
        # Thread-safe results collection
        results = [None] * self.hyperband_iterations
        results_lock = threading.Lock()
        
        def run_iteration_with_result_storage(iteration_idx):
            """Wrapper to store results in thread-safe manner."""
            result = _run_single_hyperband_iteration(
                self.min_resource,
                self.max_resource,
                self.reduction_factor,
                self.directions,
                seeds[iteration_idx],
                objective,
                n_trials,
                timeout,
                catch,
                callbacks,
                gc_after_trial,
                show_progress_bar,
                iteration_idx
            )
            
            with results_lock:
                results[iteration_idx] = result
            
            return result
        
        # Run iterations in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit tasks with timeout checking
            futures = []
            for i in range(self.hyperband_iterations):
                # Check if timeout has been exceeded before submitting new tasks
                if timeout is not None:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout:
                        if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                            print(f"Timeout reached ({timeout}s). Submitting only {i} out of {self.hyperband_iterations} iterations.")
                        break
                
                future = executor.submit(run_iteration_with_result_storage, i)
                futures.append(future)
            
            # Calculate remaining timeout for waiting
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed_time)
            
            # Wait for completion and handle any exceptions
            completed_count = 0
            try:
                for future in concurrent.futures.as_completed(futures, timeout=remaining_timeout):
                    try:
                        future.result()  # This will raise any exception that occurred
                        completed_count += 1
                        if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                            print(f"Thread {completed_count} completed successfully")
                    except Exception as e:
                        if (optuna.logging.get_verbosity() <= optuna.logging.DEBUG):
                            print(f"Thread failed with error: {e}")
            except concurrent.futures.TimeoutError:
                if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                    print(f"Timeout reached ({timeout}s). Completed {completed_count} out of {len(futures)} submitted iterations.")
                # Cancel any remaining futures
                for future in futures:
                    future.cancel()
        
        # Process results and combine all trials
        all_trials = []
        
        for study, iteration_idx, trials in results:
            if study is not None and trials is not None:
                # Store the individual iteration study
                self._iteration_studies.append(study)
                
                all_trials.extend(study.trials)
                if self.is_multi_objective:
                    pareto_size = len(_get_pareto_front(study.trials, self.directions))
                    if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                        print(f"Iteration {iteration_idx + 1} completed with Pareto front size: {pareto_size}")
                else:
                    # For single-objective, we can safely access best_value
                    try:
                        best_val = study.best_value
                        if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                            print(f"Iteration {iteration_idx + 1} completed with best value: {best_val}")
                    except AttributeError:
                        # Fallback if best_value is not available
                        if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                            print(f"Iteration {iteration_idx + 1} completed")
            else:
                if (optuna.logging.get_verbosity() <= optuna.logging.WARNING):
                    print(f"Iteration {iteration_idx + 1} failed")
        
        if not all_trials:
            raise RuntimeError("All Hyperband iterations failed!")
        
        # Create final study with all trials
        if self.is_multi_objective:
            self.study = optuna.create_study(directions=self.directions)
        else:
            self.study = optuna.create_study(direction=self.directions[0])
        
        # Add all trials to the final study
        for trial in all_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.study.add_trial(trial)
        
        print(f"\nParallel optimization completed!")
        if self.is_multi_objective:
            pareto_front = _get_pareto_front(self.study.trials, self.directions)
            if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                print(f"Final Pareto front size: {len(pareto_front)}")
                print("Pareto front solutions:")
                for i, trial in enumerate(pareto_front[:5]):  # Show first 5
                    print(f"  Solution {i+1}: values={trial.values}, params={trial.params}")
            if len(pareto_front) > 5:
                print(f"  ... and {len(pareto_front) - 5} more solutions")
        else:
            try:
                if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                    print(f"Best value across all iterations: {self.study.best_value}")
                    print(f"Best parameters: {self.study.best_params}")
            except AttributeError:
                    # Fallback for multi-objective studies that somehow got here
                    if (optuna.logging.get_verbosity() <= optuna.logging.INFO):
                        print("Optimization completed (multi-objective study)")
        
        return self.study
    
    @property
    def best_trials(self):
        """Get best trials (Pareto front for multi-objective, single best for single-objective)."""
        if self.study is None:
            return []
        
        if self.is_multi_objective:
            return _get_pareto_front(self.study.trials, self.directions)
        else:
            return [self.study.best_trial] if self.study.best_trial else []
    
    @property
    def best_trial(self):
        """Get best trial (first Pareto solution for multi-objective, best for single-objective)."""
        if self.study is None:
            return None
        
        if self.is_multi_objective:
            pareto_front = self.best_trials
            return pareto_front[0] if pareto_front else None
        else:
            return self.study.best_trial
    
    @property
    def best_value(self):
        """Get best value (first Pareto solution values for multi-objective, best value for single-objective)."""
        if self.study is None:
            return None
        
        if self.is_multi_objective:
            best_trial = self.best_trial
            return best_trial.values if best_trial else None
        else:
            return self.study.best_value
    
    @property
    def best_params(self):
        """Get best parameters (first Pareto solution params for multi-objective, best params for single-objective)."""
        if self.study is None:
            return None
        
        best_trial = self.best_trial
        return best_trial.params if best_trial else None
    
    @property
    def iteration_studies(self):
        """Get list of individual iteration studies (only available when hyperband_iterations > 1)."""
        return getattr(self, '_iteration_studies', [])
    
    @iteration_studies.setter
    def iteration_studies(self, value):
        """Set the iteration studies list."""
        self._iteration_studies = value
        

