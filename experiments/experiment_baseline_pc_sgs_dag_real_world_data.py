"""
Experiment Baselines PC-SGS-DAG Real-World Data Module

Run causal discovery experiments on real-world datasets using LLM DAG guesses.
"""

import os
import json
import numpy as np
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'causality_paper_guess_to_graph'))

from methods import run_method
from metrics import calculate_metrics


def find_next_experiment_number(results_dir: str = "results") -> int:
    """Find the next experiment number to use."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    
    existing_experiments = [d for d in os.listdir(results_dir) if d.startswith("experiment_")]
    if not existing_experiments:
        return 1
    
    numbers = []
    for exp_dir in existing_experiments:
        try:
            num = int(exp_dir.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1


def create_experiment_directory(experiment_num: int, results_dir: str = "results") -> str:
    """Create experiment directory and return path."""
    exp_dir = os.path.join(results_dir, f"experiment_{experiment_num:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_config(config: Dict[str, Any], exp_dir: str) -> None:
    """Save experiment configuration to JSON file."""
    config_with_metadata = config.copy()
    config_with_metadata['timestamp'] = datetime.now().isoformat()
    config_with_metadata['total_combinations'] = len(config['real_world_params']['datasets'])
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)


def load_real_world_dataset(dataset_name: str) -> tuple:
    """Load real-world dataset from file."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "real-world-data", f"{dataset_name}-real-world-data.npy")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
    
    adj_matrix, data, variable_names = np.load(dataset_path, allow_pickle=True)
    return adj_matrix, data, variable_names


def load_llm_guesses(guess_experiment_folder: str, model_names: List[str]) -> Dict[str, List[np.ndarray]]:
    """Load LLM DAG guesses from specified experiment folder and models."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    guess_dir = os.path.join(project_root, "guess-llms-results", guess_experiment_folder)
    
    if not os.path.exists(guess_dir):
        raise FileNotFoundError(f"Guess experiment folder {guess_experiment_folder} not found at {guess_dir}")
    
    model_guesses = {}
    
    for model_name in model_names:
        model_path = os.path.join(guess_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_name} not found in {guess_experiment_folder}")
            continue
        
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        valid_dags = []
        for trial in model_data.get('trials', []):
            if trial.get('success') == True and trial.get('error') is None:
                adj_matrix = trial.get('adjacency_matrix')
                if adj_matrix is not None:
                    valid_dags.append(np.array(adj_matrix))
        
        if valid_dags:
            model_guesses[model_name.replace('_results.json', '')] = valid_dags
        else:
            print(f"Warning: No valid DAG guesses found for model {model_name}")
    
    return model_guesses


def _trial_worker(args):
    """Worker function for parallel trial execution."""
    dataset_config, single_guess, model_name, guess_index, methods, metrics_to_compute, trial_dir, alpha, expert_params = args
    try:
        return run_single_guess_trial(dataset_config, single_guess, model_name, guess_index, methods, metrics_to_compute, trial_dir, alpha, expert_params)
    except Exception as e:
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "error.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return None


def run_single_guess_trial(
    dataset_config: Dict[str, Any],
    single_guess: np.ndarray,
    model_name: str,
    guess_index: int,
    methods: List[str],
    metrics_to_compute: List[str],
    trial_dir: str,
    alpha: float = 0.01,
    expert_params: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Run a single trial with one DAG guess on fresh data sample.
    
    Args:
        dataset_config: Dataset configuration with name, ci_test, sample_size
        single_guess: Single DAG guess as numpy array
        model_name: Name of the model that generated this guess
        guess_index: Index of this guess within the model's guesses
        methods: List of method names to run
        metrics_to_compute: List of metrics to calculate
        trial_dir: Directory to save trial results
        alpha: Significance level for independence tests
    
    Returns:
        Dictionary containing trial metrics
    """
    # Generate fresh random seed for this guess
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    
    # Load real-world dataset
    dataset_name = dataset_config['dataset']
    ci_test = dataset_config['ci_test']
    sample_size = dataset_config['sample_size']
    
    adj_matrix, full_data, variable_names = load_real_world_dataset(dataset_name)
    
    # Check sample size
    if full_data.shape[0] < sample_size:
        raise ValueError(f"Dataset {dataset_name} has {full_data.shape[0]} samples, but {sample_size} requested")
    
    # Draw fresh data sample for this guess
    if sample_size < full_data.shape[0]:
        indices = np.random.choice(full_data.shape[0], size=sample_size, replace=False)
        data = full_data[indices]
    else:
        data = full_data
    
    # Convert adjacency matrix to skeleton for metrics
    true_skeleton = (np.abs(adj_matrix) > 0).astype(int)
    true_skeleton = np.maximum(true_skeleton, true_skeleton.T)
    
    # Run algorithms
    algorithm_results = {}
    algorithm_runtimes = {}
    algorithm_metrics = {}
    
    # Run baseline methods (no DAG guess needed)
    baseline_methods = [method for method in methods if method in ['stable_pc', 'pc']]
    for method in baseline_methods:
        predicted_skeleton, runtime = run_method(method, data, alpha=alpha, ci_test=ci_test)
        algorithm_results[method] = predicted_skeleton
        algorithm_runtimes[method] = runtime
        metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
        algorithm_metrics[method] = metrics
    
    # Run expert methods (use true DAG with expert parameters)
    expert_methods = [method for method in methods if method in ['pc_guess_expert', 'sgs_guess_expert']]
    for method in expert_methods:
        if expert_params is None:
            expert_params = {}
        predicted_skeleton, runtime = run_method(
            method, 
            data, 
            true_dag=adj_matrix,
            alpha=alpha,
            ci_test=ci_test,
            **expert_params
        )
        algorithm_results[method] = predicted_skeleton
        algorithm_runtimes[method] = runtime
        metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
        algorithm_metrics[method] = metrics
    
    # Run true DAG methods (sanity check with perfect guess)
    true_dag_methods = [method for method in methods if method in ['pc_guess_dag_true', 'sgs_guess_dag_true']]
    for method in true_dag_methods:
        base_method = method.replace('_true', '')  # pc_guess_dag or sgs_guess_dag
        predicted_skeleton, runtime = run_method(base_method, data, guessed_dag=adj_matrix, alpha=alpha, ci_test=ci_test)
        algorithm_runtimes[method] = runtime
        metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
        algorithm_metrics[method] = metrics
    
    # Run DAG-guided methods with this specific guess
    dag_methods = [method for method in methods if method in ['pc_guess_dag', 'sgs_guess_dag']]
    for method in dag_methods:
        algorithm_name = f"{model_name}_{method}"
        predicted_skeleton, runtime = run_method(
            method, 
            data, 
            guessed_dag=single_guess,
            alpha=alpha,
            ci_test=ci_test
        )
        algorithm_results[algorithm_name] = predicted_skeleton
        algorithm_runtimes[algorithm_name] = runtime
        metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
        algorithm_metrics[algorithm_name] = metrics
    
    # Save trial results
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save seed and guess info
    with open(os.path.join(trial_dir, "seed.txt"), 'w') as f:
        f.write(str(seed))
    
    with open(os.path.join(trial_dir, "guess_info.json"), 'w') as f:
        json.dump({
            "model_name": model_name,
            "guess_index": guess_index
        }, f, indent=2)
    
    # Save data and graphs
    np.save(os.path.join(trial_dir, "data.npy"), data)
    np.save(os.path.join(trial_dir, "true_graph.npy"), adj_matrix)
    np.save(os.path.join(trial_dir, "true_skeleton.npy"), true_skeleton)
    np.save(os.path.join(trial_dir, "guessed_dag.npy"), single_guess)
    
    # Save variable names
    with open(os.path.join(trial_dir, "variable_names.json"), 'w') as f:
        json.dump(list(variable_names), f, indent=2)
    
    for algorithm, result in algorithm_results.items():
        np.save(os.path.join(trial_dir, f"{algorithm}_skeleton.npy"), result)
    
    # Save metrics
    trial_metrics = {
        "seed": seed,
        "model_name": model_name,
        "guess_index": guess_index,
        "dataset": dataset_name,
        "ci_test": ci_test,
        "sample_size": sample_size,
        "variable_names": list(variable_names),
        "methods": methods,
        "runtimes": algorithm_runtimes,
        **algorithm_metrics
    }
    
    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(trial_metrics, f, indent=2)
    
    return trial_metrics


def aggregate_trial_metrics(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across trials."""
    aggregated = {"num_trials": len(trial_results)}
    
    # Get all algorithm names from trials
    all_algorithms = set()
    for trial in trial_results:
        for key in trial.keys():
            if key not in ["seed", "runtimes", "dataset", "ci_test", "sample_size", "variable_names", "methods", "model_name", "guess_index"]:
                all_algorithms.add(key)
        if "runtimes" in trial:
            all_algorithms.update(trial["runtimes"].keys())
    
    for algorithm in all_algorithms:
        algorithm_metrics = {}
        
        # Get all metric names from all trials that contain this algorithm
        all_metric_names = set()
        for trial in trial_results:
            if algorithm in trial and isinstance(trial[algorithm], dict):
                all_metric_names.update(trial[algorithm].keys())
        
        for metric_name in all_metric_names:
            values = []
            for trial in trial_results:
                if (algorithm in trial and 
                    isinstance(trial[algorithm], dict) and 
                    metric_name in trial[algorithm]):
                    values.append(trial[algorithm][metric_name])
            
            if values:
                algorithm_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Add runtime statistics
        runtimes = [trial["runtimes"][algorithm] for trial in trial_results if "runtimes" in trial and algorithm in trial["runtimes"]]
        if runtimes:
            algorithm_metrics["runtime"] = {
                "mean": float(np.mean(runtimes)),
                "std": float(np.std(runtimes)),
                "min": float(np.min(runtimes)),
                "max": float(np.max(runtimes))
            }
        
        aggregated[algorithm] = algorithm_metrics
    
    return aggregated


def print_experiment_summary(exp_dir: str, config: Dict[str, Any]):
    """Print a summary of experiment results grouped by method+model combination."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for dataset_config in config['real_world_params']['datasets']:
        dataset_name = dataset_config['dataset']
        ci_test = dataset_config['ci_test']
        sample_size = dataset_config['sample_size']
        
        combo_dir = f"dataset_{dataset_name}_ci_{ci_test}_samples_{sample_size}"
        metrics_file = os.path.join(exp_dir, combo_dir, "aggregated_metrics.json")
        
        print(f"\nDataset={dataset_name}, CI={ci_test}, Samples={sample_size}:")
        print("-" * 50)
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
                # Group results by method+model combination
                method_model_results = {}
                
                # Process baseline methods
                baseline_algorithms = [alg for alg in metrics.keys() if alg in ['stable_pc', 'pc', 'pc_guess_expert', 'sgs_guess_expert', 'pc_guess_dag_true', 'sgs_guess_dag_true'] and alg != 'num_trials']
                for algorithm in baseline_algorithms:
                    if algorithm in metrics:
                        method_model_results[algorithm] = metrics[algorithm]
                
                # Process model-based methods - group by method+model
                model_algorithms = [alg for alg in metrics.keys() if alg not in baseline_algorithms + ['num_trials']]
                for algorithm in model_algorithms:
                    # Parse algorithm name: model_method_dagindex
                    parts = algorithm.split('_')
                    if len(parts) >= 3:
                        # Extract model and method (everything except last part which is dag index)
                        model_method = '_'.join(parts[:-1])
                        
                        if model_method not in method_model_results:
                            method_model_results[model_method] = {}
                            # Initialize with first algorithm's metrics structure
                            for metric_name in metrics[algorithm]:
                                method_model_results[model_method][metric_name] = {
                                    'values': [],
                                    'mean': 0,
                                    'std': 0,
                                    'min': 0,
                                    'max': 0
                                }
                        
                        # Collect values for averaging
                        for metric_name in metrics[algorithm]:
                            if metric_name in method_model_results[model_method]:
                                if isinstance(metrics[algorithm][metric_name], dict) and 'mean' in metrics[algorithm][metric_name]:
                                    method_model_results[model_method][metric_name]['values'].append(
                                        metrics[algorithm][metric_name]['mean']
                                    )
                
                # Calculate averages for grouped results
                for method_model in method_model_results:
                    if method_model not in baseline_algorithms:
                        for metric_name in method_model_results[method_model]:
                            values = method_model_results[method_model][metric_name].get('values', [])
                            if values:
                                method_model_results[method_model][metric_name] = {
                                    'mean': float(np.mean(values)),
                                    'std': float(np.std(values)),
                                    'min': float(np.min(values)),
                                    'max': float(np.max(values))
                                }
                
                # Print results
                for method_model in sorted(method_model_results.keys()):
                    print(f"  {method_model.upper()}:")
                    for metric in config['metrics']:
                        if metric in method_model_results[method_model]:
                            mean_val = method_model_results[method_model][metric]['mean']
                            std_val = method_model_results[method_model][metric]['std']
                            print(f"    {metric}: {mean_val:.3f} (±{std_val:.3f})")
                    if 'runtime' in method_model_results[method_model]:
                        runtime_mean = method_model_results[method_model]['runtime']['mean']
                        runtime_std = method_model_results[method_model]['runtime']['std']
                        print(f"    runtime: {runtime_mean:.3f}s (±{runtime_std:.3f}s)")
                    print()
        else:
            print("  No results found")
    
    print("\n" + "="*60)


def save_experiment_summary(exp_dir: str, config: Dict[str, Any], start_time: datetime):
    """Save experiment summary with overall statistics."""
    end_time = datetime.now()
    
    successful_trials = 0
    failed_trials = 0
    
    for combo_dir in os.listdir(exp_dir):
        if combo_dir.startswith("dataset_"):
            combo_path = os.path.join(exp_dir, combo_dir)
            for trial_dir in os.listdir(combo_path):
                if trial_dir.startswith("trial_"):
                    trial_path = os.path.join(combo_path, trial_dir)
                    if os.path.exists(os.path.join(trial_path, "metrics.json")):
                        successful_trials += 1
                    elif os.path.exists(os.path.join(trial_path, "error.txt")):
                        failed_trials += 1
    
    summary = {
        'experiment_number': int(exp_dir.split('_')[-1]),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'total_combinations': len(config['real_world_params']['datasets']),
        'trials_per_combination': successful_trials + failed_trials,
        'successful_trials': successful_trials,
        'failed_trials': failed_trials,
        'success_rate': successful_trials / (successful_trials + failed_trials) if (successful_trials + failed_trials) > 0 else 0
    }
    
    with open(os.path.join(exp_dir, "experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def run_experiment(config: Dict[str, Any], results_dir: str = None) -> str:
    """
    Run complete experiment with given configuration.
    
    Args:
        config: Experiment configuration dictionary
        results_dir: Directory to save results
    
    Returns:
        Path to experiment directory
    """
    start_time = datetime.now()
    
    # Default to project root results directory
    if results_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "results")
    
    # Create experiment directory
    exp_num = find_next_experiment_number(results_dir)
    exp_dir = create_experiment_directory(exp_num, results_dir)
    
    # Save configuration
    save_experiment_config(config, exp_dir)
    
    # Load LLM guesses
    model_guesses = load_llm_guesses(config['guess_experiment_folder'], config['model_names'])
    
    if not model_guesses:
        raise ValueError("No valid LLM guesses found for any specified models")
    
    datasets = config['real_world_params']['datasets']
    
    print(f"Starting experiment {exp_num} with {len(datasets)} datasets")
    print(f"Using LLM guesses from {config['guess_experiment_folder']}")
    print(f"Models: {list(model_guesses.keys())}")
    
    # Progress bar for datasets
    dataset_pbar = tqdm(datasets, desc="Datasets", unit="dataset")
    
    for dataset_config in dataset_pbar:
        dataset_name = dataset_config['dataset']
        ci_test = dataset_config['ci_test']
        sample_size = dataset_config['sample_size']
        
        dataset_pbar.set_description(f"dataset={dataset_name}, ci={ci_test}, samples={sample_size}")
        
        # Create combination directory
        combo_dir = os.path.join(exp_dir, f"dataset_{dataset_name}_ci_{ci_test}_samples_{sample_size}")
        os.makedirs(combo_dir, exist_ok=True)
        
        # Calculate total number of trials (one per guess)
        total_guesses = sum(len(guesses) for guesses in model_guesses.values())
        
        # Save combination config
        combo_config = {
            "dataset": dataset_name,
            "ci_test": ci_test,
            "sample_size": sample_size,
            "methods": config['methods'],
            "model_guesses_info": {model: len(guesses) for model, guesses in model_guesses.items()},
            "total_trials": total_guesses
        }
        with open(os.path.join(combo_dir, "combination_config.json"), 'w') as f:
            json.dump(combo_config, f, indent=2)
        
        # Prepare trial arguments - one trial per guess
        trial_args = []
        trial_idx = 0
        
        for model_name, dag_guesses in model_guesses.items():
            for guess_idx, single_guess in enumerate(dag_guesses):
                trial_dir = os.path.join(combo_dir, f"trial_{trial_idx:03d}")
                trial_args.append((
                    dataset_config, 
                    single_guess, 
                    model_name, 
                    guess_idx,
                    config['methods'], 
                    config['metrics'], 
                    trial_dir, 
                    config.get('alpha', 0.01),
                    config.get('expert_params', {})
                ))
                trial_idx += 1
        
        # Run trials in parallel
        trial_results = []
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = [executor.submit(_trial_worker, args) for args in trial_args]
            
            trial_pbar = tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Trials", 
                             unit="trial", 
                             leave=False)
            
            for future in trial_pbar:
                result = future.result()
                if result is not None:
                    trial_results.append(result)
            
            trial_pbar.close()
        
        # Aggregate and save results
        aggregated_metrics = aggregate_trial_metrics(trial_results)
        with open(os.path.join(combo_dir, "aggregated_metrics.json"), 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
    
    dataset_pbar.close()
    
    # Generate and save summary
    summary = save_experiment_summary(exp_dir, config, start_time)
    print_experiment_summary(exp_dir, config)
    print(f"\nTotal runtime: {summary['duration_seconds']:.2f} seconds")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    
    print(f"\nExperiment {exp_num} completed. Results saved to: {exp_dir}")
    return exp_dir


# Example usage
if __name__ == "__main__":
    config = {
        'alpha': 0.05,
        'real_world_params': {
            'datasets': [
                {
                    'dataset': 'sachs',
                    'ci_test': 'chisq',
                    'sample_size': 100
                }
            ]
        },
        'expert_params': {
            'p_acc_edge_true': 0.5,
            'p_acc_edge_false': 0.5,
            'p_acc_subset_MD': 0.5,
            'p_acc_subset_NMD': 0.5
        },
        'guess_experiment_folder': 'guess_experiment_34',
        'model_names': ['claude_opus_4_1_results.json'],
        # 'model_names': ['llama_4_scout_results.json'],
        # 'methods': ['pc', 'stable_pc', 'pc_guess_expert', 'sgs_guess_expert', 'pc_guess_dag', 'sgs_guess_dag', 'pc_guess_dag_true', 'sgs_guess_dag_true'],
        'methods': ['stable_pc', 'pc_guess_expert', 'sgs_guess_expert', 'pc_guess_dag', 'sgs_guess_dag'],
        'metrics': ['f1', 'precision', 'recall']
    }
    
    run_experiment(config)