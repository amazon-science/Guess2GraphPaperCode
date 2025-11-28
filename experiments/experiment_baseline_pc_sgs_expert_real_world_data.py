"""
Experiment Baselines PC-SGS-Expert Real-World Data Module

Run causal discovery experiments on real-world datasets including PC-Guess-Expert and SGS-Guess-Expert.
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


def _trial_worker(args):
    """Worker function for parallel trial execution."""
    dataset_config, algorithms, metrics_to_compute, trial_dir, alpha, pc_guess_expert_configs, sgs_guess_expert_configs = args
    try:
        return run_single_trial(dataset_config, algorithms, metrics_to_compute, trial_dir, alpha, pc_guess_expert_configs, sgs_guess_expert_configs)
    except Exception as e:
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "error.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return None


def run_single_trial(
    dataset_config: Dict[str, Any],
    algorithms: List[str],
    metrics_to_compute: List[str],
    trial_dir: str,
    alpha: float = 0.01,
    pc_guess_expert_configs: List[Dict[str, float]] = None,
    sgs_guess_expert_configs: List[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Run a single trial on real-world dataset and save results.
    
    Args:
        dataset_config: Dataset configuration with name, ci_test, sample_size
        algorithms: List of algorithm names to run
        metrics_to_compute: List of metrics to calculate
        trial_dir: Directory to save trial results
        alpha: Significance level for independence tests
        pc_guess_expert_configs: List of parameter configurations for pc_guess_expert
        sgs_guess_expert_configs: List of parameter configurations for sgs_guess_expert
    
    Returns:
        Dictionary containing trial metrics
    """
    # Generate random seed
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
    
    # Subsample data if needed
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
    
    for algorithm in algorithms:
        if algorithm == 'pc_guess_expert':
            # Run multiple configurations of pc_guess_expert
            for i, expert_config in enumerate(pc_guess_expert_configs):
                # Set a different random seed for each configuration
                config_seed = seed + i + 1000
                np.random.seed(config_seed)
                
                algorithm_name = f"pc_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'pc_guess_expert', 
                    data, 
                    true_dag=adj_matrix,
                    alpha=alpha,
                    ci_test=ci_test,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                # Calculate metrics
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
        elif algorithm == 'sgs_guess_expert':
            # Run multiple configurations of sgs_guess_expert
            for i, expert_config in enumerate(sgs_guess_expert_configs):
                # Set a different random seed for each configuration
                config_seed = seed + i + 2000
                np.random.seed(config_seed)
                
                algorithm_name = f"sgs_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'sgs_guess_expert', 
                    data, 
                    true_dag=adj_matrix,
                    alpha=alpha,
                    ci_test=ci_test,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                # Calculate metrics
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
        elif algorithm == 'sgs_guess_expert_r':
            # Run sgs_guess_expert with all p_acc parameters = 0.5
            config_seed = seed + 5000
            np.random.seed(config_seed)
            
            predicted_skeleton, runtime = run_method(
                'sgs_guess_expert', 
                data, 
                true_dag=adj_matrix,
                alpha=alpha,
                ci_test=ci_test,
                p_acc_edge_true=0.5,
                p_acc_edge_false=0.5,
                p_acc_subset_MD=0.5,
                p_acc_subset_NMD=0.5
            )
            
            algorithm_results[algorithm] = predicted_skeleton
            algorithm_runtimes[algorithm] = runtime
            
            # Calculate metrics
            metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
            algorithm_metrics[algorithm] = metrics
        
        elif algorithm == 'pc_guess_expert_r':
            # Run pc_guess_expert with all p_acc parameters = 0.5
            config_seed = seed + 7000
            np.random.seed(config_seed)
            
            predicted_skeleton, runtime = run_method(
                'pc_guess_expert', 
                data, 
                true_dag=adj_matrix,
                alpha=alpha,
                ci_test=ci_test,
                p_acc_edge_true=0.5,
                p_acc_edge_false=0.5,
                p_acc_subset_MD=0.5,
                p_acc_subset_NMD=0.5
            )
            
            algorithm_results[algorithm] = predicted_skeleton
            algorithm_runtimes[algorithm] = runtime
            
            # Calculate metrics
            metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
            algorithm_metrics[algorithm] = metrics
        else:
            predicted_skeleton, runtime = run_method(algorithm, data, alpha=alpha, ci_test=ci_test)
            
            algorithm_results[algorithm] = predicted_skeleton
            algorithm_runtimes[algorithm] = runtime
            
            # Calculate metrics
            metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
            algorithm_metrics[algorithm] = metrics
    
    # Save trial results
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save seed
    with open(os.path.join(trial_dir, "seed.txt"), 'w') as f:
        f.write(str(seed))
    
    # Save data and graphs
    np.save(os.path.join(trial_dir, "data.npy"), data)
    np.save(os.path.join(trial_dir, "true_graph.npy"), adj_matrix)
    np.save(os.path.join(trial_dir, "true_skeleton.npy"), true_skeleton)
    
    # Save variable names
    with open(os.path.join(trial_dir, "variable_names.json"), 'w') as f:
        json.dump(list(variable_names), f, indent=2)
    
    for algorithm, result in algorithm_results.items():
        np.save(os.path.join(trial_dir, f"{algorithm}_skeleton.npy"), result)
    
    # Save metrics
    trial_metrics = {
        "seed": seed,
        "dataset": dataset_name,
        "ci_test": ci_test,
        "sample_size": sample_size,
        "variable_names": list(variable_names),
        "runtimes": algorithm_runtimes,
        **algorithm_metrics
    }
    
    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(trial_metrics, f, indent=2)
    
    return trial_metrics


def aggregate_trial_metrics(trial_results: List[Dict[str, Any]], algorithms: List[str]) -> Dict[str, Any]:
    """Aggregate metrics across trials."""
    aggregated = {"num_trials": len(trial_results)}
    
    # Get all algorithm names from trials
    all_algorithms = set()
    for trial in trial_results:
        for key in trial.keys():
            if key not in ["seed", "runtimes", "dataset", "ci_test", "sample_size", "variable_names"]:
                all_algorithms.add(key)
        if "runtimes" in trial:
            all_algorithms.update(trial["runtimes"].keys())
    
    for algorithm in all_algorithms:
        algorithm_metrics = {}
        
        # Get all metric names from all trials that contain this algorithm
        all_metric_names = set()
        for trial in trial_results:
            if algorithm in trial:
                all_metric_names.update(trial[algorithm].keys())
        
        for metric_name in all_metric_names:
            values = [trial[algorithm][metric_name] for trial in trial_results 
                     if algorithm in trial and metric_name in trial[algorithm]]
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
    """Print a summary of experiment results."""
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
                
                # Print baseline algorithms first
                baseline_algorithms = [alg for alg in config['algorithms'] if alg not in ['pc_guess_expert', 'sgs_guess_expert']]
                for algorithm in baseline_algorithms:
                    if algorithm in metrics:
                        print(f"  {algorithm.upper()}:")
                        for metric in config['metrics']:
                            if metric in metrics[algorithm]:
                                mean_val = metrics[algorithm][metric]['mean']
                                std_val = metrics[algorithm][metric]['std']
                                print(f"    {metric}: {mean_val:.3f} (±{std_val:.3f})")
                        if 'runtime' in metrics[algorithm]:
                            runtime_mean = metrics[algorithm]['runtime']['mean']
                            runtime_std = metrics[algorithm]['runtime']['std']
                            print(f"    runtime: {runtime_mean:.3f}s (±{runtime_std:.3f}s)")
                        print()
                
                # Print expert algorithms (exclude -r variants)
                expert_algorithms = [alg for alg in metrics.keys() if alg.startswith(('pc_guess_expert_', 'sgs_guess_expert_')) and not alg.endswith('_r')]
                for algorithm in sorted(expert_algorithms):
                    if algorithm in metrics:
                        print(f"  {algorithm.upper()}:")
                        for metric in config['metrics']:
                            if metric in metrics[algorithm]:
                                mean_val = metrics[algorithm][metric]['mean']
                                std_val = metrics[algorithm][metric]['std']
                                print(f"    {metric}: {mean_val:.3f} (±{std_val:.3f})")
                        if 'runtime' in metrics[algorithm]:
                            runtime_mean = metrics[algorithm]['runtime']['mean']
                            runtime_std = metrics[algorithm]['runtime']['std']
                            print(f"    runtime: {runtime_mean:.3f}s (±{runtime_std:.3f}s)")
                        print()
                
                # Print -r algorithms with fixed parameters
                r_algorithms = [alg for alg in metrics.keys() if alg.endswith('_r')]
                for algorithm in sorted(r_algorithms):
                    if algorithm in metrics:
                        print(f"  {algorithm.upper()} (all p_acc=0.5):")
                        for metric in config['metrics']:
                            if metric in metrics[algorithm]:
                                mean_val = metrics[algorithm][metric]['mean']
                                std_val = metrics[algorithm][metric]['std']
                                print(f"    {metric}: {mean_val:.3f} (±{std_val:.3f})")
                        if 'runtime' in metrics[algorithm]:
                            runtime_mean = metrics[algorithm]['runtime']['mean']
                            runtime_std = metrics[algorithm]['runtime']['std']
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
        'trials_per_combination': config['num_trials'],
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
    
    datasets = config['real_world_params']['datasets']
    
    print(f"Starting experiment {exp_num} with {len(datasets)} datasets")
    
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
        
        # Save combination config
        combo_config = {
            "dataset": dataset_name,
            "ci_test": ci_test,
            "sample_size": sample_size
        }
        with open(os.path.join(combo_dir, "combination_config.json"), 'w') as f:
            json.dump(combo_config, f, indent=2)
        
        # Prepare trial arguments
        trial_args = []
        for trial_idx in range(config['num_trials']):
            trial_dir = os.path.join(combo_dir, f"trial_{trial_idx:03d}")
            trial_args.append((dataset_config, config['algorithms'], config['metrics'], trial_dir, config.get('alpha', 0.01), config.get('pc_guess_expert_configs', []), config.get('sgs_guess_expert_configs', [])))
        
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
        
        # Get all algorithm names for aggregation
        all_algorithms = set()
        for trial in trial_results:
            for key in trial.keys():
                if key not in ["seed", "runtimes", "dataset", "ci_test", "sample_size", "variable_names"]:
                    all_algorithms.add(key)
            if "runtimes" in trial:
                all_algorithms.update(trial["runtimes"].keys())
        
        # Aggregate and save results
        aggregated_metrics = aggregate_trial_metrics(trial_results, list(all_algorithms))
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
        'num_trials': 30,
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
        'algorithms': ['pc', 'stable_pc',  'pc_guess_expert_r', 'sgs_guess_expert_r', 'pc_guess_expert', 'sgs_guess_expert'],
        'metrics': ['f1', 'precision', 'recall'],
        'pc_guess_expert_configs': [
            {
                'p_acc_edge_true': 0.5,
                'p_acc_edge_false': 0.5,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.6,
                'p_acc_edge_false': 0.6,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.7,
                'p_acc_edge_false': 0.7,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.8,
                'p_acc_edge_false': 0.8,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.9,
                'p_acc_edge_false': 0.9,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 1.0,
                'p_acc_edge_false': 1.0,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            }
        ],
        'sgs_guess_expert_configs': [
            #  {
            #     'p_acc_edge_true': 0.5,
            #     'p_acc_edge_false': 1,
            #     'p_acc_subset_MD': 0.5,
            #     'p_acc_subset_NMD': 0.5
            # },
            # {
            #     'p_acc_edge_true': 1.0,
            #     'p_acc_edge_false': 0.5,
            #     'p_acc_subset_MD': 0.5,
            #     'p_acc_subset_NMD': 0.5
            # },
            {
                'p_acc_edge_true': 0.5,
                'p_acc_edge_false': 0.5,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.6,
                'p_acc_edge_false': 0.6,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.7,
                'p_acc_edge_false': 0.7,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.8,
                'p_acc_edge_false': 0.8,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 0.9,
                'p_acc_edge_false': 0.9,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            },
            {
                'p_acc_edge_true': 1.0,
                'p_acc_edge_false': 1.0,
                'p_acc_subset_MD': 0.5,
                'p_acc_subset_NMD': 0.5
            }
        ],
    }
    
    run_experiment(config)