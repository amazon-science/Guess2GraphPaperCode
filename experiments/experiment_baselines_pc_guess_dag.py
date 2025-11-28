"""
Experiment Baselines PC-Guess-DAG Module

Run causal discovery experiments including PC-Guess-DAG with true DAG as oracle.
"""

import os
import json
import numpy as np
import traceback
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'causality_paper_guess_to_graph'))

from data_generation import generate_linear_data
from methods import run_method
from metrics import calculate_metrics


def format_sparsity(sparsity: float) -> str:
    """Format sparsity value consistently for directory names."""
    return str(sparsity).replace('.', '_')


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
    config_with_metadata['total_combinations'] = (
        len(config['graph_params']['dimensionalities']) *
        len(config['graph_params']['sample_sizes']) *
        len(config['graph_params']['sparsities'])
    )
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)


def _trial_worker(args):
    """Worker function for parallel trial execution."""
    dimensionality, sample_size, sparsity, algorithms, metrics_to_compute, trial_dir, alpha = args
    try:
        return run_single_trial(dimensionality, sample_size, sparsity, algorithms, metrics_to_compute, trial_dir, alpha)
    except Exception as e:
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "error.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return None


def run_single_trial(
    dimensionality: int,
    sample_size: int,
    sparsity: float,
    algorithms: List[str],
    metrics_to_compute: List[str],
    trial_dir: str,
    alpha: float = 0.01
) -> Dict[str, Any]:
    """
    Run a single trial and save results.
    
    Args:
        dimensionality: Number of nodes in the graph
        sample_size: Number of data samples to generate
        sparsity: Expected edges per node (will be converted to total expected edges)
        algorithms: List of algorithm names to run
        metrics_to_compute: List of metrics to calculate
        trial_dir: Directory to save trial results
    
    Returns:
        Dictionary containing trial metrics
    """
    # Generate random seed
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    
    # Generate data
    # Note: sparsity parameter in config represents expected edges per node
    # generate_linear_data expects total expected edges, so multiply by dimensionality
    expected_edges_per_node = sparsity  # This is the config sparsity value
    total_expected_edges = int(expected_edges_per_node * dimensionality)
    
    true_graph, data = generate_linear_data(
        n_nodes=dimensionality,
        sparsity=total_expected_edges,  # generate_linear_data expects total expected edges
        sample_size=sample_size
    )
    
    # Convert true graph to skeleton for metrics
    true_skeleton = (np.abs(true_graph) > 0).astype(int)
    true_skeleton = np.maximum(true_skeleton, true_skeleton.T)
    
    # Run algorithms
    algorithm_results = {}
    algorithm_runtimes = {}
    algorithm_metrics = {}
    
    for algorithm in algorithms:
        if algorithm == 'pc_guess_dag':
            # Use true DAG as the guessed DAG for pc_guess_dag
            predicted_skeleton, runtime = run_method(algorithm, data, guessed_dag=true_graph, alpha=alpha)
        elif algorithm == 'pc_guess_dag_wrong':
            # Create wrong DAG: flip all edges and randomize ordering
            wrong_dag = 1 - true_graph  # Flip all edges (0->1, 1->0)
            np.fill_diagonal(wrong_dag, 0)  # Keep diagonal as 0
            
            # Apply random permutation to create wrong topological ordering
            n_vars = true_graph.shape[0]
            perm = np.random.permutation(n_vars)
            wrong_dag_permuted = wrong_dag[np.ix_(perm, perm)]
            
            predicted_skeleton, runtime = run_method('pc_guess_dag', data, guessed_dag=wrong_dag_permuted, alpha=alpha)
        else:
            predicted_skeleton, runtime = run_method(algorithm, data, alpha=alpha)
        
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
    np.save(os.path.join(trial_dir, "true_graph.npy"), true_graph)
    np.save(os.path.join(trial_dir, "true_skeleton.npy"), true_skeleton)
    
    for algorithm, result in algorithm_results.items():
        np.save(os.path.join(trial_dir, f"{algorithm}_skeleton.npy"), result)
    
    # Save metrics
    trial_metrics = {
        "seed": seed,
        "runtimes": algorithm_runtimes,
        **algorithm_metrics
    }
    
    with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
        json.dump(trial_metrics, f, indent=2)
    
    return trial_metrics


def aggregate_trial_metrics(trial_results: List[Dict[str, Any]], algorithms: List[str]) -> Dict[str, Any]:
    """Aggregate metrics across trials."""
    aggregated = {"num_trials": len(trial_results)}
    
    for algorithm in algorithms:
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
    
    combinations = list(product(
        config['graph_params']['dimensionalities'],
        config['graph_params']['sample_sizes'],
        config['graph_params']['sparsities']
    ))
    
    for dim, sample_size, sparsity in combinations:
        combo_dir = f"dim_{dim}_samples_{sample_size}_sparsity_{format_sparsity(sparsity)}"
        metrics_file = os.path.join(exp_dir, combo_dir, "aggregated_metrics.json")
        
        print(f"\nDim={dim}, Samples={sample_size}, Sparsity={sparsity}:")
        print("-" * 50)
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
                for algorithm in config['algorithms']:
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
        else:
            print("  No results found")
    
    print("\n" + "="*60)


def save_experiment_summary(exp_dir: str, config: Dict[str, Any], start_time: datetime):
    """Save experiment summary with overall statistics."""
    end_time = datetime.now()
    
    successful_trials = 0
    failed_trials = 0
    
    for combo_dir in os.listdir(exp_dir):
        if combo_dir.startswith("dim_"):
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
        'total_combinations': len(list(product(
            config['graph_params']['dimensionalities'],
            config['graph_params']['sample_sizes'],
            config['graph_params']['sparsities']
        ))),
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
    
    # Generate parameter combinations
    combinations = list(product(
        config['graph_params']['dimensionalities'],
        config['graph_params']['sample_sizes'],
        config['graph_params']['sparsities']
    ))
    
    print(f"Starting experiment {exp_num} with {len(combinations)} parameter combinations")
    
    # Progress bar for parameter combinations
    combo_pbar = tqdm(combinations, desc="Parameter combinations", unit="combo")
    
    for dim, sample_size, sparsity in combo_pbar:
        combo_pbar.set_description(f"dim={dim}, samples={sample_size}, sparsity={sparsity}")
        
        # Create combination directory
        combo_dir = os.path.join(exp_dir, f"dim_{dim}_samples_{sample_size}_sparsity_{format_sparsity(sparsity)}")
        os.makedirs(combo_dir, exist_ok=True)
        
        # Save combination config
        combo_config = {
            "dimensionality": dim,
            "sample_size": sample_size,
            "sparsity": sparsity
        }
        with open(os.path.join(combo_dir, "combination_config.json"), 'w') as f:
            json.dump(combo_config, f, indent=2)
        
        # Prepare trial arguments
        trial_args = []
        for trial_idx in range(config['num_trials']):
            trial_dir = os.path.join(combo_dir, f"trial_{trial_idx:03d}")
            trial_args.append((dim, sample_size, sparsity, config['algorithms'], config['metrics'], trial_dir, config.get('alpha', 0.01)))
        
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
        aggregated_metrics = aggregate_trial_metrics(trial_results, config['algorithms'])
        with open(os.path.join(combo_dir, "aggregated_metrics.json"), 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
    
    combo_pbar.close()
    
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
        'num_trials': 20,
        'alpha': 0.1,
        'graph_params': {
            'dimensionalities': [20],
            'sample_sizes': [100,200,300,400],
            'sparsities': [1]  # Expected edges per node (not total edges)
        },
        'algorithms': ['pc', 'stable_pc', 'pc_guess_dag', 'pc_guess_dag_wrong'],
        'metrics': ['f1', 'precision', 'recall']
    }
    
    run_experiment(config)