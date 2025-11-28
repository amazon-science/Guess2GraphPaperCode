"""
Experiment Baselines PC-Guess-Expert Module

Run causal discovery experiments including PC-Guess-Expert with different accuracy parameters.
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


def format_edge_probability(edge_probability: float) -> str:
    """Format edge_probability value consistently for directory names."""
    return str(edge_probability).replace('.', '_')


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
        len(config['graph_params']['edge_probabilities'])
    )
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)


def _trial_worker(args):
    """Worker function for parallel trial execution."""
    dimensionality, sample_size, edge_probability, algorithms, metrics_to_compute, trial_dir, alpha, pc_guess_expert_configs, sgs_guess_expert_configs, pc_guess_expert_twice_configs, sgs_guess_expert_twice_configs, pc_hc_guess_expert_configs, sgs_hc_guess_expert_configs = args
    try:
        return run_single_trial(dimensionality, sample_size, edge_probability, algorithms, metrics_to_compute, trial_dir, alpha, pc_guess_expert_configs, sgs_guess_expert_configs, pc_guess_expert_twice_configs, sgs_guess_expert_twice_configs, pc_hc_guess_expert_configs, sgs_hc_guess_expert_configs)
    except Exception as e:
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "error.txt"), 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return None


def run_single_trial(
    dimensionality: int,
    sample_size: int,
    edge_probability: float,
    algorithms: List[str],
    metrics_to_compute: List[str],
    trial_dir: str,
    alpha: float = 0.01,
    pc_guess_expert_configs: List[Dict[str, float]] = None,
    sgs_guess_expert_configs: List[Dict[str, float]] = None,
    pc_guess_expert_twice_configs: List[Dict[str, float]] = None,
    sgs_guess_expert_twice_configs: List[Dict[str, float]] = None,
    pc_hc_guess_expert_configs: List[Dict[str, float]] = None,
    sgs_hc_guess_expert_configs: List[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Run a single trial and save results.
    
    Args:
        dimensionality: Number of nodes in the graph
        sample_size: Number of data samples to generate
        edge_probability: Probability of edge existence (0 to 1)
        algorithms: List of algorithm names to run
        metrics_to_compute: List of metrics to calculate
        trial_dir: Directory to save trial results
        alpha: Significance level for independence tests
        pc_guess_expert_configs: List of parameter configurations for pc_guess_expert
    
    Returns:
        Dictionary containing trial metrics
    """
    # Generate random seed
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    
    # Generate data
    true_graph, data = generate_linear_data(
        n_nodes=dimensionality,
        edge_probability=edge_probability,
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
        if algorithm == 'pc_guess_expert':
            # Run multiple configurations of pc_guess_expert
            for i, expert_config in enumerate(pc_guess_expert_configs):
                # Set a different random seed for each configuration
                config_seed = seed + i + 1000  # Offset to avoid overlap
                np.random.seed(config_seed)
                
                algorithm_name = f"pc_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'pc_guess_expert', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
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
                config_seed = seed + i + 2000  # Offset to avoid overlap
                np.random.seed(config_seed)
                
                algorithm_name = f"sgs_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'sgs_guess_expert', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                # Calculate metrics
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
        elif algorithm == 'pc_guess_expert_twice':
            # Run multiple configurations of pc_guess_expert_twice
            for i, expert_config in enumerate(pc_guess_expert_twice_configs or []):
                # Set a different random seed for each configuration
                config_seed = seed + i + 3000  # Offset to avoid overlap
                np.random.seed(config_seed)
                
                algorithm_name = f"pc_guess_expert_twice_{i}"
                predicted_skeleton, runtime = run_method(
                    'pc_guess_expert_twice', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                # Calculate metrics
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
        elif algorithm == 'sgs_guess_expert_twice':
            # Run multiple configurations of sgs_guess_expert_twice
            for i, expert_config in enumerate(sgs_guess_expert_twice_configs or []):
                # Set a different random seed for each configuration
                config_seed = seed + i + 4000  # Offset to avoid overlap
                np.random.seed(config_seed)
                
                algorithm_name = f"sgs_guess_expert_twice_{i}"
                predicted_skeleton, runtime = run_method(
                    'sgs_guess_expert_twice', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
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
                true_dag=true_graph,
                alpha=alpha,
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
        
        elif algorithm == 'pc_guess_expert_twice_r':
            # Run pc_guess_expert_twice with all p_acc parameters = 0.5
            config_seed = seed + 6000
            np.random.seed(config_seed)
            
            predicted_skeleton, runtime = run_method(
                'pc_guess_expert_twice', 
                data, 
                true_dag=true_graph,
                alpha=alpha,
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
                true_dag=true_graph,
                alpha=alpha,
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
        
        elif algorithm == 'pc_hc_guess_expert':
            # Run multiple configurations of pc_hc_guess_expert
            for i, expert_config in enumerate(pc_hc_guess_expert_configs or []):
                config_seed = seed + i + 8000
                np.random.seed(config_seed)
                
                algorithm_name = f"pc_hc_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'pc_hc_guess_expert', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
        elif algorithm == 'sgs_hc_guess_expert':
            # Run multiple configurations of sgs_hc_guess_expert
            for i, expert_config in enumerate(sgs_hc_guess_expert_configs or []):
                config_seed = seed + i + 9000
                np.random.seed(config_seed)
                
                algorithm_name = f"sgs_hc_guess_expert_{i}"
                predicted_skeleton, runtime = run_method(
                    'sgs_hc_guess_expert', 
                    data, 
                    true_dag=true_graph,
                    alpha=alpha,
                    **expert_config
                )
                
                algorithm_results[algorithm_name] = predicted_skeleton
                algorithm_runtimes[algorithm_name] = runtime
                
                metrics = calculate_metrics(true_skeleton, predicted_skeleton, metrics_to_compute)
                algorithm_metrics[algorithm_name] = metrics
        
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
    
    # Get all algorithm names from trials (including pc_guess_expert variants)
    all_algorithms = set()
    for trial in trial_results:
        for key in trial.keys():
            if key not in ["seed", "runtimes"]:
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
    
    combinations = list(product(
        config['graph_params']['dimensionalities'],
        config['graph_params']['sample_sizes'],
        config['graph_params']['edge_probabilities']
    ))
    
    for dim, sample_size, edge_probability in combinations:
        combo_dir = f"dim_{dim}_samples_{sample_size}_edge_prob_{format_edge_probability(edge_probability)}"
        metrics_file = os.path.join(exp_dir, combo_dir, "aggregated_metrics.json")
        
        print(f"\nDim={dim}, Samples={sample_size}, Edge_Prob={edge_probability}:")
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
                
                # Print pc_guess_expert variants (exclude -r variants)
                expert_algorithms = [alg for alg in metrics.keys() if alg.startswith('pc_guess_expert_') and not alg.endswith('_r')]
                for algorithm in sorted(expert_algorithms):
                    if algorithm in metrics:
                        # Get config index
                        config_idx = int(algorithm.split('_')[-1])
                        expert_config = config['pc_guess_expert_configs'][config_idx]
                        print(f"  {algorithm.upper()} (p_edge_true={expert_config['p_acc_edge_true']}, p_edge_false={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']}):")
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
                
                # Print sgs_guess_expert variants (exclude -r variants)
                if 'sgs_guess_expert_configs' in config:
                    sgs_expert_algorithms = [alg for alg in metrics.keys() if alg.startswith('sgs_guess_expert_') and not alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r')]
                    for algorithm in sorted(sgs_expert_algorithms):
                        if algorithm in metrics:
                            # Get config index
                            config_idx = int(algorithm.split('_')[-1])
                            expert_config = config['sgs_guess_expert_configs'][config_idx]
                            print(f"  {algorithm.upper()} (p_edge_true={expert_config['p_acc_edge_true']}, p_edge_false={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']}):")
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
                
                # Print pc_guess_expert_twice variants (exclude -r variants)
                if 'pc_guess_expert_twice_configs' in config:
                    pc_expert_twice_algorithms = [alg for alg in metrics.keys() if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r')]
                    for algorithm in sorted(pc_expert_twice_algorithms):
                        if algorithm in metrics:
                            # Get config index
                            config_idx = int(algorithm.split('_')[-1])
                            expert_config = config['pc_guess_expert_twice_configs'][config_idx]
                            print(f"  {algorithm.upper()} (p_edge_true={expert_config['p_acc_edge_true']}, p_edge_false={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']}):")
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
                
                # Print sgs_guess_expert_twice variants (exclude -r variants)
                if 'sgs_guess_expert_twice_configs' in config:
                    sgs_expert_twice_algorithms = [alg for alg in metrics.keys() if alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r')]
                    for algorithm in sorted(sgs_expert_twice_algorithms):
                        if algorithm in metrics:
                            # Get config index
                            config_idx = int(algorithm.split('_')[-1])
                            expert_config = config['sgs_guess_expert_twice_configs'][config_idx]
                            print(f"  {algorithm.upper()} (p_edge_true={expert_config['p_acc_edge_true']}, p_edge_false={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']}):")
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
                
                # Print pc_hc_guess_expert variants
                if 'pc_hc_guess_expert_configs' in config:
                    pc_hc_algorithms = [alg for alg in metrics.keys() if alg.startswith('pc_hc_guess_expert_')]
                    for algorithm in sorted(pc_hc_algorithms):
                        if algorithm in metrics:
                            config_idx = int(algorithm.split('_')[-1])
                            expert_config = config['pc_hc_guess_expert_configs'][config_idx]
                            print(f"  {algorithm.upper()} (p_edge={expert_config['p_acc_edge_true']}, constraint_frac={expert_config['constraint_fraction']}, random_guidance={expert_config['use_random_guidance']}):")
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
                
                # Print sgs_hc_guess_expert variants
                if 'sgs_hc_guess_expert_configs' in config:
                    sgs_hc_algorithms = [alg for alg in metrics.keys() if alg.startswith('sgs_hc_guess_expert_')]
                    for algorithm in sorted(sgs_hc_algorithms):
                        if algorithm in metrics:
                            config_idx = int(algorithm.split('_')[-1])
                            expert_config = config['sgs_hc_guess_expert_configs'][config_idx]
                            print(f"  {algorithm.upper()} (p_edge={expert_config['p_acc_edge_true']}, constraint_frac={expert_config['constraint_fraction']}, random_guidance={expert_config['use_random_guidance']}):")
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
            config['graph_params']['edge_probabilities']
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
        config['graph_params']['edge_probabilities']
    ))
    
    print(f"Starting experiment {exp_num} with {len(combinations)} parameter combinations")
    
    # Progress bar for parameter combinations
    combo_pbar = tqdm(combinations, desc="Parameter combinations", unit="combo")
    
    for dim, sample_size, edge_probability in combo_pbar:
        combo_pbar.set_description(f"dim={dim}, samples={sample_size}, edge_prob={edge_probability}")
        
        # Create combination directory
        combo_dir = os.path.join(exp_dir, f"dim_{dim}_samples_{sample_size}_edge_prob_{format_edge_probability(edge_probability)}")
        os.makedirs(combo_dir, exist_ok=True)
        
        # Save combination config
        combo_config = {
            "dimensionality": dim,
            "sample_size": sample_size,
            "edge_probability": edge_probability
        }
        with open(os.path.join(combo_dir, "combination_config.json"), 'w') as f:
            json.dump(combo_config, f, indent=2)
        
        # Prepare trial arguments
        trial_args = []
        for trial_idx in range(config['num_trials']):
            trial_dir = os.path.join(combo_dir, f"trial_{trial_idx:03d}")
            trial_args.append((dim, sample_size, edge_probability, config['algorithms'], config['metrics'], trial_dir, config.get('alpha', 0.01), config.get('pc_guess_expert_configs', []), config.get('sgs_guess_expert_configs', []), config.get('pc_guess_expert_twice_configs', []), config.get('sgs_guess_expert_twice_configs', []), config.get('pc_hc_guess_expert_configs', []), config.get('sgs_hc_guess_expert_configs', [])))
        
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
                if key not in ["seed", "runtimes"]:
                    all_algorithms.add(key)
            if "runtimes" in trial:
                all_algorithms.update(trial["runtimes"].keys())
        
        # Aggregate and save results
        aggregated_metrics = aggregate_trial_metrics(trial_results, list(all_algorithms))
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
    # HC constraint fraction parameter
    hc_constraint_fraction = 0.2
    p_acc_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Generate HC configs
    pc_hc_configs = []
    sgs_hc_configs = []
    for p_acc in p_acc_values:
        pc_hc_configs.append({'p_acc_edge_true': p_acc, 'p_acc_edge_false': p_acc, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5, 'constraint_fraction': hc_constraint_fraction, 'use_random_guidance': False})
        sgs_hc_configs.append({'p_acc_edge_true': p_acc, 'p_acc_edge_false': p_acc, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5, 'constraint_fraction': hc_constraint_fraction, 'use_random_guidance': False})
    for p_acc in p_acc_values:
        pc_hc_configs.append({'p_acc_edge_true': p_acc, 'p_acc_edge_false': p_acc, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5, 'constraint_fraction': hc_constraint_fraction, 'use_random_guidance': True})
        sgs_hc_configs.append({'p_acc_edge_true': p_acc, 'p_acc_edge_false': p_acc, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5, 'constraint_fraction': hc_constraint_fraction, 'use_random_guidance': True})
    
    # Generate pc_guess_expert and sgs_guess_expert configs
    pc_guess_configs = []
    sgs_guess_configs = []
    # 5 settings: p_acc_edge_true = 0.5, p_acc_edge_false varies
    for p_false in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pc_guess_configs.append({'p_acc_edge_true': 0.5, 'p_acc_edge_false': p_false, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5})
        sgs_guess_configs.append({'p_acc_edge_true': 0.5, 'p_acc_edge_false': p_false, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5})
    # 5 settings: p_acc_edge_false = 0.5, p_acc_edge_true varies (skip 0.5 to avoid duplicate)
    for p_true in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pc_guess_configs.append({'p_acc_edge_true': p_true, 'p_acc_edge_false': 0.5, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5})
        sgs_guess_configs.append({'p_acc_edge_true': p_true, 'p_acc_edge_false': 0.5, 'p_acc_subset_MD': 0.5, 'p_acc_subset_NMD': 0.5})
    
    config = {
        'num_trials': 30, # usually 30
        'alpha': 0.05, # usually 0.05 or 0.01
        'hc_constraint_fraction': hc_constraint_fraction,
        'graph_params': {
            'dimensionalities': [20], # usually 10 or 20
            'sample_sizes': [100],
            'edge_probabilities':
             [6/19]  # Probability of edge existence, usualy 0.3 - use 2/(d-1)*k to get k edges in expectation.
        },
        # 'algorithms': ['stable_pc', 'pc_guess_expert', 'sgs_guess_expert', 'pc_guess_expert_twice', 'sgs_guess_expert_twice'],
        # 'algorithms': ['pc', 'stable_pc', 'pc_guess_expert', 'sgs_guess_expert',  'pc_guess_expert_r', 'sgs_guess_expert_r',],
        # 'algorithms': ['pc', 'stable_pc', 'pc_guess_expert', 'sgs_guess_expert'],
        # 'algorithms': ['pc_hc_guess_expert', 'sgs_hc_guess_expert'],
        'algorithms': ['pc_guess_expert', 'sgs_guess_expert'],
        # 'algorithms': ['pc','stable_pc'],
        'metrics': ['f1', 'precision', 'recall'],
        'pc_guess_expert_configs': pc_guess_configs,
        'sgs_guess_expert_configs': sgs_guess_configs,
        'pc_hc_guess_expert_configs': pc_hc_configs,
        'sgs_hc_guess_expert_configs': sgs_hc_configs
    }
    
    run_experiment(config)