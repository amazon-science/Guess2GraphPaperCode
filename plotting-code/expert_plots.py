"""
Expert Plots Module

Generate violin plots for PC-Guess-Expert experiments with different accuracy parameters.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any


def format_edge_probability(edge_probability: float) -> str:
    """Format edge_probability value consistently for directory names."""
    return str(edge_probability).replace('.', '_')


def detect_experiment_type(experiment_dir: str) -> str:
    """Detect if experiment is synthetic or real-world based on directory structure."""
    for item in os.listdir(experiment_dir):
        if os.path.isdir(os.path.join(experiment_dir, item)):
            if item.startswith("dim_") and "edge_prob" in item:
                return "synthetic"
            elif item.startswith("dataset_"):
                return "real_world"
    return "unknown"


def load_experiment_data(experiment_dir: str) -> pd.DataFrame:
    """Load experiment data into a pandas DataFrame."""
    data_rows = []
    
    # Load config to get parameter combinations
    config_path = os.path.join(experiment_dir, "config.json")
    print(f"Looking for config at: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Detect experiment type
    exp_type = detect_experiment_type(experiment_dir)
    print(f"Detected experiment type: {exp_type}")
    
    if exp_type == "synthetic":
        # Handle synthetic data experiments
        for combo_dir in os.listdir(experiment_dir):
            if not combo_dir.startswith("dim_") or "edge_prob" not in combo_dir:
                continue
                
            combo_path = os.path.join(experiment_dir, combo_dir)
            if not os.path.isdir(combo_path):
                continue
            
            # Parse combination parameters from directory name
            parts = combo_dir.split("_")
            dim = int(parts[1])
            samples = int(parts[3])
            edge_probability = float(parts[6].replace('_', '.'))
            
            # Load trial data
            for trial_dir in os.listdir(combo_path):
                if not trial_dir.startswith("trial_"):
                    continue
                    
                trial_path = os.path.join(combo_path, trial_dir)
                metrics_file = os.path.join(trial_path, "metrics.json")
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        trial_data = json.load(f)
                    
                    # Process each algorithm
                    for algorithm in trial_data:
                        if algorithm in ['seed', 'runtimes']:
                            continue
                        
                        # Get algorithm metrics
                        algorithm_metrics = trial_data[algorithm]
                        
                        # Create base row
                        row = {
                            'dimensionality': dim,
                            'sample_size': samples,
                            'edge_probability': edge_probability,
                            'algorithm': algorithm,
                            'trial': trial_dir
                        }
                        
                        # Add all metrics
                        for metric_name, metric_value in algorithm_metrics.items():
                            row[metric_name] = metric_value
                        
                        # Add runtime if available
                        if 'runtimes' in trial_data and algorithm in trial_data['runtimes']:
                            row['runtime'] = trial_data['runtimes'][algorithm]
                        
                        data_rows.append(row)
    
    elif exp_type == "real_world":
        # Handle real-world data experiments
        for combo_dir in os.listdir(experiment_dir):
            if not combo_dir.startswith("dataset_"):
                continue
                
            combo_path = os.path.join(experiment_dir, combo_dir)
            if not os.path.isdir(combo_path):
                continue
            
            # Parse combination parameters from directory name
            # Format: dataset_{name}_ci_{test}_samples_{size}
            parts = combo_dir.split("_")
            dataset_name = parts[1]
            ci_test = parts[3]
            sample_size = int(parts[5])
            
            # Load trial data
            for trial_dir in os.listdir(combo_path):
                if not trial_dir.startswith("trial_"):
                    continue
                    
                trial_path = os.path.join(combo_path, trial_dir)
                metrics_file = os.path.join(trial_path, "metrics.json")
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        trial_data = json.load(f)
                    
                    # Process each algorithm
                    for algorithm in trial_data:
                        if algorithm in ['seed', 'runtimes', 'dataset', 'ci_test', 'sample_size', 'variable_names']:
                            continue
                        
                        # Get algorithm metrics
                        algorithm_metrics = trial_data[algorithm]
                        
                        # Create base row
                        row = {
                            'dataset': dataset_name,
                            'ci_test': ci_test,
                            'sample_size': sample_size,
                            'algorithm': algorithm,
                            'trial': trial_dir
                        }
                        
                        # Add all metrics
                        for metric_name, metric_value in algorithm_metrics.items():
                            row[metric_name] = metric_value
                        
                        # Add runtime if available
                        if 'runtimes' in trial_data and algorithm in trial_data['runtimes']:
                            row['runtime'] = trial_data['runtimes'][algorithm]
                        
                        data_rows.append(row)
    
    return pd.DataFrame(data_rows)


def get_algorithm_display_info(algorithm: str, config: Dict[str, Any]) -> Dict[str, str]:
    """Get display name and color info for algorithm."""
    if algorithm.startswith('pc_guess_expert_twice_'):
        # Extract config index
        config_idx = int(algorithm.split('_')[-1])
        expert_config = config['pc_guess_expert_twice_configs'][config_idx]
        
        # Create short display name
        display_name = f"Expert_Twice_{config_idx}"
        full_name = f"PC-Guess-Expert-Twice (p_edge_T={expert_config['p_acc_edge_true']}, p_edge_F={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']})"
        
        return {
            'display_name': display_name,
            'full_name': full_name,
            'type': 'expert_twice'
        }
    elif algorithm.startswith('pc_guess_expert_'):
        # Extract config index
        config_idx = int(algorithm.split('_')[-1])
        expert_config = config['pc_guess_expert_configs'][config_idx]
        
        # Create short display name
        display_name = f"Expert_{config_idx}"
        full_name = f"PC-Guess-Expert (p_edge_T={expert_config['p_acc_edge_true']}, p_edge_F={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']})"
        
        return {
            'display_name': display_name,
            'full_name': full_name,
            'type': 'expert'
        }
    elif algorithm.startswith('sgs_guess_expert_twice_'):
        # Extract config index
        config_idx = int(algorithm.split('_')[-1])
        expert_config = config['sgs_guess_expert_twice_configs'][config_idx]
        
        # Create short display name
        display_name = f"SGS_Expert_Twice_{config_idx}"
        full_name = f"SGS-Guess-Expert-Twice (p_edge_T={expert_config['p_acc_edge_true']}, p_edge_F={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']})"
        
        return {
            'display_name': display_name,
            'full_name': full_name,
            'type': 'sgs_expert_twice'
        }
    elif algorithm.startswith('sgs_guess_expert_'):
        # Extract config index
        config_idx = int(algorithm.split('_')[-1])
        expert_config = config['sgs_guess_expert_configs'][config_idx]
        
        # Create short display name
        display_name = f"SGS_Expert_{config_idx}"
        full_name = f"SGS-Guess-Expert (p_edge_T={expert_config['p_acc_edge_true']}, p_edge_F={expert_config['p_acc_edge_false']}, p_MD={expert_config['p_acc_subset_MD']}, p_NMD={expert_config['p_acc_subset_NMD']})"
        
        return {
            'display_name': display_name,
            'full_name': full_name,
            'type': 'sgs_expert'
        }
    else:
        return {
            'display_name': algorithm.upper(),
            'full_name': algorithm.upper(),
            'type': 'baseline'
        }


def get_reference_colors() -> Dict[str, str]:
    """Get reference colors that match plot_parameter_sweep_varying."""
    set1_colors = sns.color_palette("Set1", 8)
    return {
        'pc_guess_expert': set1_colors[0],  # First color (blue-ish)
        'sgs_guess_expert': set1_colors[1], # Second color (orange-ish)
        'pc_guess_expert_twice': set1_colors[2],
        'sgs_guess_expert_twice': set1_colors[3],
        'pc': 'gray',
        'stable_pc': 'black'
    }


def get_color_palette(algorithms: List[str], config: Dict[str, Any]) -> Dict[str, str]:
    """Get consistent color palette for algorithms."""
    # Separate different algorithm types
    baseline_algs = [alg for alg in algorithms if not (alg.startswith('pc_guess_expert_') or alg.startswith('sgs_guess_expert_'))]
    pc_expert_algs = [alg for alg in algorithms if alg.startswith('pc_guess_expert_') and not alg.startswith('pc_guess_expert_twice_')]
    pc_expert_twice_algs = [alg for alg in algorithms if alg.startswith('pc_guess_expert_twice_')]
    sgs_expert_algs = [alg for alg in algorithms if alg.startswith('sgs_guess_expert_') and not alg.startswith('sgs_guess_expert_twice_')]
    sgs_expert_twice_algs = [alg for alg in algorithms if alg.startswith('sgs_guess_expert_twice_')]
    
    colors = {}
    
    # Use distinct colors for baselines
    if baseline_algs:
        baseline_colors = sns.color_palette("Set1", len(baseline_algs))
        for i, alg in enumerate(baseline_algs):
            colors[alg] = baseline_colors[i]
    
    # Use gradient colors for different expert variants
    if pc_expert_algs:
        pc_expert_colors = sns.color_palette("Blues", len(pc_expert_algs))
        for i, alg in enumerate(pc_expert_algs):
            colors[alg] = pc_expert_colors[i]
    
    if pc_expert_twice_algs:
        pc_expert_twice_colors = sns.color_palette("Purples", len(pc_expert_twice_algs))
        for i, alg in enumerate(pc_expert_twice_algs):
            colors[alg] = pc_expert_twice_colors[i]
    
    if sgs_expert_algs:
        sgs_expert_colors = sns.color_palette("Greens", len(sgs_expert_algs))
        for i, alg in enumerate(sgs_expert_algs):
            colors[alg] = sgs_expert_colors[i]
    
    if sgs_expert_twice_algs:
        sgs_expert_twice_colors = sns.color_palette("Oranges", len(sgs_expert_twice_algs))
        for i, alg in enumerate(sgs_expert_twice_algs):
            colors[alg] = sgs_expert_twice_colors[i]
    
    return colors


def get_output_path(experiment_dir: str, plot_type: str, **kwargs) -> str:
    """Generate output path for plots in plots/ directory."""
    exp_num = os.path.basename(experiment_dir).split('_')[-1]
    
    if plot_type == "fixed_dimension":
        filename = f"exp{exp_num}_expert_{kwargs['metric']}_dim{kwargs['fixed_dim']}.png"
    elif plot_type == "fixed_sample_size":
        filename = f"exp{exp_num}_expert_{kwargs['metric']}_samples{kwargs['fixed_samples']}.png"
    elif plot_type == "one_setting":
        filename = f"exp{exp_num}_expert_dim{kwargs['dim']}_samples{kwargs['samples']}_edge_prob{format_edge_probability(kwargs['edge_probability'])}.png"
    elif plot_type == "expert_comparison":
        filename = f"exp{exp_num}_expert_comparison_{kwargs['metric']}_dim{kwargs['dim']}_samples{kwargs['samples']}_edge_prob{format_edge_probability(kwargs['edge_probability'])}.png"
    elif plot_type == "parameter_sweep":
        filename = f"exp{exp_num}_parameter_sweep_{kwargs['metric']}_{kwargs['parameter']}_dim{kwargs['dim']}_samples{kwargs['samples']}.png"
    elif plot_type == "parameter_sweep_real":
        filename = f"exp{exp_num}_parameter_sweep_{kwargs['metric']}_{kwargs['parameter']}_{kwargs['dataset']}_{kwargs['ci_test']}_samples{kwargs['samples']}.png"
    elif plot_type == "parameter_sweep_varying":
        filename = f"exp{exp_num}_parameter_sweep_varying_{kwargs['metric']}_{kwargs['vary_type']}_fixed{kwargs['fixed_value']}.png"
    else:
        filename = f"exp{exp_num}_expert_{plot_type}.png"
    
    # Get absolute path to plots directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    full_path = os.path.join(plots_dir, filename)
    print(f"Plot will be saved to: {full_path}")
    return full_path


def create_metadata_box(fig, config: Dict[str, Any], additional_info: Optional[str] = None):
    """Add metadata text box at bottom of figure."""
    metadata_text = f"Experiment Info: "
    metadata_text += f"Date: {config.get('timestamp', 'N/A')[:10]} | "
    metadata_text += f"Trials: {config.get('num_trials', 'N/A')} | "
    metadata_text += f"Alpha: {config.get('alpha', 'N/A')} | "
    
    # Handle different experiment types
    if 'graph_params' in config:
        # Synthetic experiment
        metadata_text += f"Dimensions: {config['graph_params']['dimensionalities']} | "
        metadata_text += f"Sample Sizes: {config['graph_params']['sample_sizes']} | "
        metadata_text += f"Edge Probabilities: {config['graph_params']['edge_probabilities']}"
    elif 'real_world_params' in config:
        # Real-world experiment
        datasets = [d['dataset'] for d in config['real_world_params']['datasets']]
        ci_tests = list(set([d['ci_test'] for d in config['real_world_params']['datasets']]))
        sample_sizes = list(set([d['sample_size'] for d in config['real_world_params']['datasets']]))
        metadata_text += f"Datasets: {datasets} | "
        metadata_text += f"CI Tests: {ci_tests} | "
        metadata_text += f"Sample Sizes: {sample_sizes}"
    
    # Add expert config info
    if 'pc_guess_expert_configs' in config:
        metadata_text += f" | Expert Configs: {len(config['pc_guess_expert_configs'])}"
    
    if additional_info:
        metadata_text += f" | {additional_info}"
    
    fig.text(0.5, 0.02, metadata_text, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            fontsize=8, wrap=True)


def plot_violin_fixed_dimension(experiment_dir: str, metric: str, fixed_dim: int, 
                               output_path: Optional[str] = None, median_only: bool = False):
    """Create violin plot with sample size on x-axis, holding dimensionality fixed."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data
    filtered_df = df[df['dimensionality'] == fixed_dim]
    
    if filtered_df.empty:
        print(f"No data found for dimension {fixed_dim}")
        return
    
    # Get unique edge_probabilities
    edge_probabilities = sorted(filtered_df['edge_probability'].unique())
    n_edge_probabilities = len(edge_probabilities)
    
    # Create figure
    fig, axes = plt.subplots(1, n_edge_probabilities, figsize=(6*n_edge_probabilities, 8))
    if n_edge_probabilities == 1:
        axes = [axes]
    
    # Get all algorithms and colors
    algorithms = sorted(filtered_df['algorithm'].unique())
    colors = get_color_palette(algorithms, config)
    
    for i, edge_probability in enumerate(edge_probabilities):
        ax = axes[i]
        edge_prob_data = filtered_df[filtered_df['edge_probability'] == edge_probability]
        
        # Create plot based on median_only flag
        if median_only:
            # Calculate medians and plot as points/lines
            median_data = edge_prob_data.groupby(['sample_size', 'algorithm'])[metric].median().reset_index()
            for alg in algorithms:
                alg_data = median_data[median_data['algorithm'] == alg]
                ax.plot(alg_data['sample_size'], alg_data[metric], 
                       marker='o', label=alg, color=colors[alg], linewidth=2, markersize=6)
            ax.legend()
        else:
            # Create violin plot
            sns.violinplot(data=edge_prob_data, x='sample_size', y=metric, 
                          hue='algorithm', ax=ax, palette=colors)
        
        ax.set_xlabel("Sample Size")
        if metric == "runtime":
            ax.set_yscale('log', base=2)
            ax.set_ylabel(f'{metric.upper()} (Log₂ Scale)')
        else:
            ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(metric.upper())
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at bottom
    
    # Add metadata box at bottom
    create_metadata_box(fig, config)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "fixed_dimension", 
                                    metric=metric, fixed_dim=fixed_dim)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def plot_violin_fixed_sample_size(experiment_dir: str, metric: str, fixed_samples: int,
                                 output_path: Optional[str] = None, median_only: bool = False):
    """Create violin plot with dimensionality on x-axis, holding sample size fixed."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data
    filtered_df = df[df['sample_size'] == fixed_samples]
    
    if filtered_df.empty:
        print(f"No data found for sample size {fixed_samples}")
        return
    
    # Get unique edge_probabilities
    edge_probabilities = sorted(filtered_df['edge_probability'].unique())
    n_edge_probabilities = len(edge_probabilities)
    
    # Create figure
    fig, axes = plt.subplots(1, n_edge_probabilities, figsize=(6*n_edge_probabilities, 8))
    if n_edge_probabilities == 1:
        axes = [axes]
    
    # Get all algorithms and colors
    algorithms = sorted(filtered_df['algorithm'].unique())
    colors = get_color_palette(algorithms, config)
    
    for i, edge_probability in enumerate(edge_probabilities):
        ax = axes[i]
        edge_prob_data = filtered_df[filtered_df['edge_probability'] == edge_probability]
        
        # Create plot based on median_only flag
        if median_only:
            # Calculate medians and plot as points/lines
            median_data = edge_prob_data.groupby(['dimensionality', 'algorithm'])[metric].median().reset_index()
            for alg in algorithms:
                alg_data = median_data[median_data['algorithm'] == alg]
                ax.plot(alg_data['dimensionality'], alg_data[metric], 
                       marker='o', label=alg, color=colors[alg], linewidth=2, markersize=6)
            ax.legend()
        else:
            # Create violin plot
            sns.violinplot(data=edge_prob_data, x='dimensionality', y=metric,
                          hue='algorithm', ax=ax, palette=colors)
        
        ax.set_xlabel("Dimensionality")
        if metric == "runtime":
            ax.set_yscale('log', base=2)
            ax.set_ylabel(f'{metric.upper()} (Log₂ Scale)')
        else:
            ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(metric.upper())
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at bottom
    
    # Add metadata box at bottom
    create_metadata_box(fig, config)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "fixed_sample_size", 
                                    metric=metric, fixed_samples=fixed_samples)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def plot_one_setting(experiment_dir: str, dim: int, samples: int, edge_probability: float,
                    metrics_list: List[str], output_path: Optional[str] = None, median_only: bool = False):
    """Plot all metrics for one specific parameter combination."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data for specific setting
    filtered_df = df[(df['dimensionality'] == dim) & 
                    (df['sample_size'] == samples) & 
                    (df['edge_probability'] == edge_probability)]
    
    if filtered_df.empty:
        print(f"No data found for dim={dim}, samples={samples}, edge_probability={edge_probability}")
        return
    
    # Create figure
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]
    
    # Get all algorithms and colors
    algorithms = sorted(filtered_df['algorithm'].unique())
    colors = get_color_palette(algorithms, config)
    
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        
        # Create plot based on median_only flag
        if median_only:
            # Calculate medians and plot as bar chart
            median_data = filtered_df.groupby('algorithm')[metric].median().reset_index()
            bars = ax.bar(median_data['algorithm'], median_data[metric], 
                         color=[colors[alg] for alg in median_data['algorithm']])
            ax.set_xticklabels(median_data['algorithm'], rotation=45)
        else:
            # Create violin plot
            sns.violinplot(data=filtered_df, x='algorithm', y=metric, ax=ax, palette=colors)
        
        ax.set_title(metric.upper())
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(', '.join([m.upper() for m in metrics_list]))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add space at bottom
    
    # Add metadata box at bottom
    additional_info = f"Setting: dim={dim}, samples={samples}, edge_probability={edge_probability}"
    create_metadata_box(fig, config, additional_info)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "one_setting", 
                                    dim=dim, samples=samples, edge_probability=edge_probability)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def plot_expert_comparison(experiment_dir: str, metric: str, dim: int, samples: int, edge_probability: float,
                          output_path: Optional[str] = None, median_only: bool = False):
    """Create detailed comparison plot showing expert configurations with parameter labels."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data for specific setting
    filtered_df = df[(df['dimensionality'] == dim) & 
                    (df['sample_size'] == samples) & 
                    (df['edge_probability'] == edge_probability)]
    
    if filtered_df.empty:
        print(f"No data found for dim={dim}, samples={samples}, edge_probability={edge_probability}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get all algorithms and colors
    algorithms = sorted(filtered_df['algorithm'].unique())
    colors = get_color_palette(algorithms, config)
    
    # Create plot based on median_only flag
    if median_only:
        # Calculate medians and plot as bar chart
        median_data = filtered_df.groupby('algorithm')[metric].median().reset_index()
        bars = ax.bar(median_data['algorithm'], median_data[metric], 
                     color=[colors[alg] for alg in median_data['algorithm']])
    else:
        # Create violin plot
        sns.violinplot(data=filtered_df, x='algorithm', y=metric, ax=ax, palette=colors)
    
    # Customize x-axis labels to show parameter values
    new_labels = []
    for alg in algorithms:
        if alg.startswith('pc_guess_expert_'):
            config_idx = int(alg.split('_')[-1])
            expert_config = config['pc_guess_expert_configs'][config_idx]
            label = f"Expert_{config_idx}\n(T:{expert_config['p_acc_edge_true']}, F:{expert_config['p_acc_edge_false']},\nMD:{expert_config['p_acc_subset_MD']}, NMD:{expert_config['p_acc_subset_NMD']})"
        else:
            label = alg.upper()
        new_labels.append(label)
    
    ax.set_xticklabels(new_labels, fontsize=10)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add space at bottom
    
    # Add metadata box at bottom
    additional_info = f"Setting: dim={dim}, samples={samples}, edge_probability={edge_probability}"
    create_metadata_box(fig, config, additional_info)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "expert_comparison", 
                                    metric=metric, dim=dim, samples=samples, edge_probability=edge_probability)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def plot_parameter_sweep(experiment_dir: str, metric: str, parameter: str, 
                        dim: int = None, samples: int = None, dataset: str = None, ci_test: str = None,
                        output_path: Optional[str] = None, error_type: str = "sem", show_legend: bool = True):
    """Plot parameter sweep showing parameter-varying algorithms as points and parameter-invariant as horizontal lines."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Detect experiment type and filter accordingly
    exp_type = detect_experiment_type(experiment_dir)
    
    if exp_type == "synthetic":
        if dim is None or samples is None:
            raise ValueError("For synthetic experiments, both dim and samples must be specified")
        filtered_df = df[(df['dimensionality'] == dim) & (df['sample_size'] == samples)]
        setting_label = f"dim={dim}, samples={samples}"
    elif exp_type == "real_world":
        if dataset is None or ci_test is None or samples is None:
            raise ValueError("For real-world experiments, dataset, ci_test, and samples must be specified")
        filtered_df = df[(df['dataset'] == dataset) & (df['ci_test'] == ci_test) & (df['sample_size'] == samples)]
        setting_label = f"dataset={dataset}, ci_test={ci_test}, samples={samples}"
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    if filtered_df.empty:
        print(f"No data found for {setting_label}")
        return
    
    # Map parameter name
    if parameter == "p_edge_acc":
        param_key = "p_acc_edge_true"  # Assuming p_acc_edge_true == p_acc_edge_false
    else:
        param_key = parameter
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get all algorithms
    algorithms = sorted(filtered_df['algorithm'].unique())
    
    # Separate parameter-varying and parameter-invariant algorithms
    param_varying = []
    param_invariant = []
    
    for alg in algorithms:
        if ((alg.startswith('pc_guess_expert_') or alg.startswith('sgs_guess_expert_') or 
            alg.startswith('pc_guess_expert_twice_') or alg.startswith('sgs_guess_expert_twice_')) and 
            not alg.endswith('_r')):
            param_varying.append(alg)
        else:
            param_invariant.append(alg)
    
    # Collect all parameter values to determine x-axis range
    param_values = []
    for alg in param_varying:
        if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_twice_configs'][config_idx][param_key]
        elif alg.startswith('pc_guess_expert_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_configs'][config_idx][param_key]
        elif alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_twice_configs'][config_idx][param_key]
        elif alg.startswith('sgs_guess_expert_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_configs'][config_idx][param_key]
        else:
            continue  # Skip algorithms that don't match expected patterns
        param_values.append(param_value)
    
    # Get base algorithm types for coloring
    base_algorithms = set()
    for alg in algorithms:
        if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r'):
            base_algorithms.add('pc_guess_expert_twice')
        elif alg.startswith('pc_guess_expert_') and not alg.endswith('_r'):
            base_algorithms.add('pc_guess_expert')
        elif alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r'):
            base_algorithms.add('sgs_guess_expert_twice')
        elif alg.startswith('sgs_guess_expert_') and not alg.endswith('_r'):
            base_algorithms.add('sgs_guess_expert')
        else:
            base_algorithms.add(alg)
    
    # Use fixed reference colors to match plot_parameter_sweep_varying
    reference_colors = get_reference_colors()
    color_map = {}
    for base_alg in base_algorithms:
        if base_alg in reference_colors:
            color_map[base_alg] = reference_colors[base_alg]
        else:
            # Fallback for other algorithms
            other_colors = sns.color_palette("Set1", 8)
            color_map[base_alg] = other_colors[len(color_map) % len(other_colors)]
    
    # Calculate means, standard deviations, and counts for all algorithms
    mean_data = filtered_df.groupby('algorithm')[metric].agg(['mean', 'std', 'count']).reset_index()
    mean_data.columns = ['algorithm', 'mean', 'std', 'count']
    # Calculate SEM (Standard Error of the Mean) = std / sqrt(count)
    mean_data['sem'] = mean_data['std'] / np.sqrt(mean_data['count'])
    # Choose error metric based on error_type parameter
    error_col = 'sem' if error_type == 'sem' else 'std'
    
    # Track which legend entries have been added
    legend_added = set()
    
    # Plot parameter-invariant algorithms as horizontal dotted lines (exclude -r algorithms and PC)
    for alg in param_invariant:
        if alg.endswith('_r') or alg == 'pc':  # Skip -r algorithms and PC method
            continue
            
        alg_stats = mean_data[mean_data['algorithm'] == alg]
        alg_mean = alg_stats['mean'].iloc[0]
        alg_error = alg_stats[error_col].iloc[0]
        
        # Add label only to first occurrence of each algorithm type
        if alg not in legend_added:
            display_name = "PC-Stable" if alg == "stable_pc" else alg.upper()
            ax.axhline(y=alg_mean, color=color_map[alg], linestyle='--', alpha=0.7,
                      label=display_name)
            # Add error band for horizontal lines (skip for stable_pc)
            if alg not in ['stable_pc']:
                ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 
                               alg_mean - alg_error, alg_mean + alg_error, 
                               color=color_map[alg], alpha=0.2)
            legend_added.add(alg)
        else:
            ax.axhline(y=alg_mean, color=color_map[alg], linestyle='--', alpha=0.7)
            # Add error band for horizontal lines (skip for stable_pc)
            if alg not in ['stable_pc']:
                ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 
                               alg_mean - alg_error, alg_mean + alg_error, 
                               color=color_map[alg], alpha=0.2)
    
    # Plot -r algorithms as horizontal lines without error bars
    r_algorithms = ['pc_guess_expert_r', 'pc_guess_expert_twice_r', 'sgs_guess_expert_r']
    for r_alg in r_algorithms:
        if r_alg in algorithms:
            alg_stats = mean_data[mean_data['algorithm'] == r_alg]
            if not alg_stats.empty:
                alg_mean = alg_stats['mean'].iloc[0]
                
                # Determine display name and color
                if r_alg == 'sgs_guess_expert_r':
                    display_name = "gPC"
                    color = color_map.get('sgs_guess_expert', 'gray')
                elif r_alg == 'pc_guess_expert_twice_r':
                    display_name = "PC-GUESS-TWICE-R"
                    color = color_map.get('pc_guess_expert_twice', 'gray')
                elif r_alg == 'pc_guess_expert_r':
                    display_name = "PC"
                    color = color_map.get('pc_guess_expert', 'gray')
                
                ax.axhline(y=alg_mean, color=color, linestyle=':', alpha=0.7,
                          label=display_name)
    
    # Plot parameter-varying algorithms as points
    for alg in param_varying:
        # Get parameter value from config
        if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_twice_configs'][config_idx][param_key]
            base_alg = 'pc_guess_expert_twice'
        elif alg.startswith('pc_guess_expert_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_configs'][config_idx][param_key]
            base_alg = 'pc_guess_expert'
        elif alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_twice_configs'][config_idx][param_key]
            base_alg = 'sgs_guess_expert_twice'
        elif alg.startswith('sgs_guess_expert_') and not alg.endswith('_r'):
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_configs'][config_idx][param_key]
            base_alg = 'sgs_guess_expert'
        else:
            continue  # Skip algorithms that don't match expected patterns
        
        alg_stats = mean_data[mean_data['algorithm'] == alg]
        alg_mean = alg_stats['mean'].iloc[0]
        alg_error = alg_stats[error_col].iloc[0]
        
        # Add label only to first occurrence of each base algorithm type
        if base_alg not in legend_added:
            display_name = "PC-GUESS-TWICE" if base_alg == "pc_guess_expert_twice" else "PC-Guess" if base_alg == "pc_guess_expert" else "gPC-Guess" if base_alg == "sgs_guess_expert" else base_alg.upper().replace('_', '-')
            ax.errorbar(param_value, alg_mean, yerr=alg_error, fmt='o', 
                       color=color_map[base_alg], markersize=10, alpha=0.8, capsize=5,
                       label=display_name)
            legend_added.add(base_alg)
        else:
            ax.errorbar(param_value, alg_mean, yerr=alg_error, fmt='o', 
                       color=color_map[base_alg], markersize=10, alpha=0.8, capsize=5)
    
    # Connect points of same base algorithm with lines
    for base_alg in base_algorithms:
        if base_alg in ['pc_guess_expert', 'pc_guess_expert_twice', 'sgs_guess_expert', 'sgs_guess_expert_twice']:
            base_algs = [alg for alg in param_varying if alg.startswith(base_alg)]
            if len(base_algs) > 1:
                x_vals = []
                y_vals = []
                for alg in sorted(base_algs):
                    config_idx = int(alg.split('_')[-1])
                    if base_alg == 'pc_guess_expert':
                        param_value = config['pc_guess_expert_configs'][config_idx][param_key]
                    elif base_alg == 'pc_guess_expert_twice':
                        param_value = config['pc_guess_expert_twice_configs'][config_idx][param_key]
                    elif base_alg == 'sgs_guess_expert':
                        param_value = config['sgs_guess_expert_configs'][config_idx][param_key]
                    elif base_alg == 'sgs_guess_expert_twice':
                        param_value = config['sgs_guess_expert_twice_configs'][config_idx][param_key]
                    alg_stats = mean_data[mean_data['algorithm'] == alg]
                    alg_mean = alg_stats['mean'].iloc[0]
                    x_vals.append(param_value)
                    y_vals.append(alg_mean)
                
                ax.plot(x_vals, y_vals, color=color_map[base_alg], alpha=0.5, linewidth=1)
    
    # Set x-axis limits based on actual parameter values with padding
    if param_values:
        min_param = min(param_values)
        max_param = max(param_values)
        padding = (max_param - min_param) * 0.05  # 5% padding
        ax.set_xlim(min_param - padding, max_param + padding)
    
    xlabel = "Expert Edge Prediction Accuracy (p_psi)" if parameter == "p_edge_acc" else parameter.upper()
    ax.set_xlabel(xlabel)
    error_label = "SEM" if error_type == "sem" else "STD"
    if metric == "runtime":
        ax.set_yscale('log', base=2)
        ax.set_ylabel(f"{metric.upper()} (Mean ± {error_label}, Log₂ Scale)")
    else:
        ax.set_ylabel(f"{metric.upper()} (Mean ± {error_label})")

    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Add metadata box
    additional_info = f"Setting: {setting_label}, parameter={parameter}"
    create_metadata_box(fig, config, additional_info)
    
    # Generate output path if not provided
    if output_path is None:
        if exp_type == "synthetic":
            output_path = get_output_path(experiment_dir, "parameter_sweep", 
                                        metric=metric, parameter=parameter, dim=dim, samples=samples)
        else:
            output_path = get_output_path(experiment_dir, "parameter_sweep_real", 
                                        metric=metric, parameter=parameter, dataset=dataset, ci_test=ci_test, samples=samples)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


def plot_parameter_sweep_varying(experiment_dir: str, metric: str, 
                               vary_type: str, fixed_value: int, 
                               methods: Optional[List[str]] = None,
                               output_path: Optional[str] = None):
    """Plot parameter sweep with varying dimensionality or sample size, with method filtering and enhanced color/symbol scheme."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data based on vary_type
    if vary_type == "sample_size":
        filtered_df = df[df['dimensionality'] == fixed_value]
        x_column = 'sample_size'
        fixed_label = f"dim={fixed_value}"
    elif vary_type == "dimensionality":
        filtered_df = df[df['sample_size'] == fixed_value]
        x_column = 'dimensionality'
        fixed_label = f"samples={fixed_value}"
    else:
        raise ValueError("vary_type must be 'sample_size' or 'dimensionality'")
    
    if filtered_df.empty:
        print(f"No data found for {fixed_label}")
        return
    
    # Get all algorithms and filter by methods if specified
    all_algorithms = sorted(filtered_df['algorithm'].unique())
    
    # Filter out unwanted algorithms
    all_algorithms = [alg for alg in all_algorithms if alg != 'pc']
    
    if methods is not None:
        # Filter algorithms by base method types
        filtered_algorithms = []
        for alg in all_algorithms:
            for method in methods:
                if (alg == method or 
                    alg.startswith(f"{method}_") or 
                    (method == 'pc_guess_expert' and alg.startswith('pc_guess_expert_') and not alg.startswith('pc_guess_expert_twice_')) or
                    (method == 'pc_guess_expert_twice' and alg.startswith('pc_guess_expert_twice_')) or
                    (method == 'sgs_guess_expert' and alg.startswith('sgs_guess_expert_') and not alg.startswith('sgs_guess_expert_twice_')) or
                    (method == 'sgs_guess_expert_twice' and alg.startswith('sgs_guess_expert_twice_'))):
                    filtered_algorithms.append(alg)
                    break
        algorithms = filtered_algorithms
    else:
        algorithms = all_algorithms
    
    if not algorithms:
        print(f"No algorithms found matching methods: {methods}")
        return
    
    # Filter dataframe to only include selected algorithms
    filtered_df = filtered_df[filtered_df['algorithm'].isin(algorithms)]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get base method types and parameter values
    base_methods = set()
    param_values = set()
    
    for alg in algorithms:
        if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r'):
            base_methods.add('pc_guess_expert_twice')
            config_idx = int(alg.split('_')[-1])
            param_values.add(config['pc_guess_expert_twice_configs'][config_idx]['p_acc_edge_true'])
        elif alg.startswith('pc_guess_expert_') and not alg.endswith('_r'):
            base_methods.add('pc_guess_expert')
            config_idx = int(alg.split('_')[-1])
            param_values.add(config['pc_guess_expert_configs'][config_idx]['p_acc_edge_true'])
        elif alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r'):
            base_methods.add('sgs_guess_expert_twice')
            config_idx = int(alg.split('_')[-1])
            param_values.add(config['sgs_guess_expert_twice_configs'][config_idx]['p_acc_edge_true'])
        elif alg.startswith('sgs_guess_expert_') and not alg.endswith('_r'):
            base_methods.add('sgs_guess_expert')
            config_idx = int(alg.split('_')[-1])
            param_values.add(config['sgs_guess_expert_configs'][config_idx]['p_acc_edge_true'])
        else:
            base_methods.add(alg)
    
    # Create color mapping for base methods
    base_methods_list = sorted(base_methods)
    method_colors = sns.color_palette("Set1", len(base_methods_list))
    color_map = {method: method_colors[i] for i, method in enumerate(base_methods_list)}
    
    # Create symbol mapping for parameter values
    param_values_list = sorted(param_values)
    symbols = ['o', 's', '^', 'v', 'D', 'P']  # circle, square, triangle_up, triangle_down, diamond, plus
    symbol_map = {param: symbols[i % len(symbols)] for i, param in enumerate(param_values_list)}
    
    # Calculate means for plotting
    mean_data = filtered_df.groupby(['algorithm', x_column])[metric].mean().reset_index()
    
    # Track legend entries
    method_legend_added = set()
    param_legend_added = set()
    
    # Plot each algorithm
    for alg in algorithms:
        alg_data = mean_data[mean_data['algorithm'] == alg]
        
        if alg_data.empty:
            continue
        
        # Determine base method, color, and symbol
        if alg.startswith('pc_guess_expert_twice_') and not alg.endswith('_r'):
            base_method = 'pc_guess_expert_twice'
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_twice_configs'][config_idx]['p_acc_edge_true']
            linestyle = '-.'
        elif alg.startswith('pc_guess_expert_') and not alg.endswith('_r'):
            base_method = 'pc_guess_expert'
            config_idx = int(alg.split('_')[-1])
            param_value = config['pc_guess_expert_configs'][config_idx]['p_acc_edge_true']
            linestyle = ':'
        elif alg.startswith('sgs_guess_expert_twice_') and not alg.endswith('_r'):
            base_method = 'sgs_guess_expert_twice'
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_twice_configs'][config_idx]['p_acc_edge_true']
            linestyle = '-.'
        elif alg.startswith('sgs_guess_expert_') and not alg.endswith('_r'):
            base_method = 'sgs_guess_expert'
            config_idx = int(alg.split('_')[-1])
            param_value = config['sgs_guess_expert_configs'][config_idx]['p_acc_edge_true']
            linestyle = ':'
        else:
            base_method = alg
            param_value = None
            linestyle = '-'
        
        color = color_map[base_method]
        marker = symbol_map.get(param_value, 'o') if param_value is not None else 'o'
        
        # Plot line (no label here, will add to legends separately)
        ax.plot(alg_data[x_column], alg_data[metric], 
               marker=marker, color=color, linestyle=linestyle, 
               linewidth=2, markersize=8, alpha=0.8)
    
    # Create method legend (colors)
    method_handles = []
    for method in base_methods_list:
        if method == 'pc':
            continue  # Skip PC method
        elif method == 'pc_guess_expert':
            label = 'PC-Guess'
            style = ':'
        elif method == 'pc_guess_expert_twice':
            label = 'PC-GUESS-TWICE'
            style = '-.'
        elif method == 'sgs_guess_expert':
            label = 'gPC-Guess'
            style = ':'
        elif method == 'sgs_guess_expert_twice':
            label = 'SGS-GUESS-EXPERT-TWICE'
            style = '-.'
        elif method == 'stable_pc':
            label = 'PC-Stable'
            style = '-'
        else:
            label = method.upper()
            style = '-'
        
        handle = plt.Line2D([0], [0], color=color_map[method], linestyle=style, 
                           linewidth=2, label=label)
        method_handles.append(handle)
    
    # Create parameter legend (symbols)
    param_handles = []
    for param in param_values_list:
        handle = plt.Line2D([0], [0], marker=symbol_map[param], color='black', 
                           linestyle='None', markersize=8, label=f'{param}')
        param_handles.append(handle)
    
    # Add legends with conditional positioning
    if vary_type == "sample_size":
        # Bottom right positioning for sample_size plots
        if method_handles:
            method_legend = ax.legend(handles=method_handles, title='Methods', 
                                     bbox_to_anchor=(1, 0.3), loc='lower right')
            ax.add_artist(method_legend)
        
        if param_handles:
            param_legend = ax.legend(handles=param_handles, title='P_Edge Values', 
                                    bbox_to_anchor=(1, 0.15), loc='lower right')
    elif vary_type == "dimensionality":
        # Top right positioning for dimensionality plots - methods left, p_edge right
        if method_handles:
            method_legend = ax.legend(handles=method_handles, title='Methods', 
                                     bbox_to_anchor=(0.7, 1), loc='upper right')
            ax.add_artist(method_legend)
        
        if param_handles:
            param_legend = ax.legend(handles=param_handles, title='P_Edge Values', 
                                    bbox_to_anchor=(1, 1), loc='upper right')
    else:
        # Upper left positioning for other plots
        if method_handles:
            method_legend = ax.legend(handles=method_handles, title='Methods', 
                                     loc='upper left')
            ax.add_artist(method_legend)
        
        if param_handles:
            param_legend = ax.legend(handles=param_handles, title='P_Edge Values', 
                                    bbox_to_anchor=(0, 0.7), loc='upper left')
    
    ax.set_xlabel(vary_type.replace('_', ' ').title())
    if metric == "runtime":
        ax.set_yscale('log', base=2)
        ax.set_ylabel(f"{metric.upper()} (Mean, Log₂ Scale)")
    else:
        ax.set_ylabel(f"{metric.upper()} (Mean)")
    
    # Set x-axis ticks for sample_size plots
    if vary_type == "sample_size":
        unique_samples = sorted(filtered_df['sample_size'].unique())
        ax.set_xticks(unique_samples)

    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, right=0.75)  # Make room for legends
    
    # Add metadata box
    methods_str = ', '.join(methods) if methods else 'all'
    additional_info = f"Setting: {fixed_label}, varying {vary_type}, methods: {methods_str}"
    create_metadata_box(fig, config, additional_info)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "parameter_sweep_varying", 
                                    metric=metric, vary_type=vary_type, fixed_value=fixed_value)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Toggle for legend display
    show_legend = True  # Set to False to hide legends
    
    # Example usage - update experiment_dir with your actual experiment number
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    experiment_dir = os.path.join(project_root, "results", "experiment_396")
    
    print(f"Looking for experiment at: {experiment_dir}")
    
    if not os.path.exists(experiment_dir):
        print(f"ERROR: Experiment directory does not exist: {experiment_dir}")
        print("Available experiments:")
        results_dir = os.path.join(project_root, "results")
        if os.path.exists(results_dir):
            for exp in sorted(os.listdir(results_dir)):
                if exp.startswith("experiment_"):
                    print(f"  {exp}")
        exit()
    
    # Detect experiment type and show appropriate examples
    exp_type = detect_experiment_type(experiment_dir)
    print(f"Detected experiment type: {exp_type}")
    
    if exp_type == "synthetic":
        # Synthetic experiment examples
        # plot_parameter_sweep(experiment_dir, "runtime", "p_acc_subset_MD", dim=10, samples=30, error_type="sem")
        # plot_parameter_sweep(experiment_dir, "f1", "p_edge_acc", dim=10, samples=100, error_type="sem", show_legend=show_legend)
        # plot_parameter_sweep(experiment_dir, "f1", "p_edge_acc", dim=20, samples=100, error_type="sem")
        # plot_parameter_sweep_varying(experiment_dir, "f1", "sample_size", fixed_value=10)
        # plot_parameter_sweep_varying(experiment_dir, "f1", "sample_size", fixed_value=10)
        # plot_parameter_sweep_varying(experiment_dir, "runtime", "dimensionality", fixed_value=100)
        plot_parameter_sweep_varying(experiment_dir, "f1", "dimensionality", fixed_value=100)
    elif exp_type == "real_world":
        # Real-world experiment examples  
        plot_parameter_sweep(experiment_dir, "f1", "p_edge_acc", dataset="sachs", ci_test="chisq", samples=100, error_type="sem", show_legend=show_legend)
    else:
        print(f"Unknown experiment type: {exp_type}")

    # Additional examples (commented out)
    # Synthetic examples:
    # plot_parameter_sweep_varying(experiment_dir, "f1", "sample_size", fixed_value=10)
    # plot_violin_fixed_dimension(experiment_dir, "f1", fixed_dim=10, median_only=True)
    
    # Real-world examples:
    # (Add real-world specific plotting functions as needed)
    
    print("Expert plots completed.")