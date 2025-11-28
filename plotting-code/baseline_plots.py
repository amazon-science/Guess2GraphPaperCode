"""
Baseline Plots Module

Create violin plots from experiment results for algorithm comparison.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import numpy as np


def format_sparsity(sparsity: float) -> str:
    """Format sparsity value for directory names."""
    return str(sparsity).replace('.', '_')


def load_experiment_data(experiment_dir: str) -> pd.DataFrame:
    """Load all metrics data from experiment directory into DataFrame."""
    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        available_experiments = [d for d in os.listdir("results") if d.startswith("experiment_")] if os.path.exists("results") else []
        raise FileNotFoundError(f"Experiment directory '{experiment_dir}' not found. Available experiments: {available_experiments}")
    
    # Load config
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at '{config_path}'. Experiment may not have completed successfully.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_rows = []
    
    # Iterate through parameter combinations
    for dim in config['graph_params']['dimensionalities']:
        for sample_size in config['graph_params']['sample_sizes']:
            for sparsity in config['graph_params']['sparsities']:
                combo_dir = f"dim_{dim}_samples_{sample_size}_sparsity_{format_sparsity(sparsity)}"
                combo_path = os.path.join(experiment_dir, combo_dir)
                
                if not os.path.exists(combo_path):
                    continue
                
                # Load trial data
                for trial_file in os.listdir(combo_path):
                    if trial_file.startswith("trial_"):
                        metrics_file = os.path.join(combo_path, trial_file, "metrics.json")
                        if os.path.exists(metrics_file):
                            with open(metrics_file, 'r') as f:
                                trial_data = json.load(f)
                            
                            trial_num = int(trial_file.split("_")[1])
                            
                            # Extract metrics for each algorithm
                            for algorithm in config['algorithms']:
                                if algorithm in trial_data:
                                    row = {
                                        'dimensionality': dim,
                                        'sample_size': sample_size,
                                        'sparsity': sparsity,
                                        'algorithm': algorithm,
                                        'trial_number': trial_num
                                    }
                                    # Add all metrics
                                    for metric in config['metrics']:
                                        if metric in trial_data[algorithm]:
                                            row[metric] = trial_data[algorithm][metric]
                                    
                                    # Add runtime if available
                                    if 'runtimes' in trial_data and algorithm in trial_data['runtimes']:
                                        row['runtime'] = trial_data['runtimes'][algorithm]
                                    
                                    data_rows.append(row)
    
    return pd.DataFrame(data_rows)


def get_color_palette(algorithms: List[str]) -> Dict[str, str]:
    """Get consistent color palette for algorithms."""
    colors = sns.color_palette("Set2", len(algorithms))
    return {alg: colors[i] for i, alg in enumerate(algorithms)}


def get_output_path(experiment_dir: str, plot_type: str, **kwargs) -> str:
    """Generate output path for plots in plots/ directory."""
    exp_num = os.path.basename(experiment_dir).split('_')[-1]
    
    if plot_type == "fixed_dimension":
        filename = f"exp{exp_num}_{kwargs['metric']}_dim{kwargs['fixed_dim']}.png"
    elif plot_type == "fixed_sample_size":
        filename = f"exp{exp_num}_{kwargs['metric']}_samples{kwargs['fixed_samples']}.png"
    elif plot_type == "one_setting":
        filename = f"exp{exp_num}_dim{kwargs['dim']}_samples{kwargs['samples']}_sparsity{format_sparsity(kwargs['sparsity'])}.png"
    else:
        filename = f"exp{exp_num}_{plot_type}.png"
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)


def create_metadata_box(fig, config: Dict[str, Any], additional_info: Optional[str] = None):
    """Add metadata text box at bottom of figure."""
    metadata_text = f"Experiment Info: "
    metadata_text += f"Date: {config.get('timestamp', 'N/A')[:10]} | "
    metadata_text += f"Trials: {config.get('num_trials', 'N/A')} | "
    metadata_text += f"Alpha: {config.get('alpha', 'N/A')} | "
    metadata_text += f"Dimensions: {config['graph_params']['dimensionalities']} | "
    metadata_text += f"Sample Sizes: {config['graph_params']['sample_sizes']} | "
    metadata_text += f"Sparsities: {config['graph_params']['sparsities']} | "
    metadata_text += f"Algorithms: {config['algorithms']}"
    
    if additional_info:
        metadata_text += f" | {additional_info}"
    
    fig.text(0.5, 0.02, metadata_text, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            fontsize=8, wrap=True)


def plot_violin_fixed_dimension(experiment_dir: str, metric: str, fixed_dim: int, 
                               output_path: Optional[str] = None):
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
    
    # Get unique sparsities
    sparsities = sorted(filtered_df['sparsity'].unique())
    n_sparsities = len(sparsities)
    
    # Create figure
    fig, axes = plt.subplots(1, n_sparsities, figsize=(5*n_sparsities, 6))
    if n_sparsities == 1:
        axes = [axes]
    
    colors = get_color_palette(config['algorithms'])
    
    for i, sparsity in enumerate(sparsities):
        ax = axes[i]
        sparsity_data = filtered_df[filtered_df['sparsity'] == sparsity]
        
        # Create violin plot
        sns.violinplot(data=sparsity_data, x='sample_size', y=metric, 
                      hue='algorithm', ax=ax, palette=colors)
        
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(metric.upper())
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom
    
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
                                 output_path: Optional[str] = None):
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
    
    # Get unique sparsities
    sparsities = sorted(filtered_df['sparsity'].unique())
    n_sparsities = len(sparsities)
    
    # Create figure
    fig, axes = plt.subplots(1, n_sparsities, figsize=(5*n_sparsities, 6))
    if n_sparsities == 1:
        axes = [axes]
    
    colors = get_color_palette(config['algorithms'])
    
    for i, sparsity in enumerate(sparsities):
        ax = axes[i]
        sparsity_data = filtered_df[filtered_df['sparsity'] == sparsity]
        
        # Create violin plot
        sns.violinplot(data=sparsity_data, x='dimensionality', y=metric,
                      hue='algorithm', ax=ax, palette=colors)
        
        ax.set_xlabel("Dimensionality")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(metric.upper())
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom
    
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


def plot_one_setting(experiment_dir: str, dim: int, samples: int, sparsity: float,
                    metrics_list: List[str], output_path: Optional[str] = None):
    """Plot all metrics for one specific parameter combination."""
    # Load data
    df = load_experiment_data(experiment_dir)
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Filter data for specific setting
    filtered_df = df[(df['dimensionality'] == dim) & 
                    (df['sample_size'] == samples) & 
                    (df['sparsity'] == sparsity)]
    
    if filtered_df.empty:
        print(f"No data found for dim={dim}, samples={samples}, sparsity={sparsity}")
        return
    
    # Create figure
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    colors = get_color_palette(config['algorithms'])
    
    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        
        # Create violin plot
        sns.violinplot(data=filtered_df, x='algorithm', y=metric, ax=ax, palette=colors)
        
        ax.set_title(metric.upper())
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(', '.join([m.upper() for m in metrics_list]))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom
    
    # Add metadata box at bottom
    additional_info = f"Setting: dim={dim}, samples={samples}, sparsity={sparsity}"
    create_metadata_box(fig, config, additional_info)
    
    # Generate output path if not provided
    if output_path is None:
        output_path = get_output_path(experiment_dir, "one_setting", 
                                    dim=dim, samples=samples, sparsity=sparsity)
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Example usage
    experiment_dir = "results/experiment_055"
    
    # Plot F1 scores for all sample sizes when dimension=20 (matches config)
    plot_violin_fixed_dimension(experiment_dir, "f1", fixed_dim=20)
    
    # Plot precision for all dimensions when samples=1000
    # plot_violin_fixed_sample_size(experiment_dir, "f1", fixed_samples=300)
    
    # Plot all metrics for one specific setting
    # plot_one_setting(experiment_dir, dim=5, samples=500, sparsity=1, 
    #                 metrics_list=["f1", "precision", "recall"])
    
    print("Baseline plots module loaded. Use the plotting functions to visualize experiment results.")