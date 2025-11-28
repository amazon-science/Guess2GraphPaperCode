import os
import json
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'guess-code'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.causality_paper_guess_to_graph.metrics import evaluate_skeleton_recovery

# Configuration
DATASET_NAME = "sachs"
EXPERIMENT_NUMBER = 37  # Hardcoded for now
PLOTS_DIR = "llm-guess-plots"

def is_valid_dag(matrix):
    """Check if matrix represents a valid DAG"""
    n = len(matrix)
    
    # Check square
    if any(len(row) != n for row in matrix):
        return False
    
    # Check binary
    for row in matrix:
        for val in row:
            if val not in [0, 1]:
                return False
    
    # Check no self-loops
    for i in range(n):
        if matrix[i][i] != 0:
            return False
    
    # Check acyclic using DFS
    def has_cycle_dfs():
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def visit(node):
            if color[node] == GRAY:
                return True  # Back edge found
            if color[node] == BLACK:
                return False
            
            color[node] = GRAY
            for neighbor in range(n):
                if matrix[node][neighbor] == 1:
                    if visit(neighbor):
                        return True
            color[node] = BLACK
            return False
        
        for i in range(n):
            if color[i] == WHITE:
                if visit(i):
                    return True
        return False
    
    return not has_cycle_dfs()

def load_ground_truth(dataset_name):
    """Load ground truth adjacency matrix and variable names."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "real-world-data", f"{dataset_name}-real-world-data.npy")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
    
    adj_matrix, data, variable_names = np.load(dataset_path, allow_pickle=True)
    return adj_matrix, list(variable_names)

def load_experiment_results(experiment_number):
    """Load all model results from experiment directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(project_root, "guess-llms-results", f"guess_experiment_{experiment_number}")
    
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment {experiment_number} not found at {exp_dir}")
    
    # Load metadata
    with open(os.path.join(exp_dir, "metadata.json")) as f:
        metadata = json.load(f)
    
    # Load all model results
    model_results = {}
    for file in os.listdir(exp_dir):
        if file.endswith("_results.json") and file != "analysis_results.json":
            model_name = file.replace("_results.json", "")
            with open(os.path.join(exp_dir, file)) as f:
                model_results[model_name] = json.load(f)
    
    return metadata, model_results

def create_networkx_graph(adj_matrix):
    """Convert numpy adjacency matrix to NetworkX DiGraph."""
    G = nx.DiGraph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)  # i -> j
    return G

def count_topological_errors(M, k):
    """
    Count topological sorting errors in a DAG.
    M[i][j] = 1 means edge from i to j (our convention)
    k: nodes in topological order
    Returns: percentage of correct ancestral relations (D_top metric)
    """
    index_map = {node: idx for idx, node in enumerate(k)}
    sum_edges = 0
    errors = 0
    
    for idx_i, i in enumerate(k):
        for idx_j, j in enumerate(k):
            if M[i][j] != 0:  # Edge from i to j in our convention
                sum_edges += 1
                if idx_i > idx_j:  # i appears after j but i -> j
                    errors += 1
    
    if sum_edges == 0:
        return 1.0
    return (sum_edges - errors) / sum_edges

def calculate_d_top_score(adj_matrix):
    """Calculate D_top score for adjacency matrix."""
    try:
        G = create_networkx_graph(adj_matrix)
        if len(G.nodes()) == 0:
            return 1.0  # Empty graph is perfectly ordered
        
        topo_order = list(nx.topological_sort(G))
        return count_topological_errors(adj_matrix, topo_order)
    except nx.NetworkXError:
        # Graph has cycles, return 0.0 for cyclic graphs
        return 0.0

def create_violin_plots(results_dict, experiment_number):
    """Create violin plots for each metric across all models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Prepare data for plotting and count trials per model
    plot_data = []
    model_trial_counts = {}
    
    for model_name, model_results in results_dict["models"].items():
        model_display = model_name.replace("_", " ").title()
        trial_count = model_results["valid_trials"]
        model_trial_counts[model_display] = trial_count
        
        # Skeleton metrics
        for precision in model_results["skeleton_metrics"]["precision"]:
            plot_data.append({
                "Model": model_display,
                "Metric": "Skeleton Precision",
                "Value": precision
            })
        for recall in model_results["skeleton_metrics"]["recall"]:
            plot_data.append({
                "Model": model_display,
                "Metric": "Skeleton Recall", 
                "Value": recall
            })
        for f1 in model_results["skeleton_metrics"]["f1"]:
            plot_data.append({
                "Model": model_display,
                "Metric": "Skeleton F1",
                "Value": f1
            })
        # Topological quality
        for d_top in model_results["topological_quality"]["d_top_scores"]:
            plot_data.append({
                "Model": model_display,
                "Metric": "D_top (Topological Quality)",
                "Value": d_top
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create trial count string for title
    trial_info = ", ".join([f"{model}: {count}" for model, count in model_trial_counts.items()])
    
    # Create separate violin plot for each metric
    metrics = df["Metric"].unique()
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        metric_df = df[df["Metric"] == metric]
        
        sns.violinplot(data=metric_df, x="Model", y="Value", 
                      palette="Set2", inner="points")
        
        plt.title(f"{metric} - Experiment {experiment_number}", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric, fontsize=12)
        plt.ylim(0, 1)
        
        # Add trial count legend
        legend_labels = [f"{model} ({count} trials)" for model, count in model_trial_counts.items()]
        plt.legend(labels=legend_labels, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        filename = f"exp_{experiment_number}_{metric.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    # Create combined plot with all metrics
    plt.figure(figsize=(14, 8))
    
    sns.violinplot(data=df, x="Model", y="Value", hue="Metric",
                  palette="Set2", inner="points", split=False)
    
    plt.title(f"All Metrics Comparison - Experiment {experiment_number}", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    
    # Get existing legend and add trial count info
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add trial count as text annotation
    trial_text = "\n".join([f"{model}: {count} trials" for model, count in model_trial_counts.items()])
    plt.text(1.05, 0.5, trial_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.tight_layout()
    
    # Save combined plot
    filename = f"exp_{experiment_number}_all_metrics.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot: {filepath}")

def print_summary_table(results_dict):
    """Print summary table with mean metrics per model."""
    print("\n" + "="*80)
    print(f"LLM Guess Analysis Summary - Experiment {results_dict['experiment_number']}")
    print("="*80)
    print(f"{'Model':<25} {'Valid Trials':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'D_top':<10}")
    print("-"*80)
    
    for model_name, model_results in results_dict["models"].items():
        model_display = model_name.replace("_", " ").title()
        valid_trials = model_results["valid_trials"]
        mean_precision = model_results["skeleton_metrics"]["mean_precision"]
        mean_recall = model_results["skeleton_metrics"]["mean_recall"]
        mean_f1 = model_results["skeleton_metrics"]["mean_f1"]
        mean_d_top = model_results["topological_quality"]["mean_d_top"]
        
        print(f"{model_display:<25} {valid_trials:<12} {mean_precision:<10.3f} {mean_recall:<10.3f} {mean_f1:<10.3f} {mean_d_top:<10.3f}")
    
    print("-"*80)
    print(f"Best Skeleton F1: {results_dict['summary']['best_skeleton_f1']}")
    print(f"Best Topological: {results_dict['summary']['best_topological']}")

def analyze_experiment(experiment_number):
    """Run the complete analysis for an experiment."""
    print(f"Starting analysis for experiment {experiment_number}...")
    
    # Load ground truth and experiment results
    ground_truth, variable_names = load_ground_truth(DATASET_NAME)
    print(f"Loaded ground truth for {DATASET_NAME} with {len(variable_names)} variables")
    
    metadata, model_results = load_experiment_results(experiment_number)
    print(f"Loaded results for {len(model_results)} models")
    
    # Calculate baseline metrics
    print("Calculating baseline (True DAG) metrics...")
    # baseline_skeleton = evaluate_skeleton_recovery(ground_truth, ground_truth)
    ground_truth_skeleton = np.maximum(ground_truth, ground_truth.T)
    baseline_skeleton = evaluate_skeleton_recovery(ground_truth_skeleton, ground_truth_skeleton)
    baseline_d_top = calculate_d_top_score(ground_truth)
    
    # Analyze each model
    print("Analyzing analysis...")
    results_dict = {
        "experiment_number": experiment_number,
        "dataset_name": DATASET_NAME,
        "baseline": {
            "skeleton_precision": baseline_skeleton["precision"],
            "skeleton_recall": baseline_skeleton["recall"],
            "skeleton_f1": baseline_skeleton["f1"],
            "d_top_score": baseline_d_top
        },
        "models": {},
        "summary": {}
    }
    
    for model_name, model_data in model_results.items():
        print(f"  Processing {model_name}...")
        
        # Extract valid trials
        valid_matrices = []
        for trial in model_data["trials"]:
            if not trial["success"]:
                continue
            if trial["adjacency_matrix"] is None:
                continue
            
            matrix = np.array(trial["adjacency_matrix"])
            if not is_valid_dag(matrix.tolist()):
                continue
            
            valid_matrices.append(matrix)
        
        if not valid_matrices:
            print(f"    No valid DAGs found for {model_name}")
            continue
        
        # Calculate skeleton metrics for each valid matrix
        skeleton_metrics = []
        d_top_scores = []
        
        for matrix in valid_matrices:
            predicted_skeleton = np.maximum(matrix, matrix.T)
            skeleton = evaluate_skeleton_recovery(ground_truth_skeleton, predicted_skeleton)
            skeleton_metrics.append(skeleton)
            d_top_scores.append(calculate_d_top_score(matrix))
        
        # Store results
        results_dict["models"][model_name] = {
            "valid_trials": len(valid_matrices),
            "total_trials": len(model_data["trials"]),
            "skeleton_metrics": {
                "precision": [s["precision"] for s in skeleton_metrics],
                "recall": [s["recall"] for s in skeleton_metrics],
                "f1": [s["f1"] for s in skeleton_metrics],
                "mean_precision": np.mean([s["precision"] for s in skeleton_metrics]),
                "mean_recall": np.mean([s["recall"] for s in skeleton_metrics]),
                "mean_f1": np.mean([s["f1"] for s in skeleton_metrics])
            },
            "topological_quality": {
                "d_top_scores": d_top_scores,
                "mean_d_top": np.mean(d_top_scores)
            }
        }
    
    # Calculate summary statistics
    if results_dict["models"]:
        best_f1_model = max(results_dict["models"].items(), 
                           key=lambda x: x[1]["skeleton_metrics"]["mean_f1"])
        best_d_top_model = max(results_dict["models"].items(), 
                              key=lambda x: x[1]["topological_quality"]["mean_d_top"])
        
        results_dict["summary"] = {
            "best_skeleton_f1": f"{best_f1_model[0]} ({best_f1_model[1]['skeleton_metrics']['mean_f1']:.3f})",
            "best_topological": f"{best_d_top_model[0]} ({best_d_top_model[1]['topological_quality']['mean_d_top']:.3f})"
        }
    
    # Save analysis results
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(project_root, "guess-llms-results", f"guess_experiment_{experiment_number}")
    with open(os.path.join(exp_dir, "analysis_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Create plots and print summary
    if results_dict["models"]:
        create_violin_plots(results_dict, experiment_number)
        print_summary_table(results_dict)
    else:
        print("No valid results found for any model.")
    
    return results_dict

if __name__ == "__main__":
    analyze_experiment(EXPERIMENT_NUMBER)
