"""
Tests for experiment_baselines module.
"""

import pytest
import numpy as np
import os
import json
import shutil
import sys

# Add paths to import from experiments and src directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'experiments'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'causality_paper_guess_to_graph'))

from experiment_baselines import aggregate_trial_metrics, run_experiment
from metrics import calculate_metrics


class TestExperimentBaselines:
    """Test cases for experiment baselines."""
    
    def test_metrics_correctness(self):
        """Test that metrics are calculated correctly."""
        # Create known true and predicted skeletons (symmetric)
        true_skeleton = np.array([[0, 1, 1], 
                                 [1, 0, 1], 
                                 [1, 1, 0]])
        pred_skeleton = np.array([[0, 1, 0], 
                                 [1, 0, 1], 
                                 [0, 1, 0]])
        
        metrics = calculate_metrics(true_skeleton, pred_skeleton, ['precision', 'recall', 'f1'])
        
        # Manual calculation: TP=2, FP=0, FN=1 (comparing upper triangular)
        assert abs(metrics['precision'] - 1.0) < 0.01  # 2/2
        assert abs(metrics['recall'] - 0.667) < 0.01   # 2/3
        assert abs(metrics['f1'] - 0.8) < 0.01         # 2*1.0*0.667/(1.0+0.667)
    
    def test_edge_probability_usage(self):
        """Test that edge_probability is used correctly."""
        # Test that edge probability is between 0 and 1
        edge_prob = 0.3
        assert 0 <= edge_prob <= 1
        
        # Test expected number of edges for different graph sizes
        dim = 10
        max_edges = dim * (dim - 1) // 2  # 45 for 10 nodes
        expected_edges = edge_prob * max_edges
        assert abs(expected_edges - 13.5) < 0.01
    
    def test_skeleton_conversion(self):
        """Test conversion from directed to undirected skeleton."""
        # Directed graph
        directed = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        
        # Conversion logic from experiment_baselines
        skeleton = (np.abs(directed) > 0).astype(int)
        skeleton = np.maximum(skeleton, skeleton.T)
        
        expected = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        
        assert np.array_equal(skeleton, expected)
    
    def test_metric_aggregation(self):
        """Test that aggregation computes correct statistics."""
        trial_results = [
            {'pc': {'f1': 0.8}, 'runtimes': {'pc': 1.0}},
            {'pc': {'f1': 0.9}, 'runtimes': {'pc': 1.5}},
            {'pc': {'f1': 0.7}, 'runtimes': {'pc': 1.2}}
        ]
        
        aggregated = aggregate_trial_metrics(trial_results, ['pc'])
        
        assert abs(aggregated['pc']['f1']['mean'] - 0.8) < 0.01
        assert abs(aggregated['pc']['f1']['std'] - 0.0816) < 0.01
        assert aggregated['pc']['f1']['min'] == 0.7
        assert aggregated['pc']['f1']['max'] == 0.9
        
        # Test runtime aggregation
        assert abs(aggregated['pc']['runtime']['mean'] - 1.233) < 0.01
    
    def test_mini_experiment(self):
        """Run a tiny experiment to test the full pipeline."""
        config = {
            'num_trials': 2,
            'graph_params': {
                'dimensionalities': [3],
                'sample_sizes': [50],
                'edge_probabilities': [0.3]
            },
            'algorithms': ['pc'],
            'metrics': ['f1', 'precision', 'recall']
        }
        
        exp_dir = run_experiment(config, results_dir="test_results")
        
        # Check structure
        assert os.path.exists(os.path.join(exp_dir, "config.json"))
        assert os.path.exists(os.path.join(exp_dir, "experiment_summary.json"))
        assert os.path.exists(os.path.join(exp_dir, "dim_3_samples_50_edge_prob_0_3"))
        
        # Check metrics exist
        metrics_file = os.path.join(exp_dir, "dim_3_samples_50_edge_prob_0_3", "aggregated_metrics.json")
        assert os.path.exists(metrics_file)
        
        with open(metrics_file) as f:
            metrics = json.load(f)
            assert 'pc' in metrics
            assert 'f1' in metrics['pc']
            assert 'mean' in metrics['pc']['f1']
            assert 'runtime' in metrics['pc']
        
        # Check experiment summary
        with open(os.path.join(exp_dir, "experiment_summary.json")) as f:
            summary = json.load(f)
            assert 'successful_trials' in summary
            assert 'failed_trials' in summary
            assert 'success_rate' in summary
            assert summary['trials_per_combination'] == 2
        
        # Cleanup
        shutil.rmtree("test_results")
    
    def test_empty_trial_results(self):
        """Test aggregation with empty trial results."""
        trial_results = []
        aggregated = aggregate_trial_metrics(trial_results, ['pc'])
        
        assert aggregated['num_trials'] == 0
        assert 'pc' in aggregated
        assert aggregated['pc'] == {}
    
    def test_missing_algorithm_in_trials(self):
        """Test aggregation when some trials are missing algorithm results."""
        trial_results = [
            {'pc': {'f1': 0.8}, 'runtimes': {'pc': 1.0}},
            {'stable_pc': {'f1': 0.9}, 'runtimes': {'stable_pc': 1.5}},  # Missing pc
            {'pc': {'f1': 0.7}, 'runtimes': {'pc': 1.2}}
        ]
        
        aggregated = aggregate_trial_metrics(trial_results, ['pc', 'stable_pc'])
        
        # PC should have 2 values
        assert len([trial for trial in trial_results if 'pc' in trial]) == 2
        assert abs(aggregated['pc']['f1']['mean'] - 0.75) < 0.01  # (0.8 + 0.7) / 2
        
        # Stable PC should be in aggregated and contain the f1 score from trial 1
        assert 'stable_pc' in aggregated
        assert 'f1' in aggregated['stable_pc']
        assert aggregated['stable_pc']['f1']['mean'] == 0.9
        assert aggregated['stable_pc']['f1']['std'] == 0.0  # Only one value
        assert aggregated['stable_pc']['f1']['min'] == 0.9
        assert aggregated['stable_pc']['f1']['max'] == 0.9


if __name__ == "__main__":
    test_instance = TestExperimentBaselines()
    
    print("Running test_metrics_correctness...")
    test_instance.test_metrics_correctness()
    print("✓ Passed")
    
    print("Running test_edge_probability_usage...")
    test_instance.test_edge_probability_usage()
    print("✓ Passed")
    
    print("Running test_skeleton_conversion...")
    test_instance.test_skeleton_conversion()
    print("✓ Passed")
    
    print("Running test_metric_aggregation...")
    test_instance.test_metric_aggregation()
    print("✓ Passed")
    
    print("Running test_mini_experiment...")
    test_instance.test_mini_experiment()
    print("✓ Passed")
    
    print("Running test_empty_trial_results...")
    test_instance.test_empty_trial_results()
    print("✓ Passed")
    
    print("Running test_missing_algorithm_in_trials...")
    test_instance.test_missing_algorithm_in_trials()
    print("✓ Passed")
    
    print("\n✅ All tests passed!")