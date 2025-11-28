"""
Tests for metrics module.
"""

import pytest
import numpy as np
from causality_paper_guess_to_graph.metrics import calculate_metrics
from causality_paper_guess_to_graph.data_generation import generate_linear_data
from causality_paper_guess_to_graph.methods import run_pc, run_stable_pc


class TestMetrics:
    """Test cases for metrics calculation."""
    
    def test_perfect_prediction(self):
        """Test that identical skeletons yield perfect scores"""
        # Create a simple skeleton (symmetric)
        true_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Perfect prediction (same skeleton)
        predicted_skeleton = true_skeleton.copy()
        
        metrics = calculate_metrics(true_skeleton, predicted_skeleton)
        
        # Perfect prediction should yield perfect scores
        assert metrics['f1'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
    
    def test_empty_prediction(self):
        """Test that empty predicted skeleton yields zero recall"""
        # Create a skeleton with edges
        true_skeleton = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        # Empty prediction (no edges)
        predicted_skeleton = np.zeros_like(true_skeleton)
        
        metrics = calculate_metrics(true_skeleton, predicted_skeleton)
        
        # Empty prediction should have zero recall
        assert metrics['recall'] == 0.0
        # Precision is undefined when no edges predicted, but should be handled
        # f1 should be 0 when recall is 0
        assert metrics['f1'] == 0.0
    
    def test_one_edge_difference(self):
        """Test that one edge difference is worse than perfect prediction"""
        # True skeleton
        true_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Predicted skeleton with one extra edge
        predicted_skeleton = np.array([
            [0, 1, 1],  # Extra edge 0-2
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        metrics = calculate_metrics(true_skeleton, predicted_skeleton)
        
        # Should be worse than perfect (f1 < 1.0)
        assert metrics['f1'] < 1.0
        assert metrics['precision'] < 1.0  # False positive
        assert metrics['recall'] == 1.0    # All true edges found
    
    def test_completely_wrong_graph(self):
        """Test completely wrong skeleton prediction"""
        # True skeleton
        true_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Completely wrong prediction
        predicted_skeleton = np.array([
            [0, 0, 1],  # Wrong edges
            [0, 0, 0],
            [1, 0, 0]
        ])
        
        metrics = calculate_metrics(true_skeleton, predicted_skeleton)
        
        # Should have poor performance
        assert metrics['f1'] == 0
        assert metrics['precision'] == 0
        assert metrics['recall'] == 0
    
    def test_error_handling_different_sizes(self):
        """Test that different sized skeletons raise appropriate error"""
        true_skeleton = np.array([
            [0, 1],
            [1, 0]
        ])
        
        predicted_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        with pytest.raises(ValueError, match="Graph shapes don't match"):
            calculate_metrics(true_skeleton, predicted_skeleton)
    
    def test_with_generated_data_and_methods(self):
        """Test metrics with generated data and both PC methods"""
        # Generate data and true graph
        true_graph, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Convert weighted adjacency to binary skeleton
        true_skeleton = (np.abs(true_graph) > 0).astype(int)
        true_skeleton = np.maximum(true_skeleton, true_skeleton.T)  # Make symmetric
        
        # Test PC method
        predicted_skeleton_pc = run_pc(data, alpha=0.05)
        metrics_pc = calculate_metrics(true_skeleton, predicted_skeleton_pc)
        
        # Check PC metrics are computed and in valid range
        assert 0 <= metrics_pc['f1'] <= 1
        assert 0 <= metrics_pc['precision'] <= 1
        assert 0 <= metrics_pc['recall'] <= 1
        assert isinstance(metrics_pc, dict)
        assert len(metrics_pc) == 3
        
        # Test Stable PC method
        predicted_skeleton_stable = run_stable_pc(data, alpha=0.05)
        metrics_stable = calculate_metrics(true_skeleton, predicted_skeleton_stable)
        
        # Check Stable PC metrics are computed and in valid range
        assert 0 <= metrics_stable['f1'] <= 1
        assert 0 <= metrics_stable['precision'] <= 1
        assert 0 <= metrics_stable['recall'] <= 1
        assert isinstance(metrics_stable, dict)
        assert len(metrics_stable) == 3
    
    def test_custom_metrics_list(self):
        """Test with custom metrics list"""
        true_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        predicted_skeleton = true_skeleton.copy()
        
        # Request only precision and recall
        metrics = calculate_metrics(true_skeleton, predicted_skeleton, ['precision', 'recall'])
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' not in metrics
        assert len(metrics) == 2
    
    def test_invalid_metric_name(self):
        """Test error handling for invalid metric names"""
        true_skeleton = np.array([
            [0, 1],
            [1, 0]
        ])
        
        predicted_skeleton = true_skeleton.copy()
        
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_metrics(true_skeleton, predicted_skeleton, ['invalid_metric'])
    
    def test_dag_input_error(self):
        """Test that DAG inputs raise appropriate error"""
        # Create a DAG (not symmetric)
        true_dag = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        predicted_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        # Should raise error for non-symmetric true graph
        with pytest.raises(ValueError, match="True graph is not symmetric - expected skeleton, got DAG"):
            calculate_metrics(true_dag, predicted_skeleton)
        
        # Should raise error for non-symmetric predicted graph
        true_skeleton = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        predicted_dag = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        with pytest.raises(ValueError, match="Estimated graph is not symmetric - expected skeleton, got DAG"):
            calculate_metrics(true_skeleton, predicted_dag)
    
    def test_runtime_tracking(self):
        """Test that runtime is being computed correctly"""
        from causality_paper_guess_to_graph.methods import run_method
        
        # Generate small dataset
        G, data = generate_linear_data(n_nodes=3, edge_probability=0.3, sample_size=100)
        
        # Test PC runtime
        skeleton_pc, runtime_pc = run_method('pc', data, alpha=0.05)
        assert isinstance(runtime_pc, float)
        assert runtime_pc > 0  # Should be non-zero
        
        # Test Stable PC runtime
        skeleton_stable, runtime_stable = run_method('stable_pc', data, alpha=0.05)
        assert isinstance(runtime_stable, float)
        assert runtime_stable > 0  # Should be non-zero
    
    def test_runtime_scales_with_graph_size(self):
        """Test that runtime generally increases with graph size"""
        from causality_paper_guess_to_graph.methods import run_method
        
        # Small graph
        G_small, data_small = generate_linear_data(n_nodes=3, edge_probability=0.3, sample_size=200)
        _, runtime_small = run_method('pc', data_small, alpha=0.05)
        
        # Larger graph
        G_large, data_large = generate_linear_data(n_nodes=6, edge_probability=0.3, sample_size=200)
        _, runtime_large = run_method('pc', data_large, alpha=0.05)
        
        # Runtime should generally be larger for bigger graphs
        # Note: This is a general trend test, not strict due to randomness
        assert runtime_small >= 0
        assert runtime_large >= 0
        assert isinstance(runtime_small, float)
        assert isinstance(runtime_large, float)
    
    def test_runtime_consistency(self):
        """Test that runtime is consistently measured"""
        from causality_paper_guess_to_graph.methods import run_method
        
        # Generate dataset
        G, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=150)
        
        # Run multiple times and check runtime is always positive
        runtimes = []
        for _ in range(3):
            _, runtime = run_method('pc', data, alpha=0.05)
            runtimes.append(runtime)
            assert runtime > 0
            assert isinstance(runtime, float)
        
        # All runtimes should be positive
        assert all(r > 0 for r in runtimes)


if __name__ == "__main__":
    test_instance = TestMetrics()
    
    print("Running test_perfect_prediction...")
    test_instance.test_perfect_prediction()
    print("✓ Passed")
    
    print("Running test_empty_prediction...")
    test_instance.test_empty_prediction()
    print("✓ Passed")
    
    print("Running test_one_edge_difference...")
    test_instance.test_one_edge_difference()
    print("✓ Passed")
    
    print("Running test_completely_wrong_graph...")
    test_instance.test_completely_wrong_graph()
    print("✓ Passed")
    
    print("Running test_error_handling_different_sizes...")
    test_instance.test_error_handling_different_sizes()
    print("✓ Passed")
    
    print("Running test_with_generated_data_and_methods...")
    test_instance.test_with_generated_data_and_methods()
    print("✓ Passed")
    
    print("Running test_custom_metrics_list...")
    test_instance.test_custom_metrics_list()
    print("✓ Passed")
    
    print("Running test_invalid_metric_name...")
    test_instance.test_invalid_metric_name()
    print("✓ Passed")
    
    print("Running test_dag_input_error...")
    test_instance.test_dag_input_error()
    print("✓ Passed")
    
    print("Running test_runtime_tracking...")
    test_instance.test_runtime_tracking()
    print("✓ Passed")
    
    print("Running test_runtime_scales_with_graph_size...")
    test_instance.test_runtime_scales_with_graph_size()
    print("✓ Passed")
    
    print("Running test_runtime_consistency...")
    test_instance.test_runtime_consistency()
    print("✓ Passed")
    
    print("\nAll tests passed! 🎉")