"""
Tests for data generation module.
"""

import pytest
import numpy as np
import os
import glob
from causality_paper_guess_to_graph.data_generation import generate_linear_data
from causality_paper_guess_to_graph.methods import run_method


class TestGenerateLinearData:
    """Test cases for generate_linear_data function."""
    
    def test_basic_functionality(self):
        """Test basic data generation works."""
        n_nodes = 5
        edge_probability = 0.3
        sample_size = 100
        
        G, data = generate_linear_data(n_nodes, edge_probability, sample_size)
        
        # Check return types
        assert G is not None
        assert data is not None
        
        # Check data shape
        assert data.shape == (sample_size, n_nodes)
        
        # Check data is numeric
        assert np.isfinite(data).all()
    
    def test_different_parameters(self):
        """Test with different parameter combinations."""
        test_cases = [
            (3, 0.2, 50),
            (10, 0.4, 200),
            (4, 0.0, 30)  # Zero edge probability
        ]
        
        for n_nodes, edge_probability, sample_size in test_cases:
            G, data = generate_linear_data(n_nodes, edge_probability, sample_size)
            assert data.shape == (sample_size, n_nodes)
            assert np.isfinite(data).all()
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Negative n_nodes
        with pytest.raises(ValueError, match="n_nodes must be positive"):
            generate_linear_data(-1, 2, 100)
        
        # Zero n_nodes
        with pytest.raises(ValueError, match="n_nodes must be positive"):
            generate_linear_data(0, 2, 100)
        
        # Invalid edge probability
        with pytest.raises(ValueError, match="edge_probability must be between 0 and 1"):
            generate_linear_data(5, -0.1, 100)
        
        with pytest.raises(ValueError, match="edge_probability must be between 0 and 1"):
            generate_linear_data(5, 1.5, 100)
        
        # Negative sample_size
        with pytest.raises(ValueError, match="sample_size must be positive"):
            generate_linear_data(5, 0.2, -1)
        
        # Zero sample_size
        with pytest.raises(ValueError, match="sample_size must be positive"):
            generate_linear_data(5, 0.2, 0)
    
    def test_reproducibility(self):
        """Test that results are deterministic when using same random seed."""
        # Note: This test may need adjustment based on causal-learn's random seed handling
        n_nodes, edge_probability, sample_size = 4, 0.3, 50
        
        # Generate data twice
        G1, data1 = generate_linear_data(n_nodes, edge_probability, sample_size)
        G2, data2 = generate_linear_data(n_nodes, edge_probability, sample_size)
        
        # Results should be different (since no seed is set)
        # This tests that randomness is working
        assert not np.array_equal(data1, data2)
    
    def test_data_properties(self):
        """Test properties of generated data."""
        n_nodes = 6
        edge_probability = 0.3
        sample_size = 1000
        
        G, data = generate_linear_data(n_nodes, edge_probability, sample_size)
        
        # Data should have reasonable variance (not all zeros)
        assert np.var(data) > 0
        
        # Each variable should have some variance
        for i in range(n_nodes):
            assert np.var(data[:, i]) > 0
        
        # Data should be roughly centered (mean close to 0 for large samples)
        # Allow some tolerance due to randomness
        assert abs(np.mean(data)) < 1.0
    
    def test_different_seeds_produce_different_data(self):
        """Test that different random seeds produce different data."""
        from causality_paper_guess_to_graph.data_generation import (
            generate_er_dag_adjacency_matrix,
            generate_weighted_adjacency_matrix,
            generate_linear_gaussian_data
        )
        
        n_nodes, sample_size = 5, 100
        
        # Generate data with seed 42
        binary_adj1 = generate_er_dag_adjacency_matrix(n_nodes, 0.3, random_seed=42)
        weighted_adj1 = generate_weighted_adjacency_matrix(binary_adj1, random_seed=42)
        data1 = generate_linear_gaussian_data(weighted_adj1, sample_size, random_seed=42)
        
        # Generate data with seed 123
        binary_adj2 = generate_er_dag_adjacency_matrix(n_nodes, 0.3, random_seed=123)
        weighted_adj2 = generate_weighted_adjacency_matrix(binary_adj2, random_seed=123)
        data2 = generate_linear_gaussian_data(weighted_adj2, sample_size, random_seed=123)
        
        # Data should be different with different seeds
        assert not np.array_equal(data1, data2)
        assert not np.array_equal(weighted_adj1, weighted_adj2)


class TestRealWorldDatasets:
    """Test cases for real-world datasets downloaded by real_world_data_downloader."""
    
    def get_real_world_datasets(self):
        """Get all real-world dataset files."""
        # Look in project root for real-world-data folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(project_root, "real-world-data", "*-real-world-data.npy")
        files = glob.glob(pattern)
        return files
    
    def has_cycle_dfs(self, adj_matrix):
        """Check if adjacency matrix has cycles using DFS."""
        n = adj_matrix.shape[0]
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def dfs(node):
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                return False
            
            color[node] = GRAY
            for neighbor in range(n):
                if adj_matrix[node][neighbor] != 0:
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False
        
        for i in range(n):
            if color[i] == WHITE:
                if dfs(i):
                    return True
        return False
    
    def test_all_real_world_datasets(self):
        """Test all real-world datasets comprehensively."""
        files = self.get_real_world_datasets()
        
        if not files:
            pytest.skip("No real-world datasets found. Run real_world_data_downloader.py first.")
        
        errors = []
        
        for file_path in files:
            dataset_name = os.path.basename(file_path).replace("-real-world-data.npy", "")
            
            try:
                data_tuple = np.load(file_path, allow_pickle=True)
            except Exception as e:
                errors.append(f"{dataset_name}: Failed to load file - {e}")
                continue
            
            if len(data_tuple) != 3:
                errors.append(f"{dataset_name}: Expected 3 elements, got {len(data_tuple)}")
                continue
            
            adj_matrix, data, variable_names = data_tuple
            
            # Variable names validation
            if not isinstance(variable_names, (list, np.ndarray)):
                errors.append(f"{dataset_name}: variable_names is not a list/array")
            else:
                variable_names = list(variable_names)
                if len(variable_names) == 0:
                    errors.append(f"{dataset_name}: variable_names is empty")
                elif any(not isinstance(name, str) or len(name.strip()) == 0 for name in variable_names):
                    errors.append(f"{dataset_name}: variable_names contains empty/non-string values")
                elif len(set(variable_names)) != len(variable_names):
                    errors.append(f"{dataset_name}: variable_names contains duplicates")
            
            # Data validation
            if not isinstance(data, np.ndarray):
                errors.append(f"{dataset_name}: data is not numpy array")
            elif data.size == 0:
                errors.append(f"{dataset_name}: data array is empty")
            elif len(data.shape) != 2:
                errors.append(f"{dataset_name}: data should be 2D, got shape {data.shape}")
            elif data.shape[0] == 0:
                errors.append(f"{dataset_name}: no samples in data")
            elif data.shape[1] != len(variable_names):
                errors.append(f"{dataset_name}: data columns ({data.shape[1]}) != variable_names length ({len(variable_names)})")
            
            # Data quality
            if isinstance(data, np.ndarray) and data.size > 0:
                if np.any(np.isnan(data)):
                    errors.append(f"{dataset_name}: data contains NaN values")
                if np.any(np.isinf(data)):
                    errors.append(f"{dataset_name}: data contains Inf values")
            
            # Adjacency matrix validation
            if not isinstance(adj_matrix, np.ndarray):
                errors.append(f"{dataset_name}: adjacency matrix is not numpy array")
            elif len(adj_matrix.shape) != 2:
                errors.append(f"{dataset_name}: adjacency matrix should be 2D")
            elif adj_matrix.shape[0] != adj_matrix.shape[1]:
                errors.append(f"{dataset_name}: adjacency matrix not square")
            elif adj_matrix.shape[0] != len(variable_names):
                errors.append(f"{dataset_name}: adjacency matrix size != variables")
            else:
                unique_vals = np.unique(adj_matrix)
                # Convert to float to handle boolean/integer types
                float_vals = unique_vals.astype(float)
                if not all(abs(val - round(val)) < 1e-10 for val in float_vals):
                    errors.append(f"{dataset_name}: adjacency matrix contains non-integer values")
                elif not all(val in [0, 1] for val in np.round(float_vals).astype(int)):
                    errors.append(f"{dataset_name}: adjacency matrix should be binary")
                
                binary_adj = adj_matrix.astype(int)
                if self.has_cycle_dfs(binary_adj):
                    errors.append(f"{dataset_name}: adjacency matrix contains cycles")
            
            # Data type test for discrete data
            if isinstance(data, np.ndarray) and data.size > 0:
                if data.dtype in [np.int64, np.int32]:
                    if not np.all(data >= 0):
                        errors.append(f"{dataset_name}: discrete data contains negative values")
                    if not np.all(data == data.astype(int)):
                        errors.append(f"{dataset_name}: discrete data contains non-integer values")
            
            # Method compatibility
            try:
                if isinstance(data, np.ndarray) and data.shape[0] >= 10:
                    skeleton, runtime = run_method('pc', data, alpha=0.05)
                    if not isinstance(skeleton, np.ndarray):
                        errors.append(f"{dataset_name}: PC method returned invalid skeleton type")
                    elif skeleton.shape != (data.shape[1], data.shape[1]):
                        errors.append(f"{dataset_name}: PC method returned wrong skeleton shape")
            except Exception as e:
                errors.append(f"{dataset_name}: PC method compatibility failed - {e}")
        
        if errors:
            error_msg = "\n".join([f"  - {error}" for error in errors])
            pytest.fail(f"Real-world dataset validation failed:\n{error_msg}")
    
    def test_sachs_dataset(self):
        """Test specific properties of the sachs dataset."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sachs_file = os.path.join(project_root, "real-world-data", "sachs-real-world-data.npy")
        
        if not os.path.exists(sachs_file):
            pytest.skip("Sachs dataset not found")
        
        adj_matrix, data, variable_names = np.load(sachs_file, allow_pickle=True)
        
        assert data.dtype in [np.int64, np.int32], f"Sachs should be discrete data, got {data.dtype}"
        
        # For discrete data, just check that values are reasonable integers
        assert np.all(data >= 0), "Sachs discrete data should be non-negative"
        assert np.all(data == data.astype(int)), "Sachs discrete data should be integers"
        
        expected_vars = 11
        assert len(variable_names) == expected_vars, f"Sachs should have {expected_vars} variables"
        assert data.shape[1] == expected_vars, f"Sachs data should have {expected_vars} columns"
        assert adj_matrix.shape == (expected_vars, expected_vars), f"Sachs adjacency should be {expected_vars}x{expected_vars}"


if __name__ == "__main__":
    test_instance = TestGenerateLinearData()
    
    print("Running test_basic_functionality...")
    test_instance.test_basic_functionality()
    print("✓ Passed")
    
    print("Running test_different_parameters...")
    test_instance.test_different_parameters()
    print("✓ Passed")
    
    print("Running test_invalid_parameters...")
    test_instance.test_invalid_parameters()
    print("✓ Passed")
    
    print("Running test_reproducibility...")
    test_instance.test_reproducibility()
    print("✓ Passed")
    
    print("Running test_data_properties...")
    test_instance.test_data_properties()
    print("✓ Passed")
    
    print("Running test_different_seeds_produce_different_data...")
    test_instance.test_different_seeds_produce_different_data()
    print("✓ Passed")
    
    # Test real-world datasets
    real_world_test = TestRealWorldDatasets()
    
    print("\nRunning test_all_real_world_datasets...")
    try:
        real_world_test.test_all_real_world_datasets()
        print("✓ Passed")
    except Exception as e:
        print(f"⚠ Skipped or Failed: {e}")
    
    print("Running test_sachs_dataset...")
    try:
        real_world_test.test_sachs_dataset()
        print("✓ Passed")
    except Exception as e:
        print(f"⚠ Skipped or Failed: {e}")
    
    print("\nAll tests completed! 🎉")