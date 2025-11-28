"""
Tests for generate_guess_llm module and guess-llm-results validation.
"""

import pytest
import numpy as np
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add guess-code to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'guess-code'))
from generate_guess_llm import (
    load_dataset, load_prompt, create_prompt, parse_adjacency_matrix, 
    find_next_experiment_number, run_experiment
)


class TestGenerateGuessLLM:
    """Test cases for generate_guess_llm module."""
    
    def test_load_dataset(self):
        """Test dataset loading functionality"""
        # Test successful loading of sachs dataset
        variable_names = load_dataset("sachs")
        assert isinstance(variable_names, list)
        assert len(variable_names) == 11  # Sachs has 11 variables
        assert all(isinstance(name, str) for name in variable_names)
        
        # Test error handling for non-existent dataset
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset")
    
    def test_load_prompt(self):
        """Test prompt loading from prompts.json"""
        # Test loading existing prompt
        prompt_template = load_prompt("basic_prompt_v_1")
        assert isinstance(prompt_template, str)
        assert "{var_names_str}" in prompt_template
        assert "{matrix_size}" in prompt_template
        assert "{example_matrix}" in prompt_template
        
        # Test error handling for non-existent prompt
        with pytest.raises(KeyError):
            load_prompt("nonexistent_prompt")
    
    def test_create_prompt(self):
        """Test prompt generation"""
        # Test with sachs variable names
        variable_names = ['Erk', 'Akt', 'PKA', 'Mek', 'Jnk']
        prompt = create_prompt(variable_names)
        
        # Verify prompt contains all variable names
        for name in variable_names:
            assert name in prompt
        
        # Verify matrix dimensions match variable count
        assert f"{len(variable_names)}x{len(variable_names)}" in prompt
        
        # Verify prompt format includes all required instructions
        assert "directed adjacency matrix" in prompt
        assert "Rows represent source nodes" in prompt
        assert "Columns represent target nodes" in prompt
        assert "1 indicates a direct regulatory relationship" in prompt
        assert "0 indicates no direct relationship" in prompt
        assert "Return ONLY the matrix" in prompt
        
        # Verify example matrix has correct dimensions
        matrix_size = len(variable_names)
        expected_zeros = ",".join(["0"] * matrix_size)
        assert expected_zeros in prompt
    
    def test_parse_adjacency_matrix(self):
        """Test matrix parsing from LLM responses"""
        # Test valid matrix parsing
        valid_response = "Here is the matrix: [[0,1,0],[1,0,1],[0,1,0]]"
        success, result = parse_adjacency_matrix(valid_response, 3)
        assert success == True
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        
        # Test handling of extra text before/after matrix
        messy_response = "I'll generate the matrix for you. [[0,1],[1,0]] This is a simple 2x2 matrix."
        success, result = parse_adjacency_matrix(messy_response, 2)
        assert success == True
        assert result.shape == (2, 2)
        
        # Test rejection of wrong dimensions
        wrong_size = "[[0,1,0],[1,0,1]]"  # 2x3 instead of 3x3
        success, error = parse_adjacency_matrix(wrong_size, 3)
        assert success == False
        assert "Matrix has shape (2, 3), expected (3, 3)" in error
        
        # Test rejection of non-binary values
        non_binary = "[[0,1,2],[1,0,1],[0,1,0]]"
        success, error = parse_adjacency_matrix(non_binary, 3)
        assert success == False
        assert "Matrix contains values other than 0 and 1" in error
        
        # Test handling of malformed brackets
        malformed = "[[0,1,0],[1,0,1"  # Missing closing bracket
        success, error = parse_adjacency_matrix(malformed, 3)
        assert success == False
        assert "Unmatched brackets" in error
        
        # Test empty response handling
        empty_response = "I cannot generate that matrix."
        success, error = parse_adjacency_matrix(empty_response, 3)
        assert success == False
        assert "No matrix pattern found" in error
        
        # Test nested list structure validation
        nested_valid = "[[0,1],[1,0]]"
        success, result = parse_adjacency_matrix(nested_valid, 2)
        assert success == True
        assert result.tolist() == [[0,1],[1,0]]
    
    def test_find_next_experiment_number(self):
        """Test experiment numbering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the project root to use temp directory
            with patch('generate_guess_llm.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = temp_dir
                
                results_dir = os.path.join(temp_dir, "guess-llms-results")
                
                # Test when no experiments exist (should return 1)
                exp_num = find_next_experiment_number()
                assert exp_num == 1
                
                # Create some experiment directories
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(os.path.join(results_dir, "guess_experiment_1"))
                os.makedirs(os.path.join(results_dir, "guess_experiment_3"))
                
                # Test with existing experiments (should increment from max)
                exp_num = find_next_experiment_number()
                assert exp_num == 4
                
                # Test with gaps in numbering
                os.makedirs(os.path.join(results_dir, "guess_experiment_10"))
                exp_num = find_next_experiment_number()
                assert exp_num == 11
                
                # Test with non-standard folder names mixed in
                os.makedirs(os.path.join(results_dir, "other_folder"))
                os.makedirs(os.path.join(results_dir, "guess_experiment_invalid"))
                exp_num = find_next_experiment_number()
                assert exp_num == 11  # Should ignore invalid folders


class TestDataValidation:
    """Test cases for validating saved experiment results."""
    
    def test_dag_acyclicity(self):
        """Test that matrices represent acyclic graphs"""
        # Test known DAG is accepted
        dag_matrix = [[0,1,0],[0,0,1],[0,0,0]]
        assert is_valid_dag(dag_matrix) == True
        
        # Test known cyclic matrix is rejected
        cyclic_matrix = [[0,1,0],[0,0,1],[1,0,0]]
        assert is_valid_dag(cyclic_matrix) == False
        
        # Test empty graph (all zeros) is valid
        empty_matrix = [[0,0,0],[0,0,0],[0,0,0]]
        assert is_valid_dag(empty_matrix) == True
        
        # Test self-loop is rejected
        self_loop = [[1,0,0],[0,0,0],[0,0,0]]
        assert is_valid_dag(self_loop) == False
        
        # Test non-binary values are rejected
        non_binary = [[0,2,0],[0,0,1],[0,0,0]]
        assert is_valid_dag(non_binary) == False
    
    def test_adjacency_matrix_validity(self):
        """Test that saved adjacency matrices are valid DAGs"""
        # Create mock experiment results
        mock_results = {
            "model_id": "test-model",
            "model_name": "Test Model",
            "trials": [
                {
                    "trial_number": 1,
                    "success": True,
                    "adjacency_matrix": [[0,1,0],[0,0,1],[0,0,0]]
                },
                {
                    "trial_number": 2,
                    "success": True,
                    "adjacency_matrix": [[0,0,1],[0,0,0],[0,0,0]]
                },
                {
                    "trial_number": 3,
                    "success": False,
                    "adjacency_matrix": None
                }
            ]
        }
        
        mock_metadata = {
            "variable_names": ["A", "B", "C"],
            "dataset_name": "test"
        }
        
        # Validate successful trials
        for trial in mock_results["trials"]:
            if trial["success"] and trial["adjacency_matrix"]:
                matrix = trial["adjacency_matrix"]
                
                # Verify matrix dimensions
                assert len(matrix) == len(mock_metadata["variable_names"])
                assert all(len(row) == len(mock_metadata["variable_names"]) for row in matrix)
                
                # Verify is valid DAG
                assert is_valid_dag(matrix)
    
    def test_metadata_consistency(self):
        """Test metadata file validity"""
        mock_metadata = {
            "dataset_name": "sachs",
            "prompt_format": "basic_prompt_v_1",
            "num_trials": 5,
            "variable_names": ["Erk", "Akt", "PKA"],
            "timestamp": "2024-01-15T10:30:00.123456",
            "experiment_number": 1
        }
        
        # Verify all required fields present
        required_fields = ["dataset_name", "prompt_format", "num_trials", 
                          "variable_names", "timestamp", "experiment_number"]
        for field in required_fields:
            assert field in mock_metadata
        
        # Verify variable names is list
        assert isinstance(mock_metadata["variable_names"], list)
        
        # Verify experiment number is positive integer
        assert isinstance(mock_metadata["experiment_number"], int)
        assert mock_metadata["experiment_number"] > 0
        
        # Verify num_trials is positive integer
        assert isinstance(mock_metadata["num_trials"], int)
        assert mock_metadata["num_trials"] > 0


class TestIntegration:
    """Integration tests for end-to-end functionality."""
    
    @patch('generate_guess_llm.call_bedrock_with_retry')
    def test_end_to_end_single_model(self, mock_bedrock):
        """Test complete flow with mock LLM"""
        # Mock successful response
        mock_bedrock.return_value = ("[[0,1,0],[0,0,1],[0,0,0]]", 2.5, 1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('generate_guess_llm.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = temp_dir
                
                # Mock CLAUDE_MODELS to have only one model
                with patch('generate_guess_llm.CLAUDE_MODELS', {
                    "test_model": {
                        "model_id": "test-model-id",
                        "model_name": "Test Model"
                    }
                }):
                    with patch('generate_guess_llm.NUM_TRIALS', 2):
                        # Run experiment
                        run_experiment()
                        
                        # Verify folder structure created
                        results_dir = os.path.join(temp_dir, "guess-llms-results")
                        assert os.path.exists(results_dir)
                        
                        exp_dir = os.path.join(results_dir, "guess_experiment_1")
                        assert os.path.exists(exp_dir)
                        
                        # Verify files saved correctly
                        assert os.path.exists(os.path.join(exp_dir, "metadata.json"))
                        assert os.path.exists(os.path.join(exp_dir, "test_model_results.json"))
                        
                        # Verify metadata content
                        with open(os.path.join(exp_dir, "metadata.json")) as f:
                            metadata = json.load(f)
                        assert metadata["dataset_name"] == "sachs"
                        assert metadata["num_trials"] == 2
                        
                        # Verify results content
                        with open(os.path.join(exp_dir, "test_model_results.json")) as f:
                            results = json.load(f)
                        assert len(results["trials"]) == 2
                        assert all(trial["success"] for trial in results["trials"])
    
    @patch('generate_guess_llm.call_bedrock_with_retry')
    def test_retry_logic(self, mock_bedrock):
        """Test retry behavior on parsing failures"""
        # Mock responses: first two fail parsing, third succeeds
        mock_bedrock.side_effect = [
            ("Invalid response", 1.0, 1),  # First attempt fails
            ("Still invalid", 1.0, 1),     # Second attempt fails
            ("[[0,1,0],[0,0,1],[0,0,0]]", 2.5, 1)  # Third attempt succeeds
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('generate_guess_llm.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = temp_dir
                
                with patch('generate_guess_llm.CLAUDE_MODELS', {
                    "test_model": {
                        "model_id": "test-model-id", 
                        "model_name": "Test Model"
                    }
                }):
                    with patch('generate_guess_llm.NUM_TRIALS', 1):
                        # Capture print output to verify retry messages
                        with patch('builtins.print') as mock_print:
                            run_experiment()
                            
                            # Verify retry messages were printed
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            retry_messages = [call for call in print_calls if "failed to generate a valid DAG" in call]
                            assert len(retry_messages) >= 2  # Should have retry messages


class TestRobustness:
    """Test error handling and robustness."""
    
    def test_error_handling(self):
        """Test error handling throughout pipeline"""
        # Test dataset loading with invalid path
        with pytest.raises(FileNotFoundError):
            load_dataset("invalid_dataset")
        
        # Test parsing with completely invalid input
        success, error = parse_adjacency_matrix("", 3)
        assert success == False
        assert "No matrix pattern found" in error
        
        # Test parsing with malformed JSON-like structure
        success, error = parse_adjacency_matrix("[[0,1,],[,0,1]]", 2)
        assert success == False
        assert "Parsing error" in error


# Helper Functions
def get_dag_validation_error(matrix):
    """Get specific reason why matrix is not a valid DAG"""
    n = len(matrix)
    
    # Check square
    if any(len(row) != n for row in matrix):
        row_lengths = [len(row) for row in matrix]
        return f"Matrix is not square: row lengths are {row_lengths}"
    
    # Check binary
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val not in [0, 1]:
                return f"Non-binary value {val} at position ({i}, {j})"
    
    # Check no self-loops
    for i in range(n):
        if matrix[i][i] != 0:
            return f"Self-loop detected at node {i} (diagonal element is {matrix[i][i]})"
    
    # Check acyclic using DFS
    def has_cycle_dfs():
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def visit(node, path):
            if color[node] == GRAY:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                return True, cycle
            if color[node] == BLACK:
                return False, []
            
            color[node] = GRAY
            for neighbor in range(n):
                if matrix[node][neighbor] == 1:
                    has_cycle, cycle = visit(neighbor, path + [node])
                    if has_cycle:
                        return True, cycle
            color[node] = BLACK
            return False, []
        
        for i in range(n):
            if color[i] == WHITE:
                has_cycle, cycle = visit(i, [])
                if has_cycle:
                    return True, cycle
        return False, []
    
    has_cycle, cycle = has_cycle_dfs()
    if has_cycle:
        return f"Cycle detected: {' -> '.join(map(str, cycle))}"
    
    return "Valid DAG"

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


def load_experiment_results(exp_dir):
    """Load all results from an experiment directory"""
    results = {}
    for file in os.listdir(exp_dir):
        if file.endswith('_results.json'):
            with open(os.path.join(exp_dir, file)) as f:
                model_name = file.replace('_results.json', '')
                results[model_name] = json.load(f)
    return results


def validate_saved_results():
    """Validate all saved experiment results with detailed error reporting"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "guess-llms-results")
    
    if not os.path.exists(results_dir):
        return True, []  # No results to validate
    
    validation_errors = []
    
    for exp_folder in os.listdir(results_dir):
        if not exp_folder.startswith("guess_experiment_"):
            continue
        
        exp_path = os.path.join(results_dir, exp_folder)
        
        # Load metadata
        metadata_path = os.path.join(exp_path, "metadata.json")
        if not os.path.exists(metadata_path):
            validation_errors.append(f"Experiment {exp_folder}: Missing metadata.json")
            continue
            
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception as e:
            validation_errors.append(f"Experiment {exp_folder}: Failed to load metadata.json - {e}")
            continue
        
        # Load and validate each model's results
        for file in os.listdir(exp_path):
            if not file.endswith("_results.json"):
                continue
            
            model_name = file.replace("_results.json", "")
            
            try:
                with open(os.path.join(exp_path, file)) as f:
                    model_results = json.load(f)
            except Exception as e:
                validation_errors.append(f"Experiment {exp_folder}, Model {model_name}: Failed to load results - {e}")
                continue
            
            for trial_idx, trial in enumerate(model_results["trials"]):
                if trial["success"] and trial["adjacency_matrix"]:
                    matrix = trial["adjacency_matrix"]
                    trial_num = trial.get("trial_number", trial_idx + 1)
                    
                    try:
                        # Validate matrix dimensions
                        if len(matrix) != len(metadata["variable_names"]):
                            validation_errors.append(
                                f"Experiment {exp_folder}, Model {model_name}, Trial {trial_num}: "
                                f"Matrix has {len(matrix)} rows, expected {len(metadata['variable_names'])}"
                            )
                            continue
                        
                        if not all(len(row) == len(metadata["variable_names"]) for row in matrix):
                            row_lengths = [len(row) for row in matrix]
                            validation_errors.append(
                                f"Experiment {exp_folder}, Model {model_name}, Trial {trial_num}: "
                                f"Matrix rows have inconsistent lengths: {row_lengths}, expected {len(metadata['variable_names'])}"
                            )
                            continue
                        
                        # Validate is valid DAG
                        if not is_valid_dag(matrix):
                            # Get specific reason why it's not a valid DAG
                            dag_error = get_dag_validation_error(matrix)
                            validation_errors.append(
                                f"Experiment {exp_folder}, Model {model_name}, Trial {trial_num}: "
                                f"Invalid DAG - {dag_error}"
                            )
                            
                    except Exception as e:
                        validation_errors.append(
                            f"Experiment {exp_folder}, Model {model_name}, Trial {trial_num}: "
                            f"Validation error - {e}"
                        )
    
    return len(validation_errors) == 0, validation_errors


class TestSavedResults:
    """Test validation of actual saved results."""
    
    def test_validate_all_saved_experiments(self):
        """Validate all experiments in guess-llms-results directory"""
        # This test validates any existing saved results
        is_valid, errors = validate_saved_results()
        
        if not is_valid:
            error_msg = "\n".join([f"  - {error}" for error in errors])
            pytest.fail(f"Validation failed with {len(errors)} error(s):\n{error_msg}")
        
        assert is_valid == True
    
    def test_specific_experiment_structure(self):
        """Test structure of a specific experiment if it exists"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "guess-llms-results")
        
        if not os.path.exists(results_dir):
            pytest.skip("No guess-llms-results directory found")
        
        exp_folders = [d for d in os.listdir(results_dir) if d.startswith("guess_experiment_")]
        if not exp_folders:
            pytest.skip("No experiment folders found")
        
        # Test the first experiment found
        exp_path = os.path.join(results_dir, exp_folders[0])
        
        # Check metadata exists
        metadata_path = os.path.join(exp_path, "metadata.json")
        assert os.path.exists(metadata_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check required metadata fields
        required_fields = ["dataset_name", "prompt_format", "num_trials", 
                          "variable_names", "timestamp", "experiment_number"]
        for field in required_fields:
            assert field in metadata
        
        # Check that result files exist
        result_files = [f for f in os.listdir(exp_path) if f.endswith("_results.json")]
        assert len(result_files) > 0
        
        # Validate each result file
        for result_file in result_files:
            with open(os.path.join(exp_path, result_file)) as f:
                results = json.load(f)
            
            assert "model_id" in results
            assert "model_name" in results
            assert "trials" in results
            assert len(results["trials"]) == metadata["num_trials"]


# Mock Fixtures
@pytest.fixture
def mock_bedrock_response():
    """Mock successful Bedrock API response"""
    return ("[[0,1,0,0,0,0,0,0,0,0,0]," +
            "[0,0,1,0,0,0,0,0,0,0,0]," +
            "[0,0,0,1,0,0,0,0,0,0,0]," +
            "[0,0,0,0,1,0,0,0,0,0,0]," +
            "[0,0,0,0,0,1,0,0,0,0,0]," +
            "[0,0,0,0,0,0,1,0,0,0,0]," +
            "[0,0,0,0,0,0,0,1,0,0,0]," +
            "[0,0,0,0,0,0,0,0,1,0,0]," +
            "[0,0,0,0,0,0,0,0,0,1,0]," +
            "[0,0,0,0,0,0,0,0,0,0,1]," +
            "[0,0,0,0,0,0,0,0,0,0,0]]", 2.5, 1)


@pytest.fixture  
def mock_failed_bedrock_response():
    """Mock failed/malformed response"""
    return ("Sorry, I cannot generate that matrix", 1.0, 1)


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_instance = TestGenerateGuessLLM()
    validation_instance = TestDataValidation()
    integration_instance = TestIntegration()
    robustness_instance = TestRobustness()
    saved_results_instance = TestSavedResults()
    
    failed_tests = []
    
    # Core functionality tests
    print("Running test_load_dataset...")
    try:
        test_instance.test_load_dataset()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_load_dataset")
    
    print("Running test_load_prompt...")
    try:
        test_instance.test_load_prompt()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_load_prompt")
    
    print("Running test_create_prompt...")
    try:
        test_instance.test_create_prompt()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_create_prompt")
    
    print("Running test_parse_adjacency_matrix...")
    try:
        test_instance.test_parse_adjacency_matrix()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_parse_adjacency_matrix")
    
    print("Running test_find_next_experiment_number...")
    try:
        test_instance.test_find_next_experiment_number()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_find_next_experiment_number")
    
    # Data validation tests
    print("Running test_dag_acyclicity...")
    try:
        validation_instance.test_dag_acyclicity()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_dag_acyclicity")
    
    print("Running test_adjacency_matrix_validity...")
    try:
        validation_instance.test_adjacency_matrix_validity()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_adjacency_matrix_validity")
    
    print("Running test_metadata_consistency...")
    try:
        validation_instance.test_metadata_consistency()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_metadata_consistency")
    
    # Robustness tests
    print("Running test_error_handling...")
    try:
        robustness_instance.test_error_handling()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_error_handling")
    
    # Saved results validation
    print("Running test_validate_all_saved_experiments...")
    try:
        saved_results_instance.test_validate_all_saved_experiments()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_validate_all_saved_experiments")
        
        # Also run direct validation to show detailed errors
        print("\nDetailed validation results:")
        is_valid, errors = validate_saved_results()
        if not is_valid:
            for error in errors:
                print(f"  - {error}")
        else:
            print("  No validation errors found.")
    
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
    else:
        print("\nAll tests passed!")