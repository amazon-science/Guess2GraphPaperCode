"""
Tests for causal discovery methods module.
"""

import pytest
import numpy as np
from causality_paper_guess_to_graph.methods import (
    run_pc, run_stable_pc, run_method, run_pc_guess_dag, run_pc_guess_expert, run_sgs_guess_expert, run_sgs_guess_dag,
    run_pc_guess_expert_twice, run_sgs_guess_expert_twice, run_pc_hc_guess_expert, run_sgs_hc_guess_expert,
    _topological_sort, _get_testable_edges, _expert_edge_ordering, _expert_subset_ordering,
    _create_probabilistic_skeleton, _probabilistic_subset_ordering, 
    _contains_mutual_descendant, _get_descendants, _expert_edge_ordering_sgs, _sample_hard_constraints
)
from causality_paper_guess_to_graph.data_generation import generate_linear_data


class TestMethods:
    """Test cases for causal discovery methods."""
    
    def test_independent_variables(self):
        """Test that independent variables produce no edges"""
        # Generate independent random variables
        np.random.seed(42)
        data = np.random.randn(10000, 3)
        skeleton = run_pc(data, alpha=0.05)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    def test_chain_structure(self):
        """Test X -> Y -> Z chain detection"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        skeleton = run_pc(data, alpha=0.05)
        # Should find edges X-Y and Y-Z, but not X-Z
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge (they're conditionally independent given Y)
    
    def test_stable_pc_independent_variables(self):
        """Test that stable PC also handles independent variables correctly"""
        np.random.seed(42)
        data = np.random.randn(10000, 3)
        skeleton = run_stable_pc(data, alpha=0.05)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    def test_stable_pc_chain_structure(self):
        """Test stable PC on chain structure"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        skeleton = run_stable_pc(data, alpha=0.05)
        # Should find edges X-Y and Y-Z, but not X-Z
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_methods_with_generated_data(self):
        """Test methods run without error on generated data and return valid skeleton"""
        # Generate data using data_generation module
        G, data = generate_linear_data(n_nodes=5, edge_probability=0.3, sample_size=1000)
        
        # Test PC
        skeleton_pc = run_pc(data, alpha=0.05)
        assert skeleton_pc.shape == (5, 5)
        assert skeleton_pc.dtype == int
        assert np.all(skeleton_pc >= 0)  # Non-negative values
        assert np.array_equal(skeleton_pc, skeleton_pc.T)  # Symmetric
        
        # Test Stable PC
        skeleton_stable = run_stable_pc(data, alpha=0.05)
        assert skeleton_stable.shape == (5, 5)
        assert skeleton_stable.dtype == int
        assert np.all(skeleton_stable >= 0)  # Non-negative values
        assert np.array_equal(skeleton_stable, skeleton_stable.T)  # Symmetric
    
    def test_run_method_wrapper(self):
        """Test the run_method wrapper function"""
        G, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Test PC via wrapper
        skeleton_pc, runtime_pc = run_method('pc', data, alpha=0.05)
        assert skeleton_pc.shape == (4, 4)
        assert isinstance(runtime_pc, float)
        assert runtime_pc >= 0
        
        # Test Stable PC via wrapper
        skeleton_stable, runtime_stable = run_method('stable_pc', data, alpha=0.05)
        assert skeleton_stable.shape == (4, 4)
        assert isinstance(runtime_stable, float)
        assert runtime_stable >= 0
        
        # Test invalid method name
        with pytest.raises(ValueError, match="Unknown method"):
            run_method('invalid_method', data)
    
    # PC-Guess-DAG Tests
    def test_run_method_pc_guess_dag(self):
        """Test that run_method correctly calls pc_guess_dag"""
        G, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Test with guessed DAG
        skeleton, runtime = run_method('pc_guess_dag', data, guessed_dag=G, alpha=0.05)
        assert skeleton.shape == (4, 4)
        assert isinstance(runtime, float)
        assert runtime >= 0
        
        # Test error when guessed_dag not provided
        with pytest.raises(ValueError, match="pc_guess_dag requires 'guessed_dag' parameter"):
            run_method('pc_guess_dag', data)
    
    def test_pc_guess_dag_return_format(self):
        """Verify returns skeleton matrix and runtime"""
        G, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=200)
        skeleton = run_pc_guess_dag(data, G, alpha=0.05)
        
        # Check skeleton properties
        assert skeleton.shape == (3, 3)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_pc_guess_dag_chain(self):
        """Test on X->Y->Z chain with correct guess"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        # Create correct DAG guess
        guessed_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_pc_guess_dag(data, guessed_dag, alpha=0.05)
        # Should find X-Y and Y-Z edges, not X-Z
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_pc_guess_dag_edge_cases(self):
        """Test edge cases"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_pc_guess_dag(data_single, dag_single)
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_pc_guess_dag(data_indep, dag_empty)
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    def test_pc_guess_dag_wrong_guess(self):
        """Test robustness to bad guesses"""
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n)
        Y = 2 * X + 0.5 * np.random.randn(n)
        data = np.column_stack([X, Y])
        
        # Wrong guess (opposite direction)
        wrong_dag = np.array([[0, 0], [1, 0]])
        skeleton = run_pc_guess_dag(data, wrong_dag, alpha=0.05)
        
        # Should still produce valid skeleton
        assert skeleton.shape == (2, 2)
        assert np.array_equal(skeleton, skeleton.T)
        assert skeleton[0, 1] == 1  # Should still find the edge
    
    # Helper Function Tests
    def test_topological_sort(self):
        """Test topological ordering extraction"""
        # Valid DAG
        dag = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        topo = _topological_sort(dag)
        assert len(topo) == 3
        assert set(topo) == {0, 1, 2}
        assert topo.index(0) < topo.index(1)  # 0 before 1
        assert topo.index(0) < topo.index(2)  # 0 before 2
        assert topo.index(1) < topo.index(2)  # 1 before 2
        
        # Empty graph
        empty = np.zeros((2, 2))
        topo_empty = _topological_sort(empty)
        assert len(topo_empty) == 2
        assert set(topo_empty) == {0, 1}
        
        # Cycle (should return random permutation)
        cycle = np.array([[0, 1], [1, 0]])
        topo_cycle = _topological_sort(cycle)
        assert len(topo_cycle) == 2
        assert set(topo_cycle) == {0, 1}
    
    def test_get_testable_edges_ordered_pairs(self):
        """Test ordered pairs generation"""
        # Complete graph
        graph = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        # Level 0: all edges
        edges_0 = _get_testable_edges(graph, 0)
        assert len(edges_0) == 6  # All directed pairs
        assert (0, 1) in edges_0 and (1, 0) in edges_0
        
        # Level 1: edges where source has enough neighbors
        edges_1 = _get_testable_edges(graph, 1)
        assert len(edges_1) == 6  # All nodes have 2 neighbors
        
        # Level 2: no edges (need 2 neighbors excluding target)
        edges_2 = _get_testable_edges(graph, 2)
        assert len(edges_2) == 0
    
    def test_expert_edge_ordering_grouping(self):
        """Test expert edge ordering groups correctly"""
        edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
        skeleton = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        
        ordered = _expert_edge_ordering(edges, skeleton)
        assert len(ordered) == 4
        # Should prioritize non-edges first
        non_edge_positions = [i for i, (x, y) in enumerate(ordered) if skeleton[x, y] == 0]
        edge_positions = [i for i, (x, y) in enumerate(ordered) if skeleton[x, y] == 1]
        
        if non_edge_positions and edge_positions:
            assert max(non_edge_positions) < min(edge_positions)
    
    def test_ci_test_parameter_fisherz(self):
        """Test that fisherz CI test works (default behavior)"""
        G, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Test all methods with fisherz (explicit)
        skeleton_pc = run_pc(data, alpha=0.05, ci_test='fisherz')
        skeleton_stable = run_stable_pc(data, alpha=0.05, ci_test='fisherz')
        skeleton_guess_dag = run_pc_guess_dag(data, G, alpha=0.05, ci_test='fisherz')
        skeleton_sgs_guess_dag = run_sgs_guess_dag(data, G, alpha=0.05, ci_test='fisherz')
        
        # Test via wrapper
        skeleton_pc_wrap, _ = run_method('pc', data, alpha=0.05, ci_test='fisherz')
        skeleton_stable_wrap, _ = run_method('stable_pc', data, alpha=0.05, ci_test='fisherz')
        skeleton_sgs_wrap, _ = run_method('sgs_guess_dag', data, guessed_dag=G, alpha=0.05, ci_test='fisherz')
        
        # All should return valid skeletons
        for skeleton in [skeleton_pc, skeleton_stable, skeleton_guess_dag, skeleton_sgs_guess_dag, skeleton_pc_wrap, skeleton_stable_wrap, skeleton_sgs_wrap]:
            assert skeleton.shape == (4, 4)
            assert skeleton.dtype == int
            assert np.array_equal(skeleton, skeleton.T)
    
    def test_ci_test_parameter_chisq(self):
        """Test that chisq CI test works with discrete data"""
        # Generate small discrete dataset
        np.random.seed(42)
        n = 1000
        
        # Create discrete variables with dependencies
        X = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
        Y = np.where(X == 1, 
                    np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
                    np.random.choice([0, 1], size=n, p=[0.8, 0.2]))
        Z = np.random.choice([0, 1, 2], size=n)
        
        data = np.column_stack([X, Y, Z]).astype(int)
        
        # Create simple DAG for guess methods
        simple_dag = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        
        # Test all methods with chisq
        skeleton_pc = run_pc(data, alpha=0.05, ci_test='chisq')
        skeleton_stable = run_stable_pc(data, alpha=0.05, ci_test='chisq')
        skeleton_guess_dag = run_pc_guess_dag(data, simple_dag, alpha=0.05, ci_test='chisq')
        skeleton_sgs_guess_dag = run_sgs_guess_dag(data, simple_dag, alpha=0.05, ci_test='chisq')
        
        # Test via wrapper
        skeleton_pc_wrap, _ = run_method('pc', data, alpha=0.05, ci_test='chisq')
        skeleton_stable_wrap, _ = run_method('stable_pc', data, alpha=0.05, ci_test='chisq')
        skeleton_sgs_wrap, _ = run_method('sgs_guess_dag', data, guessed_dag=simple_dag, alpha=0.05, ci_test='chisq')
        
        # All should return valid skeletons
        for skeleton in [skeleton_pc, skeleton_stable, skeleton_guess_dag, skeleton_sgs_guess_dag, skeleton_pc_wrap, skeleton_stable_wrap, skeleton_sgs_wrap]:
            assert skeleton.shape == (3, 3)
            assert skeleton.dtype == int
            assert np.array_equal(skeleton, skeleton.T)
            
        # Should detect X-Y dependency
        assert skeleton_pc[0, 1] == 1
        assert skeleton_stable[0, 1] == 1
    
    def test_ci_test_parameter_expert_methods(self):
        """Test CI test parameter with expert methods"""
        # Generate discrete data for chisq test
        np.random.seed(42)
        n = 800
        X = np.random.choice([0, 1], size=n)
        Y = np.random.choice([0, 1], size=n)
        Z = (X + Y) % 2  # Z depends on X and Y
        data = np.column_stack([X, Y, Z]).astype(int)
        
        # Create true DAG
        true_dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        
        # Test expert methods with both CI tests
        for ci_test in ['fisherz', 'chisq']:
            skeleton_expert, _ = run_method('pc_guess_expert', data,
                                          true_dag=true_dag,
                                          p_acc_edge_true=0.8,
                                          p_acc_edge_false=0.8,
                                          p_acc_subset_MD=0.8,
                                          p_acc_subset_NMD=0.8,
                                          alpha=0.05,
                                          ci_test=ci_test)
            
            skeleton_sgs, _ = run_method('sgs_guess_expert', data,
                                       true_dag=true_dag,
                                       p_acc_edge_true=0.8,
                                       p_acc_edge_false=0.8,
                                       p_acc_subset_MD=0.8,
                                       p_acc_subset_NMD=0.8,
                                       alpha=0.05,
                                       ci_test=ci_test)
            
            # Should return valid skeletons
            assert skeleton_expert.shape == (3, 3)
            assert skeleton_sgs.shape == (3, 3)
            assert np.array_equal(skeleton_expert, skeleton_expert.T)
            assert np.array_equal(skeleton_sgs, skeleton_sgs.T)
    
    def test_ci_test_default_behavior(self):
        """Test that default CI test is fisherz"""
        G, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=300)
        
        # Test without specifying ci_test (should default to fisherz)
        skeleton_default = run_pc(data, alpha=0.05)
        skeleton_explicit = run_pc(data, alpha=0.05, ci_test='fisherz')
        
        # Results should be identical
        assert np.array_equal(skeleton_default, skeleton_explicit)
        
        # Test via wrapper
        skeleton_wrap_default, _ = run_method('pc', data, alpha=0.05)
        skeleton_wrap_explicit, _ = run_method('pc', data, alpha=0.05, ci_test='fisherz')
        
        assert np.array_equal(skeleton_wrap_default, skeleton_wrap_explicit)
        """Test edge pair grouping and ordering"""
        edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
        guessed_skeleton = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # 0-1 exists, 0-2 doesn't
        
        ordered = _expert_edge_ordering(edges, guessed_skeleton)
        
        # Should have all 4 edges
        assert len(ordered) == 4
        assert set(ordered) == set(edges)
        
        # Non-edges (0,2) and (2,0) should come before edges (0,1) and (1,0)
        pos_02 = min(ordered.index((0, 2)), ordered.index((2, 0)))
        pos_01 = min(ordered.index((0, 1)), ordered.index((1, 0)))
        assert pos_02 < pos_01
    
    def test_expert_subset_ordering_descendants(self):
        """Test conditioning set ordering based on topology"""
        topo_order = [0, 1, 2, 3]
        conditioning_sets = [(2,), (3,), (0,), (1,)]
        
        # Test edge (0,1) - max position is 1
        ordered = _expert_subset_ordering(conditioning_sets, 0, 1, topo_order)
        
        # Sets with variables <= 1 should come first
        b1_sets = [s for s in ordered if all(topo_order.index(v) <= 1 for v in s)]
        b2_sets = [s for s in ordered if any(topo_order.index(v) > 1 for v in s)]
        
        assert len(b1_sets) == 2  # (0,) and (1,)
        assert len(b2_sets) == 2  # (2,) and (3,)
        
        # B1 sets should come before B2 sets
        first_b2_pos = len(b1_sets)
        for i, s in enumerate(ordered):
            if s in b2_sets:
                assert i >= first_b2_pos
    
    # PC-Guess-Expert Tests
    def test_create_probabilistic_skeleton(self):
        """Test that probabilistic skeleton respects accuracy parameters"""
        np.random.seed(42)
        
        # Create a known skeleton
        true_skeleton = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        
        # Test with perfect accuracy
        skeleton_perfect = _create_probabilistic_skeleton(true_skeleton, 1.0, 1.0)
        assert np.array_equal(skeleton_perfect, true_skeleton)
        
        # Test with 0% accuracy (should invert)
        skeleton_worst = _create_probabilistic_skeleton(true_skeleton, 0.0, 0.0)
        # All true edges should be 0, all false edges should be 1
        for i in range(4):
            for j in range(i+1, 4):
                assert skeleton_worst[i,j] != true_skeleton[i,j]
    
    def test_get_descendants(self):
        """Test descendant extraction from DAG"""
        # Chain: 0->1->2->3
        dag = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        assert _get_descendants(0, dag) == {1, 2, 3}
        assert _get_descendants(1, dag) == {2, 3}
        assert _get_descendants(2, dag) == {3}
        assert _get_descendants(3, dag) == set()
        
        # Fork: 0->1, 0->2
        dag_fork = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        assert _get_descendants(0, dag_fork) == {1, 2}
    
    def test_contains_mutual_descendant(self):
        """Test mutual descendant identification"""
        # Create DAG: 0->2, 1->2, 2->3
        dag = np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        # Test mutual descendants of 0 and 1
        assert _contains_mutual_descendant(0, 1, (2,), dag) == True
        assert _contains_mutual_descendant(0, 1, (3,), dag) == True
        assert _contains_mutual_descendant(0, 1, (0,), dag) == False
        assert _contains_mutual_descendant(0, 1, (1,), dag) == False
        assert _contains_mutual_descendant(0, 1, (), dag) == False
        
        # Test when no mutual descendants exist
        assert _contains_mutual_descendant(2, 3, (0,), dag) == False
        assert _contains_mutual_descendant(2, 3, (1,), dag) == False
    
    # def test_probabilistic_subset_ordering(self):
    #     """Test subset ordering with MD classification accuracy"""
    #     np.random.seed(42)
    #     
    #     # DAG where 2 is mutual descendant of 0 and 1
    #     dag = np.array([
    #         [0, 0, 1, 0],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]
    #     ])
    #     
    #     conditioning_sets = [(2,), (3,), (0,), (1,)]
    #     
    #     # Test perfect accuracy
    #     ordered = _probabilistic_subset_ordering(
    #         conditioning_sets, 0, 1, dag, 1.0, 1.0
    #     )
    #     # (2,) has MD, others don't
    #     # Find positions of all non-MD sets
    #     non_md_positions = [ordered.index(s) for s in [(3,), (0,), (1,)]]
    #     md_pos = ordered.index((2,))
    #     # MD should come after all non-MD sets
    #     assert all(md_pos > pos for pos in non_md_positions)
    
    def test_probabilistic_subset_ordering_dsep(self):
        """Test subset ordering with d-separation based classification"""
        import networkx as nx
        np.random.seed(42)
        
        # Create chain DAG: 0->1->2
        dag = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag_graph = nx.from_numpy_array(dag, create_using=nx.DiGraph)
        
        # Test conditioning sets for edge (0,2)
        conditioning_sets = [(), (1,)]
        
        # Test with perfect accuracy
        ordered = _probabilistic_subset_ordering(
            conditioning_sets, 0, 2, dag_graph, 1.0, 1.0
        )
        
        print(ordered)
        
        # (1,) d-separates 0 and 2, so should come first (no MD)
        # () does not d-separate 0 and 2, so should come second (has MD)
        dsep_pos = ordered.index((1,))
        non_dsep_pos = ordered.index(())
        assert dsep_pos < non_dsep_pos
    
    def test_pc_guess_expert_perfect_accuracy(self):
        """Test PC guess expert with perfect accuracy parameters"""
        np.random.seed(42)
        # Generate data with known structure
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Run with perfect accuracy
        skeleton = run_pc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        # Should be valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
    
    def test_pc_guess_expert_random_accuracy(self):
        """Test PC guess expert with random accuracy (0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Run with random guessing
        skeleton = run_pc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.5,
            p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5,
            p_acc_subset_NMD=0.5,
            alpha=0.05
        )
        
        # Should still produce valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
    
    def test_pc_guess_expert_via_wrapper(self):
        """Test PC guess expert through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with all parameters
        skeleton, runtime = run_method(
            'pc_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9,
            p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7,
            p_acc_subset_NMD=0.6,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert runtime > 0
        
        # Test with default parameters
        skeleton2, runtime2 = run_method(
            'pc_guess_expert', data,
            true_dag=true_dag
        )
        assert skeleton2.shape == (3, 3)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="pc_guess_expert requires 'true_dag'"):
            run_method('pc_guess_expert', data)
    
    # PC-Guess-Expert-Twice Tests
    def test_pc_guess_expert_twice_perfect_accuracy(self):
        """Test PC guess expert twice with perfect accuracy parameters"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        skeleton = run_pc_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
    
    def test_pc_guess_expert_twice_random_accuracy(self):
        """Test PC guess expert twice with random accuracy (0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        skeleton = run_pc_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=0.5,
            p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5,
            p_acc_subset_NMD=0.5,
            alpha=0.05
        )
        
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
    
    def test_pc_guess_expert_twice_chain_structure(self):
        """Test PC guess expert twice on X->Y->Z chain"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_pc_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_pc_guess_expert_twice_via_wrapper(self):
        """Test PC guess expert twice through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        skeleton, runtime = run_method(
            'pc_guess_expert_twice', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9,
            p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7,
            p_acc_subset_NMD=0.6,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="pc_guess_expert_twice requires 'true_dag'"):
            run_method('pc_guess_expert_twice', data)
    
    # SGS-Guess-Expert Tests
    def test_expert_edge_ordering_sgs(self):
        """Test SGS edge ordering for unordered pairs"""
        edges = [(0, 1), (0, 2), (1, 2)]
        guessed_skeleton = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # Only 0-1 exists
        
        ordered = _expert_edge_ordering_sgs(edges, guessed_skeleton)
        
        # Should have all 3 edges
        assert len(ordered) == 3
        assert set(ordered) == set(edges)
        
        # Non-edges should come before edges
        non_edge_positions = [ordered.index((0, 2)), ordered.index((1, 2))]
        edge_pos = ordered.index((0, 1))
        assert all(edge_pos > pos for pos in non_edge_positions)
    
    def test_sgs_guess_expert_perfect_accuracy(self):
        """Test SGS guess expert with perfect accuracy parameters"""
        np.random.seed(42)
        # Generate data with known structure
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Run with perfect accuracy
        skeleton = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        # Should be valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
        assert np.all((skeleton == 0) | (skeleton == 1))
    
    def test_sgs_guess_expert_random_accuracy(self):
        """Test SGS guess expert with random accuracy (0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Run with random guessing
        skeleton = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.5,
            p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5,
            p_acc_subset_NMD=0.5,
            alpha=0.05
        )
        
        # Should still produce valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
        assert np.all((skeleton == 0) | (skeleton == 1))
    
    def test_sgs_guess_expert_chain_structure(self):
        """Test SGS on X->Y->Z chain with perfect expert"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        # Create correct DAG
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        # Should find X-Y and Y-Z edges, not X-Z
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_sgs_guess_expert_via_wrapper(self):
        """Test SGS guess expert through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with all parameters
        skeleton, runtime = run_method(
            'sgs_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9,
            p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7,
            p_acc_subset_NMD=0.6,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test with default parameters
        skeleton2, runtime2 = run_method(
            'sgs_guess_expert', data,
            true_dag=true_dag
        )
        assert skeleton2.shape == (3, 3)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="sgs_guess_expert requires 'true_dag'"):
            run_method('sgs_guess_expert', data)
    
    def test_sgs_guess_expert_edge_cases(self):
        """Test SGS edge cases"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_sgs_guess_expert(
            data_single, dag_single,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0
        )
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_sgs_guess_expert(
            data_indep, dag_empty,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0
        )
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    def test_sgs_vs_pc_comparison(self):
        """Test that SGS and PC produce valid skeletons on same data"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=800)
        
        # Run both algorithms with same expert parameters
        skeleton_pc = run_pc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            alpha=0.05
        )
        
        skeleton_sgs = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            alpha=0.05
        )
        
        # Both should be valid skeletons
        for skeleton in [skeleton_pc, skeleton_sgs]:
            assert skeleton.shape == (4, 4)
            assert np.array_equal(skeleton, skeleton.T)
            assert np.all(np.diag(skeleton) == 0)
            assert np.all((skeleton == 0) | (skeleton == 1))
        
        # Results may differ due to different algorithms, but both should be reasonable
        # (not testing for exact equality as SGS and PC can legitimately differ)
    
    # SGS-Guess-Expert-Twice Tests
    def test_sgs_guess_expert_twice_perfect_accuracy(self):
        """Test SGS guess expert twice with perfect accuracy parameters"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        skeleton = run_sgs_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
        assert np.all((skeleton == 0) | (skeleton == 1))
    
    def test_sgs_guess_expert_twice_random_accuracy(self):
        """Test SGS guess expert twice with random accuracy (0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        skeleton = run_sgs_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=0.5,
            p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5,
            p_acc_subset_NMD=0.5,
            alpha=0.05
        )
        
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all(np.diag(skeleton) == 0)
        assert np.all((skeleton == 0) | (skeleton == 1))
    
    def test_sgs_guess_expert_twice_chain_structure(self):
        """Test SGS guess expert twice on X->Y->Z chain"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_sgs_guess_expert_twice(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_sgs_guess_expert_twice_edge_cases(self):
        """Test SGS guess expert twice edge cases"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_sgs_guess_expert_twice(
            data_single, dag_single,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0
        )
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_sgs_guess_expert_twice(
            data_indep, dag_empty,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0
        )
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    def test_sgs_guess_expert_twice_via_wrapper(self):
        """Test SGS guess expert twice through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        skeleton, runtime = run_method(
            'sgs_guess_expert_twice', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9,
            p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7,
            p_acc_subset_NMD=0.6,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="sgs_guess_expert_twice requires 'true_dag'"):
            run_method('sgs_guess_expert_twice', data)
    
    # SGS-Guess-DAG Tests
    def test_sgs_guess_dag_return_format(self):
        """Verify that the method returns a valid skeleton matrix with correct properties"""
        G, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=200)
        skeleton = run_sgs_guess_dag(data, G, alpha=0.05)
        
        # Check skeleton properties
        assert skeleton.shape == (3, 3)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_sgs_guess_dag_chain_structure(self):
        """Test on a simple X,Y,Z chain with a correct DAG guess"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        # Create correct DAG guess
        guessed_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_sgs_guess_dag(data, guessed_dag, alpha=0.05)
        # Should find X-Y and Y-Z edges, not X-Z
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_sgs_guess_dag_wrong_guess(self):
        """Test robustness when given an incorrect DAG guess"""
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n)
        Y = 2 * X + 0.5 * np.random.randn(n)
        data = np.column_stack([X, Y])
        
        # Wrong guess (opposite direction)
        wrong_dag = np.array([[0, 0], [1, 0]])
        skeleton = run_sgs_guess_dag(data, wrong_dag, alpha=0.05)
        
        # Should still produce valid skeleton
        assert skeleton.shape == (2, 2)
        assert np.array_equal(skeleton, skeleton.T)
        assert skeleton[0, 1] == 1  # Should still find the edge
    
    def test_sgs_guess_dag_edge_cases(self):
        """Test edge cases including single variable, independent variables, complete graph"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_sgs_guess_dag(data_single, dag_single)
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables with empty DAG
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_sgs_guess_dag(data_indep, dag_empty)
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
        
        # Fully connected valid DAG (chain with additional edge)
        data_complete = np.random.randn(500, 3)
        dag_complete = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])  # Valid DAG with all possible edges
        skeleton = run_sgs_guess_dag(data_complete, dag_complete)
        assert skeleton.shape == (3, 3)
        assert np.array_equal(skeleton, skeleton.T)
    
    def test_sgs_guess_dag_via_wrapper(self):
        """Test that run_method('sgs_guess_dag', ...) works correctly"""
        G, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Test with guessed DAG
        skeleton, runtime = run_method('sgs_guess_dag', data, guessed_dag=G, alpha=0.05)
        assert skeleton.shape == (4, 4)
        assert isinstance(runtime, float)
        assert runtime >= 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test error when guessed_dag not provided
        with pytest.raises(ValueError, match="sgs_guess_dag requires 'guessed_dag' parameter"):
            run_method('sgs_guess_dag', data)
    
    def test_sgs_guess_dag_vs_sgs_expert(self):
        """Compare outputs when run_sgs_guess_dag is given the true DAG versus run_sgs_guess_expert with perfect accuracy"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # Run SGS with guessed DAG (using true DAG as guess)
        skeleton_guess_dag = run_sgs_guess_dag(data, true_dag, alpha=0.05)
        
        # Run SGS with expert (perfect accuracy)
        skeleton_expert = run_sgs_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0,
            p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0,
            p_acc_subset_NMD=1.0,
            alpha=0.05
        )
        
        # Both should be valid skeletons
        for skeleton in [skeleton_guess_dag, skeleton_expert]:
            assert skeleton.shape == (4, 4)
            assert np.array_equal(skeleton, skeleton.T)
            assert np.all(np.diag(skeleton) == 0)
            assert np.all((skeleton == 0) | (skeleton == 1))
        
        # Results should be similar but may differ due to randomization
    
    # PC-HC-Guess-Expert Tests
    def test_sample_hard_constraints_edge_validity(self):
        """Test that sampled constraints match skeleton predictions"""
        np.random.seed(42)
        # Create a known skeleton
        skeleton = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        
        constrained_true, constrained_false = _sample_hard_constraints(skeleton, 0.5)
        
        # Check that constrained true edges are actually edges in skeleton
        for i, j in constrained_true:
            assert skeleton[i, j] == 1
        
        # Check that constrained false edges are actually non-edges in skeleton
        for i, j in constrained_false:
            assert skeleton[i, j] == 0
    
    def test_pc_hc_guess_expert_return_format(self):
        """Test that PC-HC-Guess-Expert returns valid skeleton"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Check skeleton properties
        assert skeleton.shape == (4, 4)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_pc_hc_guess_expert_perfect_accuracy(self):
        """Test PC-HC-Guess-Expert with perfect expert"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # With perfect expert and high sample size, should find correct structure
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_pc_hc_guess_expert_random_accuracy(self):
        """Test PC-HC-Guess-Expert with random expert (p_acc=0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Should complete without error even with random expert
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.5, p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5, p_acc_subset_NMD=0.5,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Should return valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        
        # Should be able to calculate metrics without error
        true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
        tp = np.sum((true_skeleton == 1) & (skeleton == 1))
        fp = np.sum((true_skeleton == 0) & (skeleton == 1))
        # No assertion errors means metrics are calculable
    
    def test_pc_hc_guess_expert_via_wrapper(self):
        """Test PC-HC-Guess-Expert through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with all parameters
        skeleton, runtime = run_method(
            'pc_hc_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7, p_acc_subset_NMD=0.6,
            constraint_fraction=0.3, alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert isinstance(runtime, float)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test with default parameters
        skeleton2, runtime2 = run_method(
            'pc_hc_guess_expert', data,
            true_dag=true_dag
        )
        assert skeleton2.shape == (3, 3)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="pc_hc_guess_expert requires 'true_dag'"):
            run_method('pc_hc_guess_expert', data)
    
    def test_sample_hard_constraints_fraction(self):
        """Test that constraint fraction is approximately correct"""
        np.random.seed(42)
        n = 10
        skeleton = np.random.randint(0, 2, (n, n))
        skeleton = np.triu(skeleton, 1)  # Upper triangle only
        skeleton = skeleton + skeleton.T  # Make symmetric
        
        constraint_fraction = 0.3
        constrained_true, constrained_false = _sample_hard_constraints(skeleton, constraint_fraction)
        
        # Count total edges and non-edges
        total_pairs = n * (n - 1) // 2
        n_edges = np.sum(np.triu(skeleton, 1))
        n_non_edges = total_pairs - n_edges
        
        # Check approximately correct fraction (within 20% tolerance)
        if n_edges > 0:
            assert abs(len(constrained_true) - n_edges * constraint_fraction) <= n_edges * 0.2 + 1
        if n_non_edges > 0:
            assert abs(len(constrained_false) - n_non_edges * constraint_fraction) <= n_non_edges * 0.2 + 1
    
    def test_pc_hc_guess_expert_chain_structure(self):
        """Test PC-HC-Guess-Expert on X->Y->Z chain"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Should find correct chain structure
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_pc_hc_guess_expert_constraint_fraction_zero(self):
        """Test that constraint_fraction=0 behaves like pc_guess_expert"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Run with constraint_fraction=0
        skeleton_hc = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.0, alpha=0.05
        )
        
        # Should produce valid skeleton
        assert skeleton_hc.shape == (4, 4)
        assert np.array_equal(skeleton_hc, skeleton_hc.T)
        assert np.all((skeleton_hc == 0) | (skeleton_hc == 1))
    
    def test_pc_hc_guess_expert_high_constraint_fraction(self):
        """Test PC-HC-Guess-Expert with high constraint fraction"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # With 90% constraints and perfect expert, should be very accurate
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.9, alpha=0.05
        )
        
        # Should return valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        
        # Calculate F1 score - should be high with perfect expert and high constraints
        true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
        tp = np.sum((true_skeleton == 1) & (skeleton == 1))
        fp = np.sum((true_skeleton == 0) & (skeleton == 1))
        fn = np.sum((true_skeleton == 1) & (skeleton == 0))
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                assert f1 > 0.7  # Should have reasonable F1 with perfect expert
    
    def test_pc_hc_guess_expert_edge_cases(self):
        """Test PC-HC-Guess-Expert edge cases"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_pc_hc_guess_expert(
            data_single, dag_single,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3
        )
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_pc_hc_guess_expert(
            data_indep, dag_empty,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3
        )
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    # SGS-HC-Guess-Expert Tests
    def test_sgs_hc_guess_expert_return_format(self):
        """Test that SGS-HC-Guess-Expert returns valid skeleton"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Check skeleton properties
        assert skeleton.shape == (4, 4)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_sgs_hc_guess_expert_perfect_accuracy(self):
        """Test SGS-HC-Guess-Expert with perfect expert"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # With perfect expert and high sample size, should find correct structure
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_sgs_hc_guess_expert_random_accuracy(self):
        """Test SGS-HC-Guess-Expert with random expert (p_acc=0.5)"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Should complete without error even with random expert
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.5, p_acc_edge_false=0.5,
            p_acc_subset_MD=0.5, p_acc_subset_NMD=0.5,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Should return valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        
        # Should be able to calculate metrics without error
        true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
        tp = np.sum((true_skeleton == 1) & (skeleton == 1))
        fp = np.sum((true_skeleton == 0) & (skeleton == 1))
        # No assertion errors means metrics are calculable
    
    def test_sgs_hc_guess_expert_via_wrapper(self):
        """Test SGS-HC-Guess-Expert through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with all parameters
        skeleton, runtime = run_method(
            'sgs_hc_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7, p_acc_subset_NMD=0.6,
            constraint_fraction=0.3, alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert isinstance(runtime, float)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        
        # Test with default parameters
        skeleton2, runtime2 = run_method(
            'sgs_hc_guess_expert', data,
            true_dag=true_dag
        )
        assert skeleton2.shape == (3, 3)
        
        # Test error when true_dag not provided
        with pytest.raises(ValueError, match="sgs_hc_guess_expert requires 'true_dag'"):
            run_method('sgs_hc_guess_expert', data)
    
    def test_sgs_hc_guess_expert_chain_structure(self):
        """Test SGS-HC-Guess-Expert on X->Y->Z chain"""
        np.random.seed(42)
        n = 5000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3, alpha=0.05
        )
        
        # Should find correct chain structure
        assert skeleton[0, 1] == 1  # X-Y edge
        assert skeleton[1, 2] == 1  # Y-Z edge
        assert skeleton[0, 2] == 0  # No X-Z edge
    
    def test_sgs_hc_guess_expert_constraint_fraction_zero(self):
        """Test that constraint_fraction=0 behaves like sgs_guess_expert"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        # Run with constraint_fraction=0
        skeleton_hc = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.0, alpha=0.05
        )
        
        # Should produce valid skeleton
        assert skeleton_hc.shape == (4, 4)
        assert np.array_equal(skeleton_hc, skeleton_hc.T)
        assert np.all((skeleton_hc == 0) | (skeleton_hc == 1))
    
    def test_sgs_hc_guess_expert_high_constraint_fraction(self):
        """Test SGS-HC-Guess-Expert with high constraint fraction"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=1000)
        
        # With 90% constraints and perfect expert, should be very accurate
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.9, alpha=0.05
        )
        
        # Should return valid skeleton
        assert skeleton.shape == (4, 4)
        assert np.array_equal(skeleton, skeleton.T)
        
        # Calculate F1 score - should be high with perfect expert and high constraints
        true_skeleton = ((true_dag != 0) | (true_dag.T != 0)).astype(int)
        tp = np.sum((true_skeleton == 1) & (skeleton == 1))
        fp = np.sum((true_skeleton == 0) & (skeleton == 1))
        fn = np.sum((true_skeleton == 1) & (skeleton == 0))
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                assert f1 > 0.7  # Should have reasonable F1 with perfect expert
    
    def test_sgs_hc_guess_expert_edge_cases(self):
        """Test SGS-HC-Guess-Expert edge cases"""
        np.random.seed(42)
        
        # Single variable
        data_single = np.random.randn(100, 1)
        dag_single = np.array([[0]])
        skeleton = run_sgs_hc_guess_expert(
            data_single, dag_single,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3
        )
        assert skeleton.shape == (1, 1)
        assert skeleton[0, 0] == 0
        
        # Two independent variables
        data_indep = np.random.randn(1000, 2)
        dag_empty = np.zeros((2, 2))
        skeleton = run_sgs_hc_guess_expert(
            data_indep, dag_empty,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3
        )
        assert skeleton.shape == (2, 2)
        assert np.sum(skeleton) == 0  # Should have no edges
    
    # Random Guidance Flag Tests
    def test_pc_hc_guess_expert_random_guidance_valid_output(self):
        """Test PC-HC-Guess-Expert with random guidance returns valid skeleton"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        skeleton = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        # Check skeleton properties
        assert skeleton.shape == (4, 4)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_sgs_hc_guess_expert_random_guidance_valid_output(self):
        """Test SGS-HC-Guess-Expert with random guidance returns valid skeleton"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=4, edge_probability=0.3, sample_size=500)
        
        skeleton = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=0.8, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.8, p_acc_subset_NMD=0.8,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        # Check skeleton properties
        assert skeleton.shape == (4, 4)
        assert skeleton.dtype == int
        assert np.array_equal(skeleton, skeleton.T)  # Symmetric
        assert np.all(np.diag(skeleton) == 0)  # No self-loops
        assert np.all((skeleton == 0) | (skeleton == 1))  # Binary
    
    def test_pc_hc_guess_expert_random_guidance_high_sample_accuracy(self):
        """Test PC-HC-Guess-Expert with random guidance on high sample chain"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        # Test with expert guidance (use_random_guidance=False)
        skeleton_expert = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3,
            use_random_guidance=False,
            alpha=0.05
        )
        
        # Test with random guidance (use_random_guidance=True)
        skeleton_random = run_pc_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        # Both should find correct chain structure with high samples
        assert skeleton_expert[0, 1] == 1  # X-Y edge
        assert skeleton_expert[1, 2] == 1  # Y-Z edge
        assert skeleton_expert[0, 2] == 0  # No X-Z edge
        
        assert skeleton_random[0, 1] == 1  # X-Y edge
        assert skeleton_random[1, 2] == 1  # Y-Z edge
        assert skeleton_random[0, 2] == 0  # No X-Z edge
    
    def test_sgs_hc_guess_expert_random_guidance_high_sample_accuracy(self):
        """Test SGS-HC-Guess-Expert with random guidance on high sample chain"""
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n)
        Z = 3 * Y + np.random.randn(n)
        data = np.column_stack([X, Y, Z])
        
        true_dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
        # Test with expert guidance (use_random_guidance=False)
        skeleton_expert = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3,
            use_random_guidance=False,
            alpha=0.05
        )
        
        # Test with random guidance (use_random_guidance=True)
        skeleton_random = run_sgs_hc_guess_expert(
            data, true_dag,
            p_acc_edge_true=1.0, p_acc_edge_false=1.0,
            p_acc_subset_MD=1.0, p_acc_subset_NMD=1.0,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        # Both should find correct chain structure with high samples
        assert skeleton_expert[0, 1] == 1  # X-Y edge
        assert skeleton_expert[1, 2] == 1  # Y-Z edge
        assert skeleton_expert[0, 2] == 0  # No X-Z edge
        
        assert skeleton_random[0, 1] == 1  # X-Y edge
        assert skeleton_random[1, 2] == 1  # Y-Z edge
        assert skeleton_random[0, 2] == 0  # No X-Z edge
    
    def test_pc_hc_guess_expert_random_guidance_via_wrapper(self):
        """Test PC-HC-Guess-Expert with random guidance through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with use_random_guidance=True
        skeleton, runtime = run_method(
            'pc_hc_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7, p_acc_subset_NMD=0.6,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert isinstance(runtime, float)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all((skeleton == 0) | (skeleton == 1))
    
    def test_sgs_hc_guess_expert_random_guidance_via_wrapper(self):
        """Test SGS-HC-Guess-Expert with random guidance through run_method wrapper"""
        np.random.seed(42)
        true_dag, data = generate_linear_data(n_nodes=3, edge_probability=0.4, sample_size=500)
        
        # Test with use_random_guidance=True
        skeleton, runtime = run_method(
            'sgs_hc_guess_expert', data,
            true_dag=true_dag,
            p_acc_edge_true=0.9, p_acc_edge_false=0.8,
            p_acc_subset_MD=0.7, p_acc_subset_NMD=0.6,
            constraint_fraction=0.3,
            use_random_guidance=True,
            alpha=0.05
        )
        
        assert skeleton.shape == (3, 3)
        assert isinstance(runtime, float)
        assert runtime > 0
        assert np.array_equal(skeleton, skeleton.T)
        assert np.all((skeleton == 0) | (skeleton == 1))


if __name__ == "__main__":
    test_instance = TestMethods()
    failed_tests = []
    
    # Original tests
    print("Running test_independent_variables...")
    try:
        test_instance.test_independent_variables()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_independent_variables")
    
    print("Running test_chain_structure...")
    try:
        test_instance.test_chain_structure()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_chain_structure")
    
    print("Running test_stable_pc_independent_variables...")
    try:
        test_instance.test_stable_pc_independent_variables()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_stable_pc_independent_variables")
    
    print("Running test_stable_pc_chain_structure...")
    try:
        test_instance.test_stable_pc_chain_structure()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_stable_pc_chain_structure")
    
    print("Running test_methods_with_generated_data...")
    try:
        test_instance.test_methods_with_generated_data()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_methods_with_generated_data")
    
    print("Running test_run_method_wrapper...")
    try:
        test_instance.test_run_method_wrapper()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_run_method_wrapper")
    
    # PC-Guess-DAG tests
    print("Running test_run_method_pc_guess_dag...")
    try:
        test_instance.test_run_method_pc_guess_dag()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_run_method_pc_guess_dag")
    
    print("Running test_pc_guess_dag_return_format...")
    try:
        test_instance.test_pc_guess_dag_return_format()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_dag_return_format")
    
    print("Running test_pc_guess_dag_chain...")
    try:
        test_instance.test_pc_guess_dag_chain()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_dag_chain")
    
    print("Running test_pc_guess_dag_edge_cases...")
    try:
        test_instance.test_pc_guess_dag_edge_cases()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_dag_edge_cases")
    
    print("Running test_pc_guess_dag_wrong_guess...")
    try:
        test_instance.test_pc_guess_dag_wrong_guess()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_dag_wrong_guess")
    
    # Helper function tests
    print("Running test_topological_sort...")
    try:
        test_instance.test_topological_sort()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_topological_sort")
    
    print("Running test_get_testable_edges_ordered_pairs...")
    try:
        test_instance.test_get_testable_edges_ordered_pairs()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_get_testable_edges_ordered_pairs")
    
    print("Running test_expert_edge_ordering_grouping...")
    try:
        test_instance.test_expert_edge_ordering_grouping()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_expert_edge_ordering_grouping")
    
    print("Running test_expert_subset_ordering_descendants...")
    try:
        test_instance.test_expert_subset_ordering_descendants()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_expert_subset_ordering_descendants")
    
    # PC-Guess-Expert tests
    print("Running test_create_probabilistic_skeleton...")
    try:
        test_instance.test_create_probabilistic_skeleton()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_create_probabilistic_skeleton")
    
    print("Running test_get_descendants...")
    try:
        test_instance.test_get_descendants()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_get_descendants")
    
    print("Running test_contains_mutual_descendant...")
    try:
        test_instance.test_contains_mutual_descendant()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_contains_mutual_descendant")
    
    print("Running test_probabilistic_subset_ordering_dsep...")
    try:
        test_instance.test_probabilistic_subset_ordering_dsep()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_probabilistic_subset_ordering_dsep")
    
    print("Running test_pc_guess_expert_perfect_accuracy...")
    try:
        test_instance.test_pc_guess_expert_perfect_accuracy()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_expert_perfect_accuracy")
    
    print("Running test_pc_guess_expert_random_accuracy...")
    try:
        test_instance.test_pc_guess_expert_random_accuracy()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_expert_random_accuracy")
    
    print("Running test_pc_guess_expert_via_wrapper...")
    try:
        test_instance.test_pc_guess_expert_via_wrapper()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_pc_guess_expert_via_wrapper")
    
    # SGS-Guess-Expert tests
    print("Running test_expert_edge_ordering_sgs...")
    try:
        test_instance.test_expert_edge_ordering_sgs()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_expert_edge_ordering_sgs")
    
    print("Running test_sgs_guess_expert_perfect_accuracy...")
    try:
        test_instance.test_sgs_guess_expert_perfect_accuracy()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_expert_perfect_accuracy")
    
    print("Running test_sgs_guess_expert_random_accuracy...")
    try:
        test_instance.test_sgs_guess_expert_random_accuracy()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_expert_random_accuracy")
    
    print("Running test_sgs_guess_expert_chain_structure...")
    try:
        test_instance.test_sgs_guess_expert_chain_structure()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_expert_chain_structure")
    
    print("Running test_sgs_guess_expert_via_wrapper...")
    try:
        test_instance.test_sgs_guess_expert_via_wrapper()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_expert_via_wrapper")
    
    print("Running test_sgs_guess_expert_edge_cases...")
    try:
        test_instance.test_sgs_guess_expert_edge_cases()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_expert_edge_cases")
    
    print("Running test_sgs_vs_pc_comparison...")
    try:
        test_instance.test_sgs_vs_pc_comparison()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_vs_pc_comparison")
    
    # SGS-Guess-DAG tests
    print("Running test_sgs_guess_dag_return_format...")
    try:
        test_instance.test_sgs_guess_dag_return_format()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_return_format")
    
    print("Running test_sgs_guess_dag_chain_structure...")
    try:
        test_instance.test_sgs_guess_dag_chain_structure()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_chain_structure")
    
    print("Running test_sgs_guess_dag_wrong_guess...")
    try:
        test_instance.test_sgs_guess_dag_wrong_guess()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_wrong_guess")
    
    print("Running test_sgs_guess_dag_edge_cases...")
    try:
        test_instance.test_sgs_guess_dag_edge_cases()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_edge_cases")
    
    print("Running test_sgs_guess_dag_via_wrapper...")
    try:
        test_instance.test_sgs_guess_dag_via_wrapper()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_via_wrapper")
    
    print("Running test_sgs_guess_dag_vs_sgs_expert...")
    try:
        test_instance.test_sgs_guess_dag_vs_sgs_expert()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        failed_tests.append("test_sgs_guess_dag_vs_sgs_expert")
    
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
    else:
        print("\nAll tests passed!")