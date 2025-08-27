import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from supernova.recursive import (
    iterate_strategies, optimize_strategy_genetic, GeneticConfig, Individual,
    run_portfolio_optimization, run_ensemble_optimization,
    _initialize_population, _tournament_selection, _crossover, _mutate
)
from supernova.backtester import run_backtest


@pytest.fixture
def sample_strategies():
    """Sample strategies for testing"""
    return [
        {
            "name": "Fast MA Cross",
            "template": "ma_crossover",
            "params": {"fast": 5, "slow": 15}
        },
        {
            "name": "Slow MA Cross",
            "template": "ma_crossover", 
            "params": {"fast": 10, "slow": 30}
        },
        {
            "name": "RSI Breakout",
            "template": "rsi_breakout",
            "params": {"length": 14, "low_th": 30, "high_th": 70}
        }
    ]


@pytest.fixture
def sample_bars_recursive():
    """Generate sample bars for recursive testing"""
    np.random.seed(123)  # For reproducible tests
    bars = []
    base_price = 100.0
    
    for i in range(200):
        timestamp = (datetime.now() - timedelta(hours=200-i)).isoformat() + "Z"
        
        # Add trend and some noise
        trend = 0.001 if i > 100 else -0.0005
        daily_return = trend + np.random.normal(0, 0.01)
        base_price *= (1 + daily_return)
        base_price = max(1.0, base_price)
        
        # Generate OHLC
        high = base_price * (1 + abs(np.random.normal(0, 0.005)))
        low = base_price * (1 - abs(np.random.normal(0, 0.005)))
        close = base_price * (1 + np.random.normal(0, 0.003))
        
        bars.append({
            "timestamp": timestamp,
            "open": float(base_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": int(10000 + np.random.normal(0, 1000))
        })
        
        base_price = close
    
    return bars


@pytest.fixture
def multi_asset_sample():
    """Multi-asset data for portfolio testing"""
    np.random.seed(456)
    
    # Generate two correlated assets
    n_periods = 150
    base_prices = {"ASSET_1": 100.0, "ASSET_2": 50.0}
    
    asset_data = {"ASSET_1": [], "ASSET_2": []}
    
    for i in range(n_periods):
        timestamp = (datetime.now() - timedelta(days=n_periods-i)).isoformat() + "Z"
        
        # Shared market factor
        market_return = np.random.normal(0.0002, 0.01)
        
        for j, (asset, base_price) in enumerate(base_prices.items()):
            # Asset-specific return with market correlation
            asset_return = market_return * 0.7 + np.random.normal(0, 0.008)
            new_price = base_price * (1 + asset_return)
            new_price = max(1.0, new_price)
            
            high = new_price * (1 + abs(np.random.normal(0, 0.003)))
            low = new_price * (1 - abs(np.random.normal(0, 0.003)))
            close = new_price * (1 + np.random.normal(0, 0.002))
            
            asset_data[asset].append({
                "timestamp": timestamp,
                "open": float(new_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": int(5000 + np.random.normal(0, 500))
            })
            
            base_prices[asset] = close
    
    return asset_data


class TestIterateStrategies:
    """Tests for enhanced strategy iteration"""
    
    def test_basic_iteration(self, sample_bars_recursive, sample_strategies):
        """Test basic strategy iteration with parallel processing"""
        results = iterate_strategies(sample_bars_recursive, sample_strategies)
        
        assert len(results) == len(sample_strategies)
        
        for result in results:
            assert "name" in result
            assert "template" in result
            assert "params" in result
            assert "metrics" in result
            
            # Check metrics structure
            metrics = result["metrics"]
            if "error" not in metrics:
                assert "Sharpe" in metrics
                assert "CAGR" in metrics
                assert "MaxDrawdown" in metrics
    
    def test_iteration_sorting(self, sample_bars_recursive, sample_strategies):
        """Test that results are properly sorted"""
        results = iterate_strategies(sample_bars_recursive, sample_strategies)
        
        # Filter out error results for sorting test
        valid_results = [r for r in results if "error" not in r["metrics"]]
        
        if len(valid_results) >= 2:
            # Should be sorted by Sharpe ratio (descending)
            for i in range(len(valid_results) - 1):
                current_sharpe = valid_results[i]["metrics"]["Sharpe"]
                next_sharpe = valid_results[i + 1]["metrics"]["Sharpe"]
                assert current_sharpe >= next_sharpe
    
    def test_failed_strategy_handling(self, sample_bars_recursive):
        """Test handling of failed strategies"""
        bad_strategies = [
            {
                "name": "Invalid Strategy",
                "template": "nonexistent_template",
                "params": {}
            }
        ]
        
        results = iterate_strategies(sample_bars_recursive, bad_strategies)
        
        assert len(results) == 1
        assert "error" in results[0]["metrics"]
    
    def test_mixed_strategies(self, sample_bars_recursive):
        """Test mix of valid and invalid strategies"""
        mixed_strategies = [
            {
                "name": "Valid Strategy",
                "template": "ma_crossover",
                "params": {"fast": 10, "slow": 20}
            },
            {
                "name": "Invalid Strategy", 
                "template": "bad_template",
                "params": {}
            }
        ]
        
        results = iterate_strategies(sample_bars_recursive, mixed_strategies)
        
        assert len(results) == 2
        
        # Valid strategy should be first (higher score)
        valid_result = next(r for r in results if r["name"] == "Valid Strategy")
        invalid_result = next(r for r in results if r["name"] == "Invalid Strategy")
        
        assert "error" not in valid_result["metrics"]
        assert "error" in invalid_result["metrics"]


class TestGeneticOptimization:
    """Tests for genetic algorithm optimization"""
    
    def test_population_initialization(self):
        """Test population initialization"""
        param_ranges = {
            "fast": (5, 20, 'int'),
            "slow": (20, 50, 'int'),
            "threshold": (0.1, 0.9, 'float'),
            "enable_feature": (True, [True, False], 'choice')
        }
        
        population = _initialize_population(param_ranges, 10)
        
        assert len(population) == 10
        
        for individual in population:
            assert isinstance(individual, Individual)
            assert "fast" in individual.genes
            assert "slow" in individual.genes
            assert "threshold" in individual.genes
            assert "enable_feature" in individual.genes
            
            # Check ranges
            assert 5 <= individual.genes["fast"] <= 20
            assert 20 <= individual.genes["slow"] <= 50
            assert 0.1 <= individual.genes["threshold"] <= 0.9
            assert individual.genes["enable_feature"] in [True, False]
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        # Create population with known fitness values
        population = []
        fitness_values = [0.1, 0.5, 0.3, 0.8, 0.2]
        
        for i, fitness in enumerate(fitness_values):
            individual = Individual(genes={"param": i}, fitness=fitness)
            population.append(individual)
        
        # Run tournament selection multiple times
        selected_count = {}
        for _ in range(100):
            selected = _tournament_selection(population, 3)
            param_value = selected.genes["param"]
            selected_count[param_value] = selected_count.get(param_value, 0) + 1
        
        # Higher fitness individuals should be selected more often
        # Individual with fitness 0.8 (param=3) should be selected most
        most_selected = max(selected_count.items(), key=lambda x: x[1])
        assert most_selected[0] == 3  # Individual with highest fitness
    
    def test_crossover(self):
        """Test crossover operation"""
        parent1 = Individual(genes={"a": 1, "b": 2, "c": 3})
        parent2 = Individual(genes={"a": 10, "b": 20, "c": 30})
        
        child1, child2 = _crossover(parent1, parent2, generation=1)
        
        assert isinstance(child1, Individual)
        assert isinstance(child2, Individual)
        assert child1.generation == 1
        assert child2.generation == 1
        assert child1.parent1 == parent1
        assert child1.parent2 == parent2
        
        # Children should have mix of parent genes
        assert set(child1.genes.keys()) == {"a", "b", "c"}
        assert set(child2.genes.keys()) == {"a", "b", "c"}
    
    def test_mutation(self):
        """Test mutation operation"""
        param_ranges = {
            "int_param": (1, 10, 'int'),
            "float_param": (0.0, 1.0, 'float'),
            "bool_param": (False, True, 'bool'),
            "choice_param": (None, ['A', 'B', 'C'], 'choice')
        }
        
        individual = Individual(genes={
            "int_param": 5,
            "float_param": 0.5,
            "bool_param": True,
            "choice_param": 'A'
        })
        
        # Run mutation multiple times to test stochastic behavior
        mutations_occurred = 0
        for _ in range(50):
            mutated = _mutate(individual, param_ranges)
            
            if mutated.genes != individual.genes:
                mutations_occurred += 1
                
                # Check that mutated values are still within ranges
                assert 1 <= mutated.genes["int_param"] <= 10
                assert 0.0 <= mutated.genes["float_param"] <= 1.0
                assert mutated.genes["bool_param"] in [True, False]
                assert mutated.genes["choice_param"] in ['A', 'B', 'C']
        
        # Some mutations should have occurred
        assert mutations_occurred > 0
    
    def test_full_genetic_optimization(self, sample_bars_recursive):
        """Test complete genetic optimization workflow"""
        param_ranges = {
            "fast": (5, 15, 'int'),
            "slow": (20, 40, 'int')
        }
        
        config = GeneticConfig(
            population_size=8,
            generations=3,
            mutation_rate=0.2,
            crossover_rate=0.7
        )
        
        result = optimize_strategy_genetic(
            sample_bars_recursive, "ma_crossover", param_ranges, config
        )
        
        assert "optimized_params" in result
        assert "best_fitness" in result
        assert "final_metrics" in result
        assert "generations_run" in result
        assert "fitness_history" in result
        
        # Check parameter bounds
        params = result["optimized_params"]
        assert 5 <= params["fast"] <= 15
        assert 20 <= params["slow"] <= 40
        assert params["fast"] < params["slow"]  # Logical constraint
        
        # Check evolution tracking
        assert len(result["fitness_history"]) <= config.generations
        assert result["generations_run"] <= config.generations
    
    def test_genetic_config_variations(self, sample_bars_recursive):
        """Test different genetic algorithm configurations"""
        param_ranges = {"length": (10, 20, 'int')}
        
        configs = [
            GeneticConfig(fitness_function="sharpe"),
            GeneticConfig(fitness_function="calmar"),
            GeneticConfig(fitness_function="sortino")
        ]
        
        for config in configs:
            config.population_size = 6
            config.generations = 2
            
            result = optimize_strategy_genetic(
                sample_bars_recursive, "rsi_breakout", param_ranges, config
            )
            
            assert "optimized_params" in result
            assert result["optimization_config"]["fitness_function"] == config.fitness_function
    
    def test_early_convergence(self, sample_bars_recursive):
        """Test early stopping on convergence"""
        param_ranges = {"fast": (10, 12, 'int'), "slow": (25, 27, 'int')}
        
        config = GeneticConfig(
            population_size=4,
            generations=20,  # Many generations
            max_stagnant_generations=2,  # But early stopping
            convergence_threshold=0.001
        )
        
        result = optimize_strategy_genetic(
            sample_bars_recursive, "ma_crossover", param_ranges, config
        )
        
        # Should stop early due to convergence
        assert result["generations_run"] < config.generations


class TestPortfolioOptimization:
    """Tests for portfolio optimization"""
    
    def test_risk_parity_optimization(self, multi_asset_sample):
        """Test risk parity optimization"""
        result = run_portfolio_optimization(
            multi_asset_sample, optimization_method="risk_parity"
        )
        
        assert "weights" in result
        assert "expected_return" in result
        assert "expected_volatility" in result
        assert "sharpe_ratio" in result
        assert result["optimization_method"] == "risk_parity"
        
        # Weights should sum to approximately 1
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be positive
        assert all(w > 0 for w in result["weights"].values())
    
    def test_different_optimization_methods(self, multi_asset_sample):
        """Test different portfolio optimization methods"""
        methods = ["risk_parity", "min_variance", "mean_variance", "max_diversification"]
        
        for method in methods:
            result = run_portfolio_optimization(
                multi_asset_sample, optimization_method=method
            )
            
            assert result["optimization_method"] == method
            assert "weights" in result
            assert sum(result["weights"].values()) > 0.95  # Should sum to ~1
    
    def test_portfolio_constraints(self, multi_asset_sample):
        """Test portfolio optimization with constraints"""
        constraints = {
            "max_weight": 0.7,
            "min_weight": 0.2,
            "target_volatility": 0.15
        }
        
        result = run_portfolio_optimization(
            multi_asset_sample,
            optimization_method="risk_parity", 
            constraints=constraints
        )
        
        assert result["constraints"] == constraints
        
        # Check weight constraints
        for weight in result["weights"].values():
            assert constraints["min_weight"] <= weight <= constraints["max_weight"]
    
    def test_correlation_matrix(self, multi_asset_sample):
        """Test correlation matrix calculation"""
        result = run_portfolio_optimization(multi_asset_sample)
        
        assert "correlation_matrix" in result
        
        corr_matrix = result["correlation_matrix"]
        assets = list(multi_asset_sample.keys())
        
        # Check diagonal elements are 1
        for asset in assets:
            assert abs(corr_matrix[asset][asset] - 1.0) < 0.01
        
        # Check symmetry
        if len(assets) >= 2:
            asset1, asset2 = assets[0], assets[1]
            assert abs(corr_matrix[asset1][asset2] - corr_matrix[asset2][asset1]) < 0.01


class TestEnsembleOptimization:
    """Tests for ensemble strategy optimization"""
    
    def test_basic_ensemble_optimization(self, sample_bars_recursive):
        """Test basic ensemble optimization"""
        strategies = ["ma_crossover", "rsi_breakout", "macd_trend"]
        
        result = run_ensemble_optimization(sample_bars_recursive, strategies)
        
        if "error" not in result:
            assert "ensemble_weights" in result
            assert "individual_results" in result
            assert "correlation_matrix" in result
            assert "ensemble_sharpe" in result
            assert "ensemble_cagr" in result
            assert "ensemble_equity_curve" in result
            
            # Weights should sum to 1
            total_weight = sum(result["ensemble_weights"].values())
            assert abs(total_weight - 1.0) < 0.01
            
            # Should have results for each strategy
            assert len(result["individual_results"]) <= len(strategies)
    
    def test_ensemble_with_parameters(self, sample_bars_recursive):
        """Test ensemble optimization with strategy parameters"""
        strategies = ["ma_crossover", "rsi_breakout"]
        optimization_params = {
            "ma_crossover": {"fast": 8, "slow": 24},
            "rsi_breakout": {"length": 12, "low_th": 35, "high_th": 65}
        }
        
        result = run_ensemble_optimization(
            sample_bars_recursive, strategies, optimization_params
        )
        
        if "error" not in result:
            # Check that parameters were used
            for individual_result in result["individual_results"]:
                strategy_name = individual_result["strategy"]
                if strategy_name in optimization_params:
                    assert individual_result["params"] == optimization_params[strategy_name]
    
    def test_ensemble_correlation_handling(self, sample_bars_recursive):
        """Test ensemble handling of strategy correlations"""
        # Use similar strategies that should be correlated
        strategies = ["ma_crossover", "ma_crossover", "ma_crossover"]
        optimization_params = {
            "ma_crossover": [
                {"fast": 5, "slow": 15},
                {"fast": 6, "slow": 16}, 
                {"fast": 7, "slow": 17}
            ]
        }
        
        result = run_ensemble_optimization(sample_bars_recursive, strategies)
        
        if "error" not in result:
            # Should detect high correlation and use Sharpe weighting
            assert "correlation_matrix" in result
            correlation_matrix = np.array(result["correlation_matrix"])
            
            if correlation_matrix.size > 0:
                # Check correlation matrix properties
                assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
    
    def test_ensemble_insufficient_strategies(self, sample_bars_recursive):
        """Test ensemble with insufficient successful strategies"""
        strategies = ["nonexistent_strategy1", "nonexistent_strategy2"]
        
        result = run_ensemble_optimization(sample_bars_recursive, strategies)
        
        assert "error" in result
    
    def test_ensemble_diversification_calculation(self, sample_bars_recursive):
        """Test diversification ratio calculation"""
        strategies = ["ma_crossover", "rsi_breakout"]
        
        result = run_ensemble_optimization(sample_bars_recursive, strategies)
        
        if "error" not in result:
            assert "diversification_ratio" in result
            assert isinstance(result["diversification_ratio"], (int, float))
            assert 0 < result["diversification_ratio"] <= 1


class TestAdvancedFeatures:
    """Tests for advanced recursive features"""
    
    def test_genetic_config_dataclass(self):
        """Test GeneticConfig dataclass functionality"""
        config = GeneticConfig()
        
        # Check default values
        assert config.population_size == 50
        assert config.generations == 20
        assert config.mutation_rate == 0.1
        assert config.fitness_function == "sharpe"
        
        # Test custom values
        custom_config = GeneticConfig(
            population_size=30,
            generations=15,
            fitness_function="calmar"
        )
        
        assert custom_config.population_size == 30
        assert custom_config.generations == 15
        assert custom_config.fitness_function == "calmar"
    
    def test_individual_dataclass(self):
        """Test Individual dataclass functionality"""
        genes = {"param1": 10, "param2": 0.5}
        individual = Individual(genes=genes, fitness=0.8, generation=2)
        
        assert individual.genes == genes
        assert individual.fitness == 0.8
        assert individual.generation == 2
        assert individual.parent1 is None
        assert individual.parent2 is None
        
        # Test with parents
        parent = Individual(genes={"param": 1})
        child = Individual(genes=genes, parent1=parent)
        assert child.parent1 == parent
    
    def test_error_handling_robustness(self, sample_bars_recursive):
        """Test robustness of error handling"""
        # Test with empty bars
        empty_result = iterate_strategies([], [{"name": "test", "template": "ma_crossover"}])
        assert len(empty_result) == 1
        assert "error" in empty_result[0]["metrics"]
        
        # Test with invalid parameter ranges
        with pytest.raises((ValueError, KeyError)):
            optimize_strategy_genetic(
                sample_bars_recursive,
                "ma_crossover", 
                {"invalid_param": (1, 10, 'int')}
            )
    
    def test_performance_monitoring(self, sample_bars_recursive):
        """Test performance monitoring features"""
        strategies = [
            {"name": f"Strategy_{i}", "template": "ma_crossover", "params": {"fast": 5+i, "slow": 20+i}}
            for i in range(3)
        ]
        
        # This should complete in reasonable time with parallel processing
        import time
        start_time = time.time()
        
        results = iterate_strategies(sample_bars_recursive, strategies)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 30  # 30 seconds max for 3 strategies
        assert len(results) == len(strategies)