from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from .backtester import run_backtest, run_multi_asset_backtest
from .strategy_engine import TEMPLATES

@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm optimization"""
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.2
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnant_generations: int = 5
    fitness_function: str = "sharpe"  # 'sharpe', 'calmar', 'sortino', 'profit_factor'
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ['Sharpe', 'MaxDrawdown'])

@dataclass
class Individual:
    """Individual in genetic algorithm population"""
    genes: Dict[str, Any]  # Parameter values
    fitness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent1: Optional['Individual'] = None
    parent2: Optional['Individual'] = None

def iterate_strategies(bars: list[dict], strategies: list[dict]) -> list[Dict[str, Any]]:
    """Enhanced strategy iteration with parallel processing"""
    results = []
    
    # Parallel execution for multiple strategies
    with ThreadPoolExecutor(max_workers=min(4, len(strategies))) as executor:
        future_to_strategy = {}
        
        for s in strategies:
            future = executor.submit(
                run_backtest, 
                bars, 
                s["template"], 
                s.get("params", {}),
                s.get("start_cash", 10000),
                s.get("fee_rate", 0.0005),
                s.get("allow_short", False)
            )
            future_to_strategy[future] = s
        
        for future in as_completed(future_to_strategy):
            strategy = future_to_strategy[future]
            try:
                metrics = future.result()
                results.append({
                    "name": strategy["name"],
                    "template": strategy["template"],
                    "params": strategy.get("params", {}),
                    "metrics": metrics
                })
            except Exception as e:
                print(f"Strategy {strategy['name']} failed: {str(e)}")
                results.append({
                    "name": strategy["name"],
                    "template": strategy["template"],
                    "params": strategy.get("params", {}),
                    "metrics": {"error": str(e)}
                })
    
    # Enhanced sorting with multiple criteria
    def sort_key(r):
        metrics = r["metrics"]
        if "error" in metrics:
            return (-1000, 100, -1000)  # Push errors to bottom
        
        sharpe = metrics.get("Sharpe", 0)
        max_dd = metrics.get("MaxDrawdown", 100)
        calmar = metrics.get("Calmar", 0)
        
        return (-sharpe, max_dd, -calmar)
    
    results.sort(key=sort_key)
    return results

def optimize_strategy_genetic(bars: list[dict], template: str, param_ranges: Dict[str, Tuple],
                            config: GeneticConfig = None) -> Dict[str, Any]:
    """Genetic algorithm optimization for strategy parameters"""
    
    if config is None:
        config = GeneticConfig()
    
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    
    # Initialize population
    population = _initialize_population(param_ranges, config.population_size)
    
    # Evolution tracking
    best_fitness_history = []
    avg_fitness_history = []
    stagnant_generations = 0
    best_individual = None
    
    for generation in range(config.generations):
        # Evaluate population
        population = _evaluate_population(population, bars, template, config)
        
        # Track progress
        current_best = max(population, key=lambda x: x.fitness)
        avg_fitness = np.mean([ind.fitness for ind in population])
        
        best_fitness_history.append(current_best.fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Check for improvement
        if best_individual is None or current_best.fitness > best_individual.fitness:
            if best_individual and abs(current_best.fitness - best_individual.fitness) < config.convergence_threshold:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
            best_individual = current_best
        else:
            stagnant_generations += 1
        
        # Early stopping
        if stagnant_generations >= config.max_stagnant_generations:
            break
        
        # Create next generation
        if generation < config.generations - 1:
            population = _create_next_generation(population, config, generation + 1)
    
    # Final evaluation with detailed metrics
    final_metrics = run_backtest(bars, template, best_individual.genes)
    
    return {
        "optimized_params": best_individual.genes,
        "best_fitness": best_individual.fitness,
        "final_metrics": final_metrics,
        "generations_run": generation + 1,
        "fitness_history": best_fitness_history,
        "avg_fitness_history": avg_fitness_history,
        "convergence_generation": generation - stagnant_generations,
        "optimization_config": config.__dict__
    }

def _initialize_population(param_ranges: Dict[str, Tuple], population_size: int) -> List[Individual]:
    """Initialize random population within parameter ranges"""
    population = []
    
    for _ in range(population_size):
        genes = {}
        for param, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'int':
                genes[param] = random.randint(int(min_val), int(max_val))
            elif param_type == 'float':
                genes[param] = random.uniform(min_val, max_val)
            elif param_type == 'bool':
                genes[param] = random.choice([True, False])
            elif param_type == 'choice':
                genes[param] = random.choice(max_val)  # max_val contains choices list
        
        population.append(Individual(genes=genes))
    
    return population

def _evaluate_population(population: List[Individual], bars: list[dict], 
                        template: str, config: GeneticConfig) -> List[Individual]:
    """Evaluate fitness for entire population"""
    
    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=min(4, len(population))) as executor:
        future_to_individual = {}
        
        for individual in population:
            future = executor.submit(run_backtest, bars, template, individual.genes)
            future_to_individual[future] = individual
        
        for future in as_completed(future_to_individual):
            individual = future_to_individual[future]
            try:
                metrics = future.result()
                individual.metrics = metrics
                
                # Calculate fitness based on configuration
                if config.fitness_function == "sharpe":
                    individual.fitness = metrics.get("Sharpe", -1000)
                elif config.fitness_function == "calmar":
                    individual.fitness = metrics.get("Calmar", -1000)
                elif config.fitness_function == "sortino":
                    individual.fitness = metrics.get("Sortino", -1000)
                elif config.fitness_function == "profit_factor":
                    individual.fitness = metrics.get("ProfitFactor", 0)
                else:
                    # Multi-objective fitness (weighted combination)
                    fitness = 0
                    weights = {'Sharpe': 0.4, 'Calmar': 0.3, 'Sortino': 0.2, 'ProfitFactor': 0.1}
                    for metric, weight in weights.items():
                        fitness += metrics.get(metric, 0) * weight
                    individual.fitness = fitness
                
            except Exception as e:
                individual.fitness = -1000  # Penalty for failed evaluations
                individual.metrics = {"error": str(e)}
    
    return population

def _create_next_generation(population: List[Individual], config: GeneticConfig, 
                          generation: int) -> List[Individual]:
    """Create next generation through selection, crossover, and mutation"""
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    next_generation = []
    
    # Elitism - keep best individuals
    elite_count = int(len(population) * config.elitism_rate)
    for i in range(elite_count):
        elite = Individual(genes=population[i].genes.copy(), generation=generation)
        next_generation.append(elite)
    
    # Generate offspring
    while len(next_generation) < len(population):
        # Tournament selection
        parent1 = _tournament_selection(population, config.tournament_size)
        parent2 = _tournament_selection(population, config.tournament_size)
        
        # Crossover
        if random.random() < config.crossover_rate:
            child1, child2 = _crossover(parent1, parent2, generation)
        else:
            child1 = Individual(genes=parent1.genes.copy(), generation=generation, parent1=parent1)
            child2 = Individual(genes=parent2.genes.copy(), generation=generation, parent1=parent2)
        
        # Mutation
        if random.random() < config.mutation_rate:
            child1 = _mutate(child1, _extract_param_ranges(population))
        if random.random() < config.mutation_rate:
            child2 = _mutate(child2, _extract_param_ranges(population))
        
        next_generation.extend([child1, child2])
    
    return next_generation[:len(population)]

def _tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    """Tournament selection"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)

def _crossover(parent1: Individual, parent2: Individual, generation: int) -> Tuple[Individual, Individual]:
    """Single-point crossover"""
    genes1 = parent1.genes.copy()
    genes2 = parent2.genes.copy()
    
    # Single-point crossover
    params = list(genes1.keys())
    if len(params) > 1:
        crossover_point = random.randint(1, len(params) - 1)
        
        for i, param in enumerate(params):
            if i >= crossover_point:
                genes1[param], genes2[param] = genes2[param], genes1[param]
    
    child1 = Individual(genes=genes1, generation=generation, parent1=parent1, parent2=parent2)
    child2 = Individual(genes=genes2, generation=generation, parent1=parent1, parent2=parent2)
    
    return child1, child2

def _mutate(individual: Individual, param_ranges: Dict[str, Tuple]) -> Individual:
    """Gaussian mutation"""
    mutated_genes = individual.genes.copy()
    
    for param, (min_val, max_val, param_type) in param_ranges.items():
        if random.random() < 0.1:  # 10% chance to mutate each parameter
            if param_type == 'int':
                # Gaussian mutation for integers
                current_val = mutated_genes[param]
                mutation_strength = (max_val - min_val) * 0.1
                new_val = int(current_val + random.gauss(0, mutation_strength))
                mutated_genes[param] = max(min_val, min(max_val, new_val))
            
            elif param_type == 'float':
                # Gaussian mutation for floats
                current_val = mutated_genes[param]
                mutation_strength = (max_val - min_val) * 0.1
                new_val = current_val + random.gauss(0, mutation_strength)
                mutated_genes[param] = max(min_val, min(max_val, new_val))
            
            elif param_type == 'bool':
                # Flip boolean
                mutated_genes[param] = not mutated_genes[param]
            
            elif param_type == 'choice':
                # Random choice from available options
                mutated_genes[param] = random.choice(max_val)
    
    return Individual(genes=mutated_genes, generation=individual.generation)

def _extract_param_ranges(population: List[Individual]) -> Dict[str, Tuple]:
    """Extract parameter ranges from population for mutation"""
    if not population:
        return {}
    
    param_ranges = {}
    first_genes = population[0].genes
    
    for param in first_genes.keys():
        values = [ind.genes[param] for ind in population]
        
        if isinstance(first_genes[param], int):
            param_ranges[param] = (min(values), max(values), 'int')
        elif isinstance(first_genes[param], float):
            param_ranges[param] = (min(values), max(values), 'float')
        elif isinstance(first_genes[param], bool):
            param_ranges[param] = (False, True, 'bool')
        else:
            unique_values = list(set(values))
            param_ranges[param] = (None, unique_values, 'choice')
    
    return param_ranges

def run_portfolio_optimization(asset_data: Dict[str, list], 
                             optimization_method: str = "risk_parity",
                             constraints: Dict[str, Any] = None) -> Dict[str, Any]:
    """Portfolio optimization with multiple methods"""
    
    if constraints is None:
        constraints = {
            "max_weight": 0.4,
            "min_weight": 0.05,
            "target_volatility": 0.15,
            "max_drawdown": 0.2
        }
    
    # Calculate returns and covariance matrix
    returns_data = {}
    for asset, bars in asset_data.items():
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        returns_data[asset] = df["close"].pct_change().dropna()
    
    # Align returns data
    min_length = min(len(returns) for returns in returns_data.values())
    aligned_returns = {asset: returns.iloc[-min_length:] for asset, returns in returns_data.items()}
    
    # Create returns matrix
    returns_matrix = pd.DataFrame(aligned_returns)
    mean_returns = returns_matrix.mean() * 252  # Annualized
    cov_matrix = returns_matrix.cov() * 252     # Annualized
    
    # Optimization based on method
    if optimization_method == "risk_parity":
        weights = _risk_parity_optimization(cov_matrix, constraints)
    elif optimization_method == "mean_variance":
        weights = _mean_variance_optimization(mean_returns, cov_matrix, constraints)
    elif optimization_method == "min_variance":
        weights = _min_variance_optimization(cov_matrix, constraints)
    elif optimization_method == "max_diversification":
        weights = _max_diversification_optimization(mean_returns, cov_matrix, constraints)
    else:
        # Equal weight as fallback
        n_assets = len(asset_data)
        weights = {asset: 1.0 / n_assets for asset in asset_data.keys()}
    
    # Calculate portfolio metrics
    portfolio_return = sum(weights[asset] * mean_returns[asset] for asset in weights.keys())
    portfolio_vol = np.sqrt(sum(sum(weights[i] * weights[j] * cov_matrix.loc[i, j] 
                                   for j in weights.keys()) for i in weights.keys()))
    sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol  # Assuming 2% risk-free rate
    
    return {
        "weights": weights,
        "expected_return": float(portfolio_return),
        "expected_volatility": float(portfolio_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "optimization_method": optimization_method,
        "constraints": constraints,
        "correlation_matrix": cov_matrix.corr().to_dict()
    }

def _risk_parity_optimization(cov_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> Dict[str, float]:
    """Risk parity portfolio optimization"""
    # Simplified risk parity - inverse volatility weighting
    volatilities = np.sqrt(np.diag(cov_matrix))
    inv_vol_weights = 1 / volatilities
    weights = inv_vol_weights / inv_vol_weights.sum()
    
    # Apply constraints
    max_weight = constraints.get("max_weight", 0.4)
    min_weight = constraints.get("min_weight", 0.05)
    
    # Clip weights to constraints
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()  # Renormalize
    
    return dict(zip(cov_matrix.index, weights))

def _mean_variance_optimization(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                              constraints: Dict[str, Any]) -> Dict[str, float]:
    """Mean-variance optimization (simplified)"""
    # Simple mean-variance without optimization library
    # This is a simplified version - in practice, you'd use scipy.optimize
    
    # Equal weight as baseline
    n_assets = len(mean_returns)
    weights = np.ones(n_assets) / n_assets
    
    # Apply constraints
    max_weight = constraints.get("max_weight", 0.4)
    min_weight = constraints.get("min_weight", 0.05)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()
    
    return dict(zip(mean_returns.index, weights))

def _min_variance_optimization(cov_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> Dict[str, float]:
    """Minimum variance portfolio"""
    # Inverse covariance weighting (simplified)
    try:
        inv_cov = np.linalg.pinv(cov_matrix.values)
        ones = np.ones((len(cov_matrix), 1))
        weights = inv_cov @ ones
        weights = weights.flatten() / weights.sum()
    except:
        # Fallback to equal weights if matrix inversion fails
        n_assets = len(cov_matrix)
        weights = np.ones(n_assets) / n_assets
    
    # Apply constraints
    max_weight = constraints.get("max_weight", 0.4)
    min_weight = constraints.get("min_weight", 0.05)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()
    
    return dict(zip(cov_matrix.index, weights))

def _max_diversification_optimization(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                   constraints: Dict[str, Any]) -> Dict[str, float]:
    """Maximum diversification portfolio"""
    # Simplified diversification ratio maximization
    volatilities = np.sqrt(np.diag(cov_matrix))
    weights = 1 / volatilities
    weights = weights / weights.sum()
    
    # Apply constraints
    max_weight = constraints.get("max_weight", 0.4)
    min_weight = constraints.get("min_weight", 0.05)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()
    
    return dict(zip(cov_matrix.index, weights))

# Enhanced strategy combination functions
def run_ensemble_optimization(bars: list[dict], strategies: List[str], 
                             optimization_params: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Optimize ensemble weights for multiple strategies"""
    
    if optimization_params is None:
        optimization_params = {strategy: {} for strategy in strategies}
    
    # Run individual strategies
    strategy_results = []
    for strategy in strategies:
        params = optimization_params.get(strategy, {})
        result = run_backtest(bars, strategy, params)
        if "error" not in result:
            strategy_results.append({
                "strategy": strategy,
                "params": params,
                "metrics": result,
                "equity_curve": result.get("equity_curve", [])
            })
    
    if len(strategy_results) < 2:
        return {"error": "Need at least 2 successful strategies for ensemble"}
    
    # Calculate correlation matrix between strategies
    equity_curves = [r["equity_curve"] for r in strategy_results]
    min_length = min(len(curve) for curve in equity_curves)
    
    correlation_matrix = np.corrcoef([curve[:min_length] for curve in equity_curves])
    
    # Optimize ensemble weights (simplified - equal weight for low correlation, weighted by Sharpe for high correlation)
    weights = []
    sharpe_scores = [r["metrics"].get("Sharpe", 0) for r in strategy_results]
    
    avg_correlation = correlation_matrix.mean()
    if avg_correlation < 0.3:  # Low correlation - equal weight
        weights = [1.0 / len(strategy_results)] * len(strategy_results)
    else:  # High correlation - Sharpe-weighted
        total_sharpe = sum(max(0, s) for s in sharpe_scores)
        if total_sharpe > 0:
            weights = [max(0, s) / total_sharpe for s in sharpe_scores]
        else:
            weights = [1.0 / len(strategy_results)] * len(strategy_results)
    
    # Calculate ensemble performance
    ensemble_equity = np.zeros(min_length)
    for i, weight in enumerate(weights):
        ensemble_equity += weight * np.array(equity_curves[i][:min_length])
    
    # Calculate ensemble metrics
    ensemble_returns = np.diff(np.log(ensemble_equity + 1e-9))
    ensemble_sharpe = (ensemble_returns.mean() * 252) / (ensemble_returns.std() * np.sqrt(252) + 1e-9)
    ensemble_cagr = (ensemble_equity[-1] / ensemble_equity[0]) ** (252 / len(ensemble_equity)) - 1
    
    return {
        "ensemble_weights": dict(zip([r["strategy"] for r in strategy_results], weights)),
        "individual_results": strategy_results,
        "correlation_matrix": correlation_matrix.tolist(),
        "ensemble_sharpe": float(ensemble_sharpe),
        "ensemble_cagr": float(ensemble_cagr * 100),
        "ensemble_equity_curve": ensemble_equity.tolist(),
        "diversification_ratio": float(1.0 / np.sqrt(len(strategies)))  # Simplified
    }

def run_bayesian_optimization(bars: list[dict], template: str, param_ranges: Dict[str, Tuple],
                             n_iterations: int = 50, acquisition_function: str = "ei") -> Dict[str, Any]:
    """
    Bayesian optimization for strategy parameter tuning
    Uses Gaussian Process regression to model the objective function
    """
    try:
        from scipy.stats import norm
        from scipy.optimize import minimize
    except ImportError:
        return {"error": "scipy required for Bayesian optimization"}
    
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    
    # Initialize with random samples
    n_init = min(10, n_iterations // 3)
    X_samples = []  # Parameter combinations
    y_samples = []  # Objective values (negative for minimization)
    
    # Random initialization
    for _ in range(n_init):
        sample = {}
        for param, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'int':
                sample[param] = random.randint(int(min_val), int(max_val))
            elif param_type == 'float':
                sample[param] = random.uniform(min_val, max_val)
            elif param_type == 'bool':
                sample[param] = random.choice([True, False])
            elif param_type == 'choice':
                sample[param] = random.choice(max_val)  # max_val contains choices
        
        # Evaluate sample
        try:
            result = run_backtest(bars, template, sample)
            objective = result.get("Sharpe", -1000)  # Use Sharpe as objective
            X_samples.append([sample[p] for p in sorted(param_ranges.keys()) if isinstance(sample[p], (int, float))])
            y_samples.append(objective)
        except:
            continue
    
    if not y_samples:
        return {"error": "No successful evaluations during initialization"}
    
    # Convert to numpy arrays
    X = np.array(X_samples)
    y = np.array(y_samples)
    
    best_idx = np.argmax(y)
    best_x = X[best_idx]
    best_y = y[best_idx]
    
    # Simplified Bayesian optimization (without full GP implementation)
    # In a full implementation, would use scikit-learn GaussianProcessRegressor
    iteration_history = []
    
    for iteration in range(n_init, n_iterations):
        # Simple acquisition: random search with bias toward best regions
        candidate_samples = []
        candidate_scores = []
        
        for _ in range(20):  # Generate candidates
            # Bias toward best parameters with some exploration
            if random.random() < 0.7:  # Exploitation
                candidate = best_x + np.random.normal(0, 0.1 * np.std(X, axis=0))
            else:  # Exploration
                candidate = np.random.uniform(X.min(axis=0), X.max(axis=0))
            
            # Convert back to parameter dict
            param_dict = {}
            float_params = [p for p in sorted(param_ranges.keys()) if param_ranges[p][2] in ['int', 'float']]
            
            for i, param in enumerate(float_params):
                if i < len(candidate):
                    min_val, max_val, param_type = param_ranges[param]
                    val = np.clip(candidate[i], min_val, max_val)
                    param_dict[param] = int(val) if param_type == 'int' else float(val)
            
            # Add non-numeric parameters
            for param, (min_val, max_val, param_type) in param_ranges.items():
                if param not in param_dict:
                    if param_type == 'bool':
                        param_dict[param] = random.choice([True, False])
                    elif param_type == 'choice':
                        param_dict[param] = random.choice(max_val)
            
            # Evaluate candidate
            try:
                result = run_backtest(bars, template, param_dict)
                objective = result.get("Sharpe", -1000)
                candidate_samples.append((candidate, objective, param_dict))
                candidate_scores.append(objective)
            except:
                continue
        
        if candidate_samples:
            # Select best candidate
            best_candidate_idx = np.argmax(candidate_scores)
            best_candidate = candidate_samples[best_candidate_idx]
            
            # Update dataset
            X = np.vstack([X, best_candidate[0].reshape(1, -1)])
            y = np.append(y, best_candidate[1])
            
            # Update best if improved
            if best_candidate[1] > best_y:
                best_x = best_candidate[0]
                best_y = best_candidate[1]
                best_params = best_candidate[2]
            
            iteration_history.append({
                "iteration": iteration,
                "best_objective": float(best_y),
                "current_objective": float(best_candidate[1])
            })
    
    # Final evaluation
    final_result = run_backtest(bars, template, best_params)
    
    return {
        "optimized_params": best_params,
        "best_objective": float(best_y),
        "final_metrics": final_result,
        "iterations_run": len(iteration_history) + n_init,
        "iteration_history": iteration_history,
        "optimization_method": "bayesian",
        "acquisition_function": acquisition_function
    }

def run_particle_swarm_optimization(bars: list[dict], template: str, param_ranges: Dict[str, Tuple],
                                  n_particles: int = 20, n_iterations: int = 30,
                                  inertia: float = 0.7, cognitive: float = 1.4, 
                                  social: float = 1.4) -> Dict[str, Any]:
    """
    Particle Swarm Optimization for strategy parameters
    """
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    
    # Initialize particles
    particles = []
    global_best_position = None
    global_best_fitness = -np.inf
    
    # Create particles
    for _ in range(n_particles):
        position = {}
        velocity = {}
        
        for param, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'int':
                position[param] = random.randint(int(min_val), int(max_val))
                velocity[param] = random.uniform(-1, 1)
            elif param_type == 'float':
                position[param] = random.uniform(min_val, max_val)
                velocity[param] = random.uniform(-0.1, 0.1) * (max_val - min_val)
            elif param_type == 'bool':
                position[param] = random.choice([True, False])
                velocity[param] = 0  # Boolean parameters don't have velocity
            elif param_type == 'choice':
                position[param] = random.choice(max_val)
                velocity[param] = 0  # Choice parameters don't have velocity
        
        # Evaluate initial position
        try:
            result = run_backtest(bars, template, position)
            fitness = result.get("Sharpe", -1000)
        except:
            fitness = -1000
        
        particle = {
            "position": position.copy(),
            "velocity": velocity.copy(),
            "best_position": position.copy(),
            "best_fitness": fitness,
            "current_fitness": fitness
        }
        
        particles.append(particle)
        
        # Update global best
        if fitness > global_best_fitness:
            global_best_fitness = fitness
            global_best_position = position.copy()
    
    # PSO iterations
    iteration_history = []
    
    for iteration in range(n_iterations):
        for particle in particles:
            # Update velocity and position for numeric parameters
            for param, (min_val, max_val, param_type) in param_ranges.items():
                if param_type in ['int', 'float']:
                    r1, r2 = random.random(), random.random()
                    
                    cognitive_component = cognitive * r1 * (particle["best_position"][param] - particle["position"][param])
                    social_component = social * r2 * (global_best_position[param] - particle["position"][param])
                    
                    # Update velocity
                    particle["velocity"][param] = (
                        inertia * particle["velocity"][param] + 
                        cognitive_component + 
                        social_component
                    )
                    
                    # Update position
                    new_position = particle["position"][param] + particle["velocity"][param]
                    particle["position"][param] = np.clip(new_position, min_val, max_val)
                    
                    # Discretize for integer parameters
                    if param_type == 'int':
                        particle["position"][param] = int(round(particle["position"][param]))
                
                elif param_type in ['bool', 'choice']:
                    # Occasional random changes for discrete parameters
                    if random.random() < 0.1:
                        if param_type == 'bool':
                            particle["position"][param] = random.choice([True, False])
                        else:
                            particle["position"][param] = random.choice(max_val)
            
            # Evaluate new position
            try:
                result = run_backtest(bars, template, particle["position"])
                fitness = result.get("Sharpe", -1000)
                particle["current_fitness"] = fitness
                
                # Update personal best
                if fitness > particle["best_fitness"]:
                    particle["best_fitness"] = fitness
                    particle["best_position"] = particle["position"].copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle["position"].copy()
                    
            except:
                particle["current_fitness"] = -1000
        
        # Record iteration
        avg_fitness = np.mean([p["current_fitness"] for p in particles])
        iteration_history.append({
            "iteration": iteration,
            "best_fitness": float(global_best_fitness),
            "avg_fitness": float(avg_fitness)
        })
    
    # Final evaluation
    final_result = run_backtest(bars, template, global_best_position)
    
    return {
        "optimized_params": global_best_position,
        "best_fitness": float(global_best_fitness),
        "final_metrics": final_result,
        "iterations_run": n_iterations,
        "iteration_history": iteration_history,
        "optimization_method": "particle_swarm",
        "n_particles": n_particles,
        "final_swarm_diversity": _calculate_swarm_diversity(particles, param_ranges)
    }

def _calculate_swarm_diversity(particles: List[Dict], param_ranges: Dict) -> float:
    """Calculate diversity of particle swarm"""
    if len(particles) < 2:
        return 0.0
    
    total_diversity = 0.0
    param_count = 0
    
    for param, (min_val, max_val, param_type) in param_ranges.items():
        if param_type in ['int', 'float']:
            values = [p["position"][param] for p in particles]
            if max_val != min_val:
                diversity = np.std(values) / (max_val - min_val)
                total_diversity += diversity
                param_count += 1
    
    return total_diversity / max(1, param_count)

def run_multi_objective_optimization(bars: list[dict], template: str, param_ranges: Dict[str, Tuple],
                                   objectives: List[str] = None, n_iterations: int = 50,
                                   population_size: int = 30) -> Dict[str, Any]:
    """
    Multi-objective optimization using NSGA-II inspired approach
    """
    if objectives is None:
        objectives = ["Sharpe", "MaxDrawdown", "Calmar"]
    
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    
    # Initialize population
    population = _initialize_population(param_ranges, population_size)
    
    # Evaluate population
    for individual in population:
        try:
            result = run_backtest(bars, template, individual.genes)
            individual.metrics = result
            
            # Multi-objective fitness (Pareto ranking would be ideal)
            fitness_values = []
            for obj in objectives:
                if obj == "MaxDrawdown":
                    # For drawdown, lower is better
                    fitness_values.append(-result.get(obj, 100))
                else:
                    fitness_values.append(result.get(obj, 0))
            
            # Simple weighted combination (in practice, would use proper Pareto ranking)
            individual.fitness = np.mean(fitness_values)
            individual.objective_values = fitness_values
            
        except:
            individual.fitness = -1000
            individual.objective_values = [-1000] * len(objectives)
    
    # Evolution
    best_individuals_history = []
    pareto_front_history = []
    
    for generation in range(n_iterations):
        # Simple selection (would use NSGA-II selection in full implementation)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep best half
        survivors = population[:population_size//2]
        
        # Generate offspring
        offspring = []
        while len(offspring) < population_size - len(survivors):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            # Crossover
            child_genes = {}
            for param in parent1.genes:
                child_genes[param] = random.choice([parent1.genes[param], parent2.genes[param]])
            
            # Mutation
            for param, (min_val, max_val, param_type) in param_ranges.items():
                if random.random() < 0.1:  # Mutation rate
                    if param_type == 'int':
                        child_genes[param] = random.randint(int(min_val), int(max_val))
                    elif param_type == 'float':
                        child_genes[param] = random.uniform(min_val, max_val)
                    elif param_type == 'bool':
                        child_genes[param] = random.choice([True, False])
                    elif param_type == 'choice':
                        child_genes[param] = random.choice(max_val)
            
            child = Individual(genes=child_genes, generation=generation)
            
            # Evaluate child
            try:
                result = run_backtest(bars, template, child.genes)
                child.metrics = result
                
                fitness_values = []
                for obj in objectives:
                    if obj == "MaxDrawdown":
                        fitness_values.append(-result.get(obj, 100))
                    else:
                        fitness_values.append(result.get(obj, 0))
                
                child.fitness = np.mean(fitness_values)
                child.objective_values = fitness_values
                
            except:
                child.fitness = -1000
                child.objective_values = [-1000] * len(objectives)
            
            offspring.append(child)
        
        # New population
        population = survivors + offspring
        
        # Record best individual
        best_individual = max(population, key=lambda x: x.fitness)
        best_individuals_history.append({
            "generation": generation,
            "fitness": best_individual.fitness,
            "objectives": dict(zip(objectives, best_individual.objective_values)),
            "params": best_individual.genes.copy()
        })
        
        # Simple Pareto front approximation (top 10% individuals)
        top_individuals = sorted(population, key=lambda x: x.fitness, reverse=True)[:max(3, population_size//10)]
        pareto_front_history.append([
            {
                "objectives": dict(zip(objectives, ind.objective_values)),
                "params": ind.genes.copy()
            }
            for ind in top_individuals
        ])
    
    # Final results
    final_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    best_individual = final_population[0]
    
    return {
        "optimized_params": best_individual.genes,
        "best_fitness": best_individual.fitness,
        "best_objectives": dict(zip(objectives, best_individual.objective_values)),
        "final_metrics": best_individual.metrics,
        "pareto_front": pareto_front_history[-1] if pareto_front_history else [],
        "evolution_history": best_individuals_history,
        "optimization_method": "multi_objective",
        "objectives": objectives,
        "generations_run": n_iterations
    }

def run_performance_attribution_analysis(strategies: List[Dict], bars: list[dict],
                                       benchmark_strategy: str = None) -> Dict[str, Any]:
    """
    Analyze performance attribution across multiple strategies
    """
    results = {}
    strategy_results = []
    
    # Run all strategies
    for strategy_config in strategies:
        template = strategy_config["template"]
        params = strategy_config.get("params", {})
        name = strategy_config.get("name", template)
        
        try:
            result = run_backtest(bars, template, params)
            strategy_results.append({
                "name": name,
                "template": template,
                "params": params,
                "metrics": result
            })
        except Exception as e:
            strategy_results.append({
                "name": name,
                "template": template,
                "params": params,
                "metrics": {"error": str(e)}
            })
    
    # Calculate attribution metrics
    if strategy_results:
        successful_results = [r for r in strategy_results if "error" not in r["metrics"]]
        
        if successful_results:
            # Performance ranking
            performance_ranking = sorted(
                successful_results,
                key=lambda x: x["metrics"].get("Sharpe", -1000),
                reverse=True
            )
            
            # Risk-adjusted returns analysis
            returns_analysis = {}
            for result in successful_results:
                metrics = result["metrics"]
                returns_analysis[result["name"]] = {
                    "sharpe": metrics.get("Sharpe", 0),
                    "sortino": metrics.get("Sortino", 0),
                    "calmar": metrics.get("Calmar", 0),
                    "max_drawdown": metrics.get("MaxDrawdown", 0),
                    "volatility": metrics.get("volatility", 0),
                    "cagr": metrics.get("CAGR", 0),
                    "win_rate": metrics.get("WinRate", 0)
                }
            
            # Style analysis (simplified)
            style_analysis = _analyze_strategy_styles(successful_results)
            
            # Risk contribution analysis
            risk_analysis = _analyze_risk_contributions(successful_results)
            
            results = {
                "performance_ranking": [
                    {
                        "rank": i + 1,
                        "name": r["name"],
                        "sharpe": r["metrics"].get("Sharpe", 0),
                        "cagr": r["metrics"].get("CAGR", 0),
                        "max_drawdown": r["metrics"].get("MaxDrawdown", 0)
                    }
                    for i, r in enumerate(performance_ranking)
                ],
                "returns_analysis": returns_analysis,
                "style_analysis": style_analysis,
                "risk_analysis": risk_analysis,
                "summary_statistics": {
                    "best_sharpe": max(r["metrics"].get("Sharpe", 0) for r in successful_results),
                    "worst_drawdown": max(r["metrics"].get("MaxDrawdown", 0) for r in successful_results),
                    "avg_cagr": np.mean([r["metrics"].get("CAGR", 0) for r in successful_results]),
                    "avg_volatility": np.mean([r["metrics"].get("volatility", 0) for r in successful_results])
                },
                "successful_strategies": len(successful_results),
                "total_strategies": len(strategy_results)
            }
    
    return results

def _analyze_strategy_styles(strategy_results: List[Dict]) -> Dict[str, Any]:
    """Analyze strategy styles and characteristics"""
    styles = {
        "trend_following": 0,
        "mean_reversion": 0,
        "momentum": 0,
        "volatility_based": 0,
        "multi_factor": 0
    }
    
    # Simple style classification based on template names
    for result in strategy_results:
        template = result["template"].lower()
        
        if "trend" in template or "ma" in template or "ema" in template:
            styles["trend_following"] += 1
        elif "rsi" in template or "bb" in template or "bollinger" in template:
            styles["mean_reversion"] += 1
        elif "momentum" in template or "macd" in template:
            styles["momentum"] += 1
        elif "volatility" in template or "atr" in template:
            styles["volatility_based"] += 1
        elif "ensemble" in template or "adaptive" in template:
            styles["multi_factor"] += 1
    
    total = len(strategy_results)
    style_percentages = {k: (v/total)*100 if total > 0 else 0 for k, v in styles.items()}
    
    return {
        "style_counts": styles,
        "style_percentages": style_percentages,
        "dominant_style": max(style_percentages, key=style_percentages.get) if style_percentages else None
    }

def _analyze_risk_contributions(strategy_results: List[Dict]) -> Dict[str, Any]:
    """Analyze risk contributions and correlations"""
    risk_metrics = {}
    
    for result in strategy_results:
        name = result["name"]
        metrics = result["metrics"]
        
        risk_metrics[name] = {
            "volatility": metrics.get("volatility", 0),
            "max_drawdown": metrics.get("MaxDrawdown", 0),
            "var_95": metrics.get("VaR_95", 0),
            "downside_deviation": metrics.get("Sortino", 0),  # Proxy
            "tail_risk": metrics.get("VaR_99", 0)
        }
    
    # Calculate risk-adjusted performance
    risk_adjusted_scores = {}
    for name, risks in risk_metrics.items():
        # Simple risk score (lower is better)
        risk_score = (
            risks["volatility"] * 0.3 +
            risks["max_drawdown"] * 0.4 +
            abs(risks["var_95"]) * 0.3
        )
        risk_adjusted_scores[name] = risk_score
    
    return {
        "individual_risk_metrics": risk_metrics,
        "risk_adjusted_scores": risk_adjusted_scores,
        "lowest_risk_strategy": min(risk_adjusted_scores, key=risk_adjusted_scores.get) if risk_adjusted_scores else None,
        "highest_risk_strategy": max(risk_adjusted_scores, key=risk_adjusted_scores.get) if risk_adjusted_scores else None
    }