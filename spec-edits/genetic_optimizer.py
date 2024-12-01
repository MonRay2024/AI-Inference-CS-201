from typing import List, Callable
import random
import numpy as np

class GeneticCodeOptimizer:
    def __init__(self, population_size=100, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
    def optimize(self, code: str, fitness_func: Callable,
                generations: int = 50) -> str:
        """Optimize code using genetic algorithm"""
        population = self._initialize_population(code)
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness_func(ind) for ind in population]
            
            # Select parents
            parents = self._select_parents(population, fitness_scores)
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
                
            population = new_population
            
        # Return best individual
        best_idx = np.argmax([fitness_func(ind) for ind in population])
        return population[best_idx]
    
    def _initialize_population(self, code: str) -> List[str]:
        """Initialize population with variations of original code"""
        population = [code]
        for _ in range(self.population_size - 1):
            variant = self._create_variant(code)
            population.append(variant)
        return population 