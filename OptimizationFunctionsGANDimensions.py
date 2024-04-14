import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing
import time

class Individual:
    def __init__(self, alleles, cromosome):
        self._alleles = alleles
        self._cromosome = cromosome
        self._fitness_score = 0

class GeneticAlgorithm:
    def __init__(self, population, alleles, generations, mutation_rate, problem, is_maximization_problem = False, verbose_enabled = False):
        self._population = population
        self._alleles = alleles
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._problem = problem
        self._is_maximization_problem = is_maximization_problem
        self._individuals = np.array([])
        self._current_generation = 0
        self._historical_chart = np.zeros((self._generations // 100, 2))
        self._verbose_enabled = verbose_enabled

    def run(self):
        self.create_individuals()
        self._best_individual = self._individuals[0]
        self._current_generation = 1

        while self._current_generation <= self._generations:
            self.evaluate_individuals()
            self.get_best_individual()
            self.add_to_historical_chart()
            children = np.array([])

            while len(children) < len(self._individuals):
                parent_1 = self.roulette()
                parent_2 = self.roulette()

                while parent_1 == parent_2:
                    parent_2 = self.roulette()

                child_1, child_2 = self.crossover(self._individuals[parent_1], self._individuals[parent_2])
                children = np.append(children, [child_1])
                children = np.append(children, [child_2])

            self.mutate(children)
            self._individuals = np.copy(children)

            if self._verbose_enabled and self._current_generation % 100 == 0:
                print(f'Generación: {self._current_generation} Mejor Histórico: \
{self._best_individual._cromosome} {self._best_individual._fitness_score :.5f}')
            self._current_generation += 1

    def create_individuals(self):
        values_range = self._problem.get_upper_bound() - self._problem.get_lower_bound()

        for individual in range(self._population):
            genetic_data = np.random.random(size = self._alleles)
            cromosome = self._problem.get_lower_bound() + genetic_data * values_range
            individual = Individual(self._alleles, cromosome)
            self._individuals = np.append(self._individuals, [individual])

    def evaluate_individuals(self):
        for individual in self._individuals:
            individual._fitness_score = self._problem.get_fitness_score(individual._cromosome)

            if not self._is_maximization_problem:
                individual._fitness_score *= -1

    def roulette(self):
        initial_fitness_sum = np.sum([individual._fitness_score for individual in self._individuals])
        threshold = np.random.randint(np.abs(initial_fitness_sum + 1), dtype = np.int64)

        if initial_fitness_sum < 0:
            threshold *= -1

        selected_individual = 0
        fitness_sum = self._individuals[selected_individual]._fitness_score
        if initial_fitness_sum < 0:
            while fitness_sum > threshold and selected_individual < len(self._individuals) - 1:
                selected_individual += 1
                fitness_sum += self._individuals[selected_individual]._fitness_score
        else:
            while fitness_sum < threshold and selected_individual < len(self._individuals) - 1:
                selected_individual += 1
                fitness_sum += self._individuals[selected_individual]._fitness_score

        return selected_individual

    def crossover(self, parent_1, parent_2):
        child_1 = copy.deepcopy(parent_1)
        child_2 = copy.deepcopy(parent_2)

        maximum_crossover_point = self._alleles - 1
        crossover_point = np.random.randint(maximum_crossover_point) + 1
        child_1._cromosome[crossover_point:], child_2._cromosome[crossover_point:] = child_2._cromosome[crossover_point:], child_1._cromosome[crossover_point:]

        return child_1, child_2

    def mutate(self, children):
        values_range = (self._problem.get_upper_bound() - self._problem.get_lower_bound())

        for child in children:
            for allele in range(len(child._cromosome)):
                if np.random.rand() < self._mutation_rate:
                    child._cromosome[allele] = self._problem.get_lower_bound() + np.random.random() * values_range

    def get_best_individual(self):
        for individual in self._individuals:
            if individual._fitness_score > self._best_individual._fitness_score:
                self._best_individual = copy.deepcopy(individual)

    def get_best_solution(self):
        return self._best_individual

    def get_historical_chart(self):
        return self._historical_chart

    def add_to_historical_chart(self):
      if self._current_generation % 100 != 0:
        return


      self._historical_chart[self._current_generation // 100 - 1, 0] = self._current_generation
      self._historical_chart[self._current_generation // 100 - 1, 1] = self._best_individual._fitness_score

class OptimizationTestFunction:
    def __init__(self, function, lower_bound, upper_bound):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._function = function 

    def get_lower_bound(self):
        return self._lower_bound

    def get_upper_bound(self):
        return self._upper_bound

    def evaluate_function_one_variable(self, variable_1):
        return self._function(variable_1)

    def evaluate_function_two_variables(self, variable_1, variable_2):
        return self._function(variable_1, variable_2)

    def get_fitness_score(self, cromosome):
        fitness_score = 0

        for allele in cromosome:
          fitness_score += self.evaluate_function_one_Variable(allele)

        return fitness_score


class Sphere(OptimizationTestFunction):
    def __init__(self, lower_bound = -5.12, upper_bound = 5.12):
        super().__init__(lambda x : x ** 2, lower_bound, upper_bound)

    def get_fitness_score(self, cromosome):
        fitness_score = 0

        for allele in cromosome:
          fitness_score += self.evaluate_function_one_variable(allele)

        return fitness_score

class Rosenbrock(OptimizationTestFunction):
  def __init__(self, lower_bound = -2.048, upper_bound = 2.048):
    super().__init__(lambda x, y: 100 * ((y - (x ** 2)) ** 2) + ((x - 1) ** 2), lower_bound, upper_bound)

  def get_fitness_score(self, cromosome):
    fitness_score = 0

    for allele in range(len(cromosome) - 1):
      fitness_score += self.evaluate_function_two_variables(cromosome[allele], cromosome[allele + 1])

    return fitness_score

class Rastrigin(OptimizationTestFunction):
  def __init__(self, lower_bound = -5.12, upper_bound = 5.12):
    super().__init__(lambda x : x ** 2 - (10 * math.cos(2 * math.pi * x)), lower_bound, upper_bound)

  def get_fitness_score(self, cromosome):
    fitness_score = 0

    for allele in cromosome:
      fitness_score += self.evaluate_function_one_variable(allele)

    return 10 * len(cromosome) + fitness_score

class Quartic(OptimizationTestFunction):
  def __init__(self, lower_bound = -1.28, upper_bound = 1.28):
    super().__init__(lambda i, x : i * (x ** 4), lower_bound, upper_bound)

  def get_fitness_score(self, cromosome):
    fitness_score = 0

    for allele in range(len(cromosome)):
      fitness_score += self.evaluate_function_two_variables(allele + 1, cromosome[allele])

    return fitness_score

class GeneticAlgorithmConfiguration:
  def __init__(self, population, alleles, generations, mutation_rate, is_maximization_problem = False, verbose_enabled = False):
    self._population = population
    self._alleles = alleles
    self._generations = generations
    self._mutation_rate = mutation_rate
    self._is_maximization_problem = is_maximization_problem
    self._verbose_enabled = verbose_enabled

  def get_population(self):
    return self._population

  def get_alleles(self):
    return self._alleles

  def get_generations(self):
    return self._generations

  def get_mutation_rate(self):
    return self._mutation_rate

  def is_maximization(self):
    return self._is_maximization_problem

  def set_population(self, population):
    self._population = population

  def set_alleles(self, alleles):
    self._alleles = alleles

  def set_generations(self, generations):
    self._generations = generations

  def set_mutation_rate(self, mutation_rate):
    self._mutation_rate = mutation_rate

  def set_is_maximization(self, is_maximization_problem = False):
    self._is_maximization_problem = is_maximization_problem

def execute_optimization_function_and_get_historical(problem, genetic_algorithm_configuration):
    genetic_algorithm = GeneticAlgorithm(genetic_algorithm_configuration.get_population(), genetic_algorithm_configuration.get_alleles(), genetic_algorithm_configuration.get_generations(), genetic_algorithm_configuration.get_mutation_rate(), problem, genetic_algorithm_configuration.is_maximization())
    genetic_algorithm.run()
    return genetic_algorithm.get_historical_chart()

def plot_data(plot_chart, historical_data, title = "GA - Function de optimizacion - Minimo Global", x_label = "Generacion", y_label = "Minimo", line_label = "historico"):
    plot_chart.plot(historical_data[:, 0], historical_data[:, 1], label = line_label)
    plot_chart.set_xlabel(x_label)
    plot_chart.set_ylabel(y_label)
    plot_chart.set_title(title)
    plot_chart.legend()

def execute_sphere(genetic_algorithm_configuration):
    return execute_optimization_function_and_get_historical(Sphere(), genetic_algorithm_configuration)

def execute_rosenbrock(genetic_algorithm_configuration):
    return execute_optimization_function_and_get_historical(Rosenbrock(), genetic_algorithm_configuration)

def execute_rastrigin(genetic_algorithm_configuration):
    return execute_optimization_function_and_get_historical(Rastrigin(), genetic_algorithm_configuration)

def execute_quartic(genetic_algorithm_configuration):
    return execute_optimization_function_and_get_historical(Quartic(), genetic_algorithm_configuration)

def benchmark_optimization_test_function(optimization_function, genetic_algorithm_configuration, cycles, historical_data_x_label, historical_data_y_label):
    cycles_historical_data = [] * cycles
    for cycle in range(cycles):
      cycles_historical_data.append(optimization_function(genetic_algorithm_configuration))
  
    for historical in cycles_historical_data:
      for generation in range(len(historical)):
          historical_data_x_label[generation] = historical[generation, 0]
          historical_data_y_label[generation] += historical[generation, 1]

      for generation in range(len(historical)):
          historical_data_y_label[generation] = historical_data_y_label[generation] / cycles
    
def benchmark_multiple_dimensions(optimization_function, genetic_algorithm_configuration_2d, genetic_algorithm_configuration_4d, genetic_algorithm_configuration_8d, cycles = 5, benchmark_name = "Sphere"):
    benchmarks = []
    multiprocessing_manager = multiprocessing.Manager()
    
    historical_data_2d_x_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_2d.get_generations() // 100))
    historical_data_2d_y_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_2d.get_generations() // 100))
    benchmarks.append(multiprocessing.Process(target = benchmark_optimization_test_function, args = (optimization_function, genetic_algorithm_configuration_2d, cycles, historical_data_2d_x_label, historical_data_2d_y_label)))

    historical_data_4d_x_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_4d.get_generations() // 100))
    historical_data_4d_y_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_4d.get_generations() // 100))
    benchmarks.append(multiprocessing.Process(target = benchmark_optimization_test_function, args = (optimization_function, genetic_algorithm_configuration_4d, cycles, historical_data_4d_x_label, historical_data_4d_y_label)))
    
    historical_data_8d_x_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_8d.get_generations() // 100))
    historical_data_8d_y_label = multiprocessing_manager.list(range(genetic_algorithm_configuration_8d.get_generations() // 100))
    benchmarks.append(multiprocessing.Process(target = benchmark_optimization_test_function, args = (optimization_function, genetic_algorithm_configuration_8d, cycles, historical_data_8d_x_label, historical_data_8d_y_label)))

    for benchmark in benchmarks:
      benchmark.start()

    for benchmark in benchmarks:
      benchmark.join()
    
    historical_data_2d = np.zeros((genetic_algorithm_configuration_2d.get_generations() // 100, 2))
    historical_data_4d = np.zeros((genetic_algorithm_configuration_4d.get_generations() // 100, 2))
    historical_data_8d = np.zeros((genetic_algorithm_configuration_8d.get_generations() // 100, 2))
    
    for generation in range(len(historical_data_2d)):
      historical_data_2d[generation, 0] = historical_data_2d_x_label[generation]
      historical_data_2d[generation, 1] = historical_data_2d_y_label[generation]

    for generation in range(len(historical_data_4d)):
      historical_data_4d[generation, 0] = historical_data_4d_x_label[generation]
      historical_data_4d[generation, 1] = historical_data_4d_y_label[generation]

    for generation in range(len(historical_data_8d)):
      historical_data_8d[generation, 0] = historical_data_8d_x_label[generation]
      historical_data_8d[generation, 1] = historical_data_8d_y_label[generation]
    
    type_problem_2d = "Maximo Global" if genetic_algorithm_configuration_2d.is_maximization() else "Minimo Global"
    type_problem_4d = "Maximo Global" if genetic_algorithm_configuration_4d.is_maximization() else "Minimo Global"
    type_problem_8d = "Maximo Global" if genetic_algorithm_configuration_8d.is_maximization() else "Minimo Global"

    title_2d = 'GA - Funcion de optimizacion ' + benchmark_name  + ' - 2D - ' + type_problem_2d
    title_4d = 'GA - Funcion de optimizacion ' + benchmark_name  + ' - 4D - ' + type_problem_4d
    title_8d = 'GA - Funcion de optimizacion ' + benchmark_name  + ' - 8D - ' + type_problem_8d
    title_all = 'GA - Funcion de optimizacion ' + benchmark_name  + ' - Todos'

    plot_2d = plt.subplot2grid((2, 2), (0, 0)) 
    plot_4d = plt.subplot2grid((2, 2), (0, 1)) 
    plot_8d = plt.subplot2grid((2, 2), (1, 0))
    plot_all = plt.subplot2grid((2, 2), (1, 1))
    
    plot_data(plot_2d, historical_data_2d, title_2d, "Generacion", type_problem_2d, "2D")
    plot_data(plot_4d, historical_data_4d, title_4d, "Generacion", type_problem_4d, "4D")
    plot_data(plot_8d, historical_data_8d, title_8d, "Generacion", type_problem_8d, "8D")
    plot_data(plot_all, historical_data_2d, title_all, "Generacion", type_problem_2d, "2D")
    plot_data(plot_all, historical_data_4d, title_all, "Generacion", type_problem_2d, "4D")
    plot_data(plot_all, historical_data_8d, title_all, "Generacion", type_problem_2d, "8D")
    plt.show()

if __name__ == '__main__':
    genetic_algorithm_configuration_2d = GeneticAlgorithmConfiguration(64, 2, 2000, 0.02, False, False)
    genetic_algorithm_configuration_4d = GeneticAlgorithmConfiguration(64, 4, 2000, 0.02, False, False)
    genetic_algorithm_configuration_8d = GeneticAlgorithmConfiguration(64, 8, 2000, 0.02, False, False)
    number_of_cycles = 5

    start = time.time()
    print(f"Ejecutando {number_of_cycles} ciclo(s) para funcion Sphere en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s)")
    benchmark_multiple_dimensions(execute_sphere, genetic_algorithm_configuration_2d, genetic_algorithm_configuration_4d, genetic_algorithm_configuration_8d, number_of_cycles, "Sphere")
    end = time.time()
    print(f"Ejecutados {number_of_cycles} ciclo(s) para funcion Sphere en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s). Tiempo total: {end - start}")

    start = time.time()
    print(f"Ejecutando {number_of_cycles} ciclo(s) para funcion Rosenbrock en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s)")
    benchmark_multiple_dimensions(execute_rosenbrock, genetic_algorithm_configuration_2d, genetic_algorithm_configuration_4d, genetic_algorithm_configuration_8d, number_of_cycles, "Rosenbrock")
    end = time.time()
    print(f"Ejecutados {number_of_cycles} ciclo(s) para funcion Rosenbrock en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s). Tiempo total: {end - start}")
    
    start = time.time()
    print(f"Ejecutando {number_of_cycles} ciclo(s) para funcion Rastrigin en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s)")
    benchmark_multiple_dimensions(execute_rastrigin, genetic_algorithm_configuration_2d, genetic_algorithm_configuration_4d, genetic_algorithm_configuration_8d, number_of_cycles, "Rastrigin")
    end = time.time()
    print(f"Ejecutados {number_of_cycles} ciclo(s) para funcion Rastrigin en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s). Tiempo total: {end - start}")

    start = time.time()
    print(f"Ejecutando {number_of_cycles} ciclo(s) para funcion Quartic en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s)")
    benchmark_multiple_dimensions(execute_quartic, genetic_algorithm_configuration_2d, genetic_algorithm_configuration_4d, genetic_algorithm_configuration_8d, number_of_cycles, "Quartic")
    end = time.time()
    print(f"Ejecutados {number_of_cycles} ciclo(s) para funcion Quartic en 2, 4 y 8 dimensiones. Total: {number_of_cycles * 3} ciclo(s). Tiempo total: {end - start}")
