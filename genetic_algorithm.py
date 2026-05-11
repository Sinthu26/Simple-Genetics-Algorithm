"""
CP468 - Artificial Intelligence
Term Project: Simple Genetic Algorithm (SGA)
Test Functions: De Jong's Sphere Function, Rosenbrock's Valley, Himmelblau's Function
"""

import random
import math
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# CONFIGURATION — tweak these to run experiments
# ─────────────────────────────────────────────
POPULATION_SIZE   = 100       # Number of individuals in the population
NUM_GENERATIONS   = 200       # How many generations to run
MUTATION_RATE     = 0.01      # Probability of a bit flipping (0.0 - 1.0)
CROSSOVER_RATE    = 0.8       # Probability of crossover happening (0.0 - 1.0)
NUM_VARIABLES     = 3         # Number of dimensions (De Jong's Sphere uses 3)
LOWER_BOUND       = -5.12     # Search space lower bound (standard for De Jong's)
UPPER_BOUND       = 5.12      # Search space upper bound
BITS_PER_VAR      = 16        # Binary bits used to encode each variable
CHROMOSOME_LENGTH = NUM_VARIABLES * BITS_PER_VAR
NUM_RUNS          = 5         # How many independent runs to average results over


# ─────────────────────────────────────────────
# ENCODING & DECODING
# ─────────────────────────────────────────────

def decode_chromosome(chromosome):
    """Convert a binary chromosome (list of 0s and 1s) into real-valued variables."""
    variables = []
    for i in range(NUM_VARIABLES):
        # Extract the bits for this variable
        bits = chromosome[i * BITS_PER_VAR : (i + 1) * BITS_PER_VAR]
        # Convert binary to integer
        integer_val = int("".join(str(b) for b in bits), 2)
        # Map integer to the real-valued range [LOWER_BOUND, UPPER_BOUND]
        max_int = 2 ** BITS_PER_VAR - 1
        real_val = LOWER_BOUND + (integer_val / max_int) * (UPPER_BOUND - LOWER_BOUND)
        variables.append(real_val)
    return variables


def random_chromosome():
    """Generate a random binary chromosome."""
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]


# ─────────────────────────────────────────────
# OBJECTIVE & FITNESS FUNCTIONS
# ─────────────────────────────────────────────

def de_jong_sphere(variables):
    """
    De Jong's Sphere Function - f(x) = sum(xi^2)
    Global minimum is 0 at x = (0, 0, 0)
    Search space: [-5.12, 5.12]
    """
    return sum(x ** 2 for x in variables)


def rosenbrock_valley(variables):
    """
    Rosenbrock's Valley Function
    f(x) = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
    Global minimum is 0 at x = (1, 1, ..., 1)
    Search space: [-2.048, 2.048]
    """
    result = 0.0
    for i in range(len(variables) - 1):
        result += 100.0 * (variables[i+1] - variables[i] ** 2) ** 2 + (1 - variables[i]) ** 2
    return result


# FIX 2: Added Himmelblau's Function, which was required by the project spec
# but was missing from the original code.
# Note: Himmelblau is a 2D function — set NUM_VARIABLES = 2 when running it.
def himmelblau_function(variables):
    """
    Himmelblau's Function - f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Has 4 global minima of 0 at approximately:
      (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    Search space: [-5, 5]
    """
    x, y = variables[0], variables[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def fitness(chromosome, active_function="sphere"):
    """
    Convert the objective value to a fitness value.
    Since we minimize, fitness = 1 / (1 + objective_value)
    Higher fitness = better solution.
    """
    variables = decode_chromosome(chromosome)
    if active_function == "rosenbrock":
        obj_value = rosenbrock_valley(variables)
    elif active_function == "himmelblau":
        obj_value = himmelblau_function(variables)
    else:
        obj_value = de_jong_sphere(variables)
    return 1.0 / (1.0 + obj_value)


# ─────────────────────────────────────────────
# SELECTION — Roulette Wheel (Fitness Proportionate)
# ─────────────────────────────────────────────

def roulette_wheel_selection(population, fitness_values):
    """
    Select one individual using roulette wheel selection.
    Individuals with higher fitness have a proportionally higher chance of being selected.
    """
    total_fitness = sum(fitness_values)
    pick = random.uniform(0, total_fitness)
    running = 0
    for individual, fit in zip(population, fitness_values):
        running += fit
        if running >= pick:
            return individual
    return population[-1]  # Fallback


# ─────────────────────────────────────────────
# CROSSOVER — Single-Point Crossover
# ─────────────────────────────────────────────

def single_point_crossover(parent1, parent2):
    """
    Perform single-point crossover between two parents.
    A random cut point is chosen; genes are swapped after that point.
    """
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    # No crossover — children are copies of parents
    return parent1[:], parent2[:]


# ─────────────────────────────────────────────
# MUTATION — Bit-Flip Mutation
# ─────────────────────────────────────────────

def mutate(chromosome):
    """
    Flip each bit with probability MUTATION_RATE.
    """
    return [1 - bit if random.random() < MUTATION_RATE else bit for bit in chromosome]


# ─────────────────────────────────────────────
# MAIN GENETIC ALGORITHM LOOP
# ─────────────────────────────────────────────

def run_genetic_algorithm(active_function="sphere"):
    """
    Run one full execution of the Simple Genetic Algorithm.
    Returns the best fitness per generation and the best solution found.
    Termination criterion: fixed number of generations (NUM_GENERATIONS).
    """
    # Step 1: Initialize random population
    population = [random_chromosome() for _ in range(POPULATION_SIZE)]
 
    best_fitness_per_gen = []
    best_solution = None
    best_fitness_overall = -1
 
    for generation in range(NUM_GENERATIONS):
        # Step 2: Evaluate fitness of each individual
        fitness_values = [fitness(ind, active_function) for ind in population]
 
        # Track the best individual this generation
        best_idx = fitness_values.index(max(fitness_values))
        best_gen_fitness = fitness_values[best_idx]
        best_fitness_per_gen.append(best_gen_fitness)
 
        if best_gen_fitness > best_fitness_overall:
            best_fitness_overall = best_gen_fitness
            best_solution = population[best_idx][:]
 
        # Step 3: Build the next generation
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            # Crossover
            child1, child2 = single_point_crossover(parent1, parent2)
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
 
        # Replace old population
        population = new_population[:POPULATION_SIZE]
 
    return best_fitness_per_gen, best_solution, best_fitness_overall


# ─────────────────────────────────────────────
# EXPERIMENT: PARAMETER VARIATIONS
# ─────────────────────────────────────────────

# FIX 3: run_experiment now accepts bounds/dimensions and saves+restores all
# globals so that state is not corrupted between experiments.
def run_experiment(label, pop_size, mut_rate, cross_rate, active_function="sphere", num_vars=3, low=-5.12, high=5.12, runs=NUM_RUNS):
    """
    Run the GA multiple times with given parameters and return averaged results.
    All globals are saved before the experiment and restored afterward.
    """
    global POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE
    global NUM_VARIABLES, LOWER_BOUND, UPPER_BOUND, CHROMOSOME_LENGTH
 
    # Save originals
    orig_pop   = POPULATION_SIZE
    orig_mut   = MUTATION_RATE
    orig_cross = CROSSOVER_RATE
    orig_vars  = NUM_VARIABLES
    orig_low   = LOWER_BOUND
    orig_high  = UPPER_BOUND
    orig_chrom = CHROMOSOME_LENGTH
 
    # Apply experiment settings
    POPULATION_SIZE   = pop_size
    MUTATION_RATE     = mut_rate
    CROSSOVER_RATE    = cross_rate
    NUM_VARIABLES     = num_vars
    LOWER_BOUND       = low
    UPPER_BOUND       = high
    CHROMOSOME_LENGTH = num_vars * BITS_PER_VAR
 
    all_curves = []
    all_best   = []
 
    for r in range(runs):
        curve, _, best = run_genetic_algorithm(active_function)
        all_curves.append(curve)
        all_best.append(best)
 
    # Average convergence curve across all runs
    avg_curve = [
        sum(all_curves[r][g] for r in range(runs)) / runs
        for g in range(NUM_GENERATIONS)
    ]
    avg_best = sum(all_best) / runs
    print(f"  [{label}] Avg best fitness over {runs} runs: {avg_best:.6f}")
 
    # Restore originals
    POPULATION_SIZE   = orig_pop
    MUTATION_RATE     = orig_mut
    CROSSOVER_RATE    = orig_cross
    NUM_VARIABLES     = orig_vars
    LOWER_BOUND       = orig_low
    UPPER_BOUND       = orig_high
    CHROMOSOME_LENGTH = orig_chrom
 
    return avg_curve, avg_best


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_results(experiments, title="SGA Convergence on De Jong's Sphere Function", filename="convergence_plot.png"):
    """
    Plot convergence curves for all experiments on one graph.
    experiments: list of (label, avg_curve)
    """
    plt.figure(figsize=(10, 6))
    for label, curve in experiments:
        plt.plot(range(NUM_GENERATIONS), curve, label=label)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (higher = better)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nPlot saved as {filename}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  CP468 Term Project — Simple Genetic Algorithm")
    print("  Test Functions: De Jong's Sphere, Rosenbrock's Valley,")
    print("                  Himmelblau's Function")
    print("=" * 55)
 
    start_time = time.time()
 
    experiments_to_plot = []
 
    # ── De Jong's Sphere Experiments ──────────────────────
    print("\n--- De Jong's Sphere Function Experiments ---")
 
    # Experiment 1: Baseline
    print("\nExperiment 1: Baseline (pop=100, mut=0.01, cross=0.8)")
    curve1, best1 = run_experiment("Baseline (pop=100, mut=0.01)", 100, 0.01, 0.8, active_function="sphere", num_vars=3, low=-5.12, high=5.12)
    experiments_to_plot.append(("Baseline (pop=100, mut=0.01)", curve1))
 
    # Experiment 2: Larger population
    print("\nExperiment 2: Large population (pop=200, mut=0.01, cross=0.8)")
    curve2, best2 = run_experiment("Large pop (pop=200, mut=0.01)", 200, 0.01, 0.8, active_function="sphere", num_vars=3, low=-5.12, high=5.12)
    experiments_to_plot.append(("Large pop (pop=200, mut=0.01)", curve2))
 
    # Experiment 3: Higher mutation rate
    print("\nExperiment 3: High mutation (pop=100, mut=0.05, cross=0.8)")
    curve3, best3 = run_experiment("High mutation (pop=100, mut=0.05)", 100, 0.05, 0.8, active_function="sphere", num_vars=3, low=-5.12, high=5.12)
    experiments_to_plot.append(("High mutation (pop=100, mut=0.05)", curve3))
 
    # Experiment 4: Low crossover rate
    print("\nExperiment 4: Low crossover (pop=100, mut=0.01, cross=0.4)")
    curve4, best4 = run_experiment("Low crossover (pop=100, cross=0.4)", 100, 0.01, 0.4, active_function="sphere", num_vars=3, low=-5.12, high=5.12)
    experiments_to_plot.append(("Low crossover (pop=100, cross=0.4)", curve4))
 
    plot_results(experiments_to_plot, title="SGA Convergence on De Jong's Sphere Function", filename="convergence_plot.png")
 
    # ── Rosenbrock's Valley ───────────────────────────────
    print("\n--- Rosenbrock's Valley Experiment ---")
 
    print("\nRosenbrock Baseline (pop=100, mut=0.01, cross=0.8)")
    curveR, bestR = run_experiment("Rosenbrock Baseline", 100, 0.01, 0.8, active_function="rosenbrock", num_vars=3, low=-2.048, high=2.048)
    plot_results([("Rosenbrock Baseline", curveR)], title="SGA Convergence on Rosenbrock's Valley", filename="convergence_rosenbrock.png")
 
    # ── Himmelblau's Function ─────────────────────────────
    # Himmelblau is 2D — uses num_vars=2 and bounds [-5, 5]
    print("\n--- Himmelblau's Function Experiment ---")
 
    print("\nHimmelblau Baseline (pop=100, mut=0.01, cross=0.8)")
    curveH, bestH = run_experiment("Himmelblau Baseline", 100, 0.01, 0.8, active_function="himmelblau", num_vars=2, low=-5.0, high=5.0)
    plot_results([("Himmelblau Baseline", curveH)], title="SGA Convergence on Himmelblau's Function", filename="convergence_himmelblau.png")
 
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
 