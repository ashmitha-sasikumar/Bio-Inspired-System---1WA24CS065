import random
import numpy as np

# ---------------------------------------------------------
# Traffic Signal Timing Optimization using Genetic Algorithm
# ---------------------------------------------------------

# Number of signal phases
NUM_PHASES = 4

# Traffic demand (vehicles/hour) for each phase
DEMAND = np.array([600, 300, 800, 400])

# Saturation flow rate (vehicles/hour) for each phase
SATURATION_FLOW = np.array([1800, 1800, 1800, 1800])

# Minimum and maximum green time for each phase (seconds)
MIN_GREEN = np.array([10, 10, 10, 10])
MAX_GREEN = np.array([60, 60, 60, 60])

# Yellow + all-red lost time per phase
LOST_TIME = 4

# Desired cycle time range
MIN_CYCLE = 60
MAX_CYCLE = 180


# ---------------------------------------------------------
# Fitness Function
# ---------------------------------------------------------

def calculate_delay(green_times):
    """
    Estimate total traffic delay for a signal timing plan.

    green_times: list/array containing green time for each phase
    """

    green_times = np.array(green_times)

    # Total cycle length
    cycle_time = np.sum(green_times) + NUM_PHASES * LOST_TIME

    total_delay = 0

    for i in range(NUM_PHASES):

        # Green ratio
        green_ratio = green_times[i] / cycle_time

        # Degree of saturation
        capacity = SATURATION_FLOW[i] * green_ratio
        demand = DEMAND[i]

        # Avoid division by zero
        if capacity <= 0:
            return 1e9

        # Queue/delay approximation
        if demand >= capacity:
            # Oversaturated phase -> very large penalty
            delay = 100000 * (demand - capacity + 1)
        else:
            # Approximate delay
            v_c = demand / capacity

            delay = (
                demand
                * cycle_time
                * (1 - green_ratio) ** 2
                / (2 * (1 - min(v_c, 0.99)))
            )

        total_delay += delay

    # Penalize cycles outside desired range
    if cycle_time < MIN_CYCLE:
        total_delay += 10000 * (MIN_CYCLE - cycle_time)

    if cycle_time > MAX_CYCLE:
        total_delay += 10000 * (cycle_time - MAX_CYCLE)

    return total_delay


# ---------------------------------------------------------
# Create Initial Population
# ---------------------------------------------------------

def create_individual():
    return [
        random.randint(MIN_GREEN[i], MAX_GREEN[i])
        for i in range(NUM_PHASES)
    ]


def create_population(population_size):
    return [
        create_individual()
        for _ in range(population_size)
    ]


# ---------------------------------------------------------
# Selection
# ---------------------------------------------------------

def tournament_selection(population, fitnesses, tournament_size=3):

    selected = random.sample(
        range(len(population)),
        tournament_size
    )

    winner = min(
        selected,
        key=lambda i: fitnesses[i]
    )

    return population[winner].copy()


# ---------------------------------------------------------
# Crossover
# ---------------------------------------------------------

def crossover(parent1, parent2):

    point = random.randint(1, NUM_PHASES - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# ---------------------------------------------------------
# Mutation
# ---------------------------------------------------------

def mutation(individual, mutation_rate=0.1):

    individual = individual.copy()

    for i in range(NUM_PHASES):

        if random.random() < mutation_rate:

            # Random change
            change = random.randint(-5, 5)

            individual[i] += change

            # Keep within limits
            individual[i] = max(
                MIN_GREEN[i],
                min(MAX_GREEN[i], individual[i])
            )

    return individual


# ---------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------

def genetic_algorithm(
    population_size=100,
    generations=200,
    mutation_rate=0.1,
    crossover_rate=0.8
):

    population = create_population(population_size)

    best_solution = None
    best_fitness = float("inf")

    history = []

    for generation in range(generations):

        # Calculate fitness
        fitnesses = [
            calculate_delay(individual)
            for individual in population
        ]

        # Find best individual
        generation_best_index = np.argmin(fitnesses)

        generation_best = population[generation_best_index]
        generation_best_fitness = fitnesses[generation_best_index]

        if generation_best_fitness < best_fitness:

            best_fitness = generation_best_fitness
            best_solution = generation_best.copy()

        history.append(best_fitness)

        # New population
        new_population = []

        # Elitism: preserve best solution
        new_population.append(best_solution.copy())

        while len(new_population) < population_size:

            # Selection
            parent1 = tournament_selection(
                population,
                fitnesses
            )

            parent2 = tournament_selection(
                population,
                fitnesses
            )

            # Crossover
            if random.random() < crossover_rate:

                child1, child2 = crossover(
                    parent1,
                    parent2
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            child1 = mutation(
                child1,
                mutation_rate
            )

            child2 = mutation(
                child2,
                mutation_rate
            )

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_solution, best_fitness, history


# ---------------------------------------------------------
# Run the Genetic Algorithm
# ---------------------------------------------------------

best_solution, best_fitness, history = genetic_algorithm()

# Calculate cycle time
cycle_time = (
    sum(best_solution)
    + NUM_PHASES * LOST_TIME
)

print("\nOptimal Traffic Signal Timing")
print("--------------------------------")

for i, green in enumerate(best_solution):

    print(
        f"Phase {i + 1}: "
        f"{green} seconds green"
    )

print(f"\nCycle time: {cycle_time:.2f} seconds")
print(f"Estimated delay: {best_fitness:.2f}")
