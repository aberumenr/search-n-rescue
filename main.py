import pygame
import sys
import random
import csv
import matplotlib.pyplot as plt

pygame.init()

# config
GRID_SIZE = 20
CELL_SIZE = 32

POP_SIZE = 16
EVALUATION_RUNS = 5

FINAL_VALIDATION_RUNS = 30

GENERATIONS = 30

GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE

WIDTH = GRID_WIDTH
HEIGHT = GRID_HEIGHT

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Search & Rescue")

clock = pygame.time.Clock()

# colors

BACKGROUND = (10, 15, 25)
CELL_COLOR = (30, 35, 45)
GRID_COLOR = (55, 65, 80)

OBSTACLE_COLOR = (90, 90, 100)
DOG_COLOR = (60, 150, 220)
VICTIM_COLOR = (190, 55, 65)

TEXT_COLOR = (245, 245, 245)

# world

START_POSITION = [1, 1]

dog_pos = START_POSITION.copy()
victim_pos = [17, 15]

obstacles = {
    (5, 5), (5, 6), (5, 7), (5, 8),
    (8, 10), (9, 10), (10, 10), (11, 10),
    (14, 4), (14, 5), (14, 6),
    (3, 14), (4, 14), (5, 14), (6, 14)
}

# manual chromosome

chromosome = {
    "scent": 0.90,
    "exploration": 0.45,
    "obstacle": 0.70,
    "memory": 0.65
}

# how far can the doggy smell
SCENT_RADIUS = 7

# how many times each position has been visited.
visit_count = {}

# state

current_population = []
evolution_history = []
best_evolved_chromosome = None

rescued = False
show_scent = False
auto_mode = False

step_count = 0
repeated_visits = 0
run_finished = False

MAX_STEPS = 200
FAILURE_PENALTY = 500

last_auto_move = 0
AUTO_DELAY = 150

# basic map functions

def valid_position(x, y):

    if x < 0 or x >= GRID_SIZE:
        return False

    if y < 0 or y >= GRID_SIZE:
        return False

    if (x, y) in obstacles:
        return False

    return True


def record_visit(position):

    global repeated_visits

    key = tuple(position)

    if key in visit_count:
        repeated_visits += 1

    visit_count[key] = visit_count.get(key, 0) + 1


record_visit(dog_pos)

def manhattan_distance():

    return (
        abs(dog_pos[0] - victim_pos[0])
        + abs(dog_pos[1] - victim_pos[1])
    )


def calculate_cost():

    distance = manhattan_distance()

    failure_penalty = 0

    if not rescued:
        failure_penalty = FAILURE_PENALTY

    cost = (
        step_count
        + 3 * repeated_visits
        + 2 * distance
        + failure_penalty
    )

    return cost

# scent

def scent_strength(x, y):
    """
    returns a local scent value between 0 and 1

    doggy does NOT receive the victim's coordinates
    it only receives this scalar scent measurement
    """

    distance = abs(x - victim_pos[0]) + abs(y - victim_pos[1])

    if distance > SCENT_RADIUS:
        return 0.0

    return 1.0 - (distance / SCENT_RADIUS)


# behavioral features

def exploration_score(x, y):

    if (x, y) not in visit_count:
        return 1.0

    return 0.0


def memory_penalty(x, y):

    return visit_count.get((x, y), 0)


def obstacle_penalty(x, y):
    """
    blocked cells immediately surrounding
    the candidate position
    """
    neighbors = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1)
    ]

    blocked = 0

    for nx, ny in neighbors:

        if not valid_position(nx, ny):
            blocked += 1

    return blocked / 4


# doggy decision-making

def evaluate_cell(x, y):

    scent = scent_strength(x, y)

    exploration = exploration_score(x, y)

    obstacle = obstacle_penalty(x, y)

    memory = memory_penalty(x, y)

    score = (
        chromosome["scent"] * scent
        + chromosome["exploration"] * exploration
        - chromosome["obstacle"] * obstacle
        - chromosome["memory"] * memory
    )

    # tiny noise prevents identical choices from always
    # resolving in exactly the same direction.
    score += random.uniform(-0.01, 0.01)

    return score


def autonomous_step():

    global dog_pos
    global step_count
    global run_finished

    x, y = dog_pos

    candidates = [
        (x, y - 1),  # UP
        (x, y + 1),  # DOWN
        (x - 1, y),  # LEFT
        (x + 1, y)   # RIGHT
    ]

    valid_candidates = []

    for nx, ny in candidates:

        if valid_position(nx, ny):

            score = evaluate_cell(nx, ny)

            valid_candidates.append(
                (score, nx, ny)
            )

    if not valid_candidates:
        return

    # choose cell with the highest behavioral score
    valid_candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, best_x, best_y = valid_candidates[0]

    dog_pos = [best_x, best_y]

    step_count += 1

    record_visit(dog_pos)

    if dog_pos == victim_pos:
        run_finished = True

    elif step_count >= MAX_STEPS:
        run_finished = True


# manual movement

def move_dog(dx, dy):

    global dog_pos
    global step_count
    global run_finished

    new_x = dog_pos[0] + dx
    new_y = dog_pos[1] + dy

    if valid_position(new_x, new_y):

        dog_pos = [new_x, new_y]

        step_count += 1

        record_visit(dog_pos)

        if dog_pos == victim_pos:
            run_finished = True

        elif step_count >= MAX_STEPS:
            run_finished = True


# reset

def reset_simulation():

    global dog_pos
    global rescued
    global auto_mode
    global visit_count
    global step_count
    global repeated_visits
    global run_finished

    dog_pos = START_POSITION.copy()

    rescued = False
    auto_mode = False

    visit_count = {}

    step_count = 0
    repeated_visits = 0
    run_finished = False

    record_visit(dog_pos)

# fast simulation

def simulate_chromosome(test_chromosome):

    sim_pos = START_POSITION.copy()

    sim_visits = {
        tuple(sim_pos): 1
    }

    sim_steps = 0
    sim_repeated = 0
    sim_rescued = False

    while sim_steps < MAX_STEPS and not sim_rescued:

        x, y = sim_pos

        candidates = [
            (x, y - 1),
            (x, y + 1),
            (x - 1, y),
            (x + 1, y)
        ]

        valid_candidates = []

        for nx, ny in candidates:

            if not valid_position(nx, ny):
                continue

            # scent
            scent = scent_strength(nx, ny)

            # exploration
            if (nx, ny) not in sim_visits:
                exploration = 1.0
            else:
                exploration = 0.0

            # obstacle avoidance
            obstacle = obstacle_penalty(nx, ny)

            # memory
            memory = sim_visits.get((nx, ny), 0)

            score = (
                test_chromosome["scent"] * scent
                + test_chromosome["exploration"] * exploration
                - test_chromosome["obstacle"] * obstacle
                - test_chromosome["memory"] * memory
            )

            score += random.uniform(-0.01, 0.01)

            valid_candidates.append(
                (score, nx, ny)
            )

        if not valid_candidates:
            break

        valid_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        _, best_x, best_y = valid_candidates[0]

        sim_pos = [best_x, best_y]

        sim_steps += 1

        key = tuple(sim_pos)

        if key in sim_visits:
            sim_repeated += 1

        sim_visits[key] = sim_visits.get(key, 0) + 1

        if sim_pos == victim_pos:
            sim_rescued = True

    distance = (
        abs(sim_pos[0] - victim_pos[0])
        + abs(sim_pos[1] - victim_pos[1])
    )

    failure_penalty = 0

    if not sim_rescued:
        failure_penalty = FAILURE_PENALTY

    cost = (
        sim_steps
        + 3 * sim_repeated
        + 2 * distance
        + failure_penalty
    )

    return {
        "rescued": sim_rescued,
        "steps": sim_steps,
        "repeated": sim_repeated,
        "distance": distance,
        "cost": cost
    }

def evaluate_chromosome(test_chromosome, runs=3):

    results = []

    for _ in range(runs):
        result = simulate_chromosome(test_chromosome)
        results.append(result)

    average_cost = sum(
        result["cost"] for result in results
    ) / runs

    return average_cost, results

# population

def random_chromosome():

    return {
        "scent": random.uniform(0.0, 1.0),
        "exploration": random.uniform(0.0, 1.0),
        "obstacle": random.uniform(0.0, 1.0),
        "memory": random.uniform(0.0, 1.0)
    }

def create_population(size=POP_SIZE):

    population = []

    for _ in range(size):
        population.append(
            random_chromosome()
        )

    return population

def evaluate_population(population):

    evaluated = []

    for chromosome_index, test_chromosome in enumerate(
        population,
        start=1
    ):

        average_cost, results = evaluate_chromosome(
            test_chromosome,
            runs=EVALUATION_RUNS
        )

        successful_runs = sum(
            1 for result in results
            if result["rescued"]
        )

        evaluated.append({
            "id": chromosome_index,
            "chromosome": test_chromosome,
            "average_cost": average_cost,
            "successful_runs": successful_runs,
            "results": results
        })

    evaluated.sort(
        key=lambda individual: individual["average_cost"]
    )

    return evaluated

def print_population_results(evaluated_population):

    print("\n" + "=" * 70)
    print("INITIAL POPULATION")
    print("=" * 70)

    for rank, individual in enumerate(
        evaluated_population,
        start=1
    ):

        genes = individual["chromosome"]

        print(
            f"{rank:02}. "
            f"Dog {individual['id']:02} | "
            f"Cost: {individual['average_cost']:.2f} | "
            f"Rescues: {individual['successful_runs']}/{EVALUATION_RUNS} | "
            f"Scent: {genes['scent']:.2f} | "
            f"Explore: {genes['exploration']:.2f} | "
            f"Obstacle: {genes['obstacle']:.2f} | "
            f"Memory: {genes['memory']:.2f}"
        )

    best = evaluated_population[0]

    print("\nBest dog:")
    print(best["chromosome"])
    print(
        f"Average cost: {best['average_cost']:.2f}"
    )

# selection

SELECTION_RATE = 0.5


def select_survivors(evaluated_population):

    keep_count = int(
        len(evaluated_population) * SELECTION_RATE
    )

    survivors = evaluated_population[:keep_count]

    return survivors

# parent selection

TOURNAMENT_SIZE = 3

def tournament_select(survivors):

    competitors = random.sample(
        survivors,
        TOURNAMENT_SIZE
    )

    winner = min(
        competitors,
        key=lambda individual: individual["average_cost"]
    )

    return winner

def select_parent_pair(survivors):

    parent_1 = tournament_select(survivors)
    parent_2 = tournament_select(survivors)

    while parent_2["id"] == parent_1["id"]:
        parent_2 = tournament_select(survivors)

    return parent_1, parent_2

def print_survivors(survivors):

    print("\n" + "=" * 70)
    print("SELECTED SURVIVORS")
    print("=" * 70)

    for rank, individual in enumerate(
        survivors,
        start=1
    ):

        genes = individual["chromosome"]

        print(
            f"{rank:02}. "
            f"Dog {individual['id']:02} | "
            f"Cost: {individual['average_cost']:.2f} | "
            f"Rescues: {individual['successful_runs']}/{EVALUATION_RUNS} | "
            f"Scent: {genes['scent']:.2f} | "
            f"Explore: {genes['exploration']:.2f} | "
            f"Obstacle: {genes['obstacle']:.2f} | "
            f"Memory: {genes['memory']:.2f}"
        )

def print_parent_pairs(survivors, pairs=4):

    print("\n" + "=" * 70)
    print("SELECTED PARENTS")
    print("=" * 70)

    for i in range(1, pairs + 1):

        parent_1, parent_2 = select_parent_pair(
            survivors
        )

        print(
            f"Pair {i}: "
            f"Dog {parent_1['id']:02} "
            f"(cost {parent_1['average_cost']:.2f}) "
            f"x "
            f"Dog {parent_2['id']:02} "
            f"(cost {parent_2['average_cost']:.2f})"
        )

# crossover

GENES = [
    "scent",
    "exploration",
    "obstacle",
    "memory"
]


def crossover(parent_1, parent_2):

    chromosome_1 = parent_1["chromosome"]
    chromosome_2 = parent_2["chromosome"]

    child_1 = {}
    child_2 = {}

    for gene in GENES:

        alpha = random.uniform(0.0, 1.0)

        child_1[gene] = (
            alpha * chromosome_1[gene]
            + (1 - alpha) * chromosome_2[gene]
        )

        child_2[gene] = (
            (1 - alpha) * chromosome_1[gene]
            + alpha * chromosome_2[gene]
        )

    return child_1, child_2

def create_offspring(survivors, offspring_count):

    offspring = []

    while len(offspring) < offspring_count:

        parent_1, parent_2 = select_parent_pair(
            survivors
        )

        child_1, child_2 = crossover(
            parent_1,
            parent_2
        )

        child_1 = mutate(child_1)
        child_2 = mutate(child_2)

        offspring.append(child_1)

        if len(offspring) < offspring_count:
            offspring.append(child_2)

    return offspring

def print_offspring(offspring):

    print("\n" + "=" * 70)
    print("OFFSPRING")
    print("=" * 70)

    for i, child in enumerate(
        offspring,
        start=1
    ):

        print(
            f"Puppy {i:02} | "
            f"Scent: {child['scent']:.2f} | "
            f"Explore: {child['exploration']:.2f} | "
            f"Obstacle: {child['obstacle']:.2f} | "
            f"Memory: {child['memory']:.2f}"
        )

# mutation

MUTATION_RATE = 0.20
MUTATION_STRENGTH = 0.15


def mutate(chromosome):

    mutated = chromosome.copy()

    for gene in GENES:

        if random.random() < MUTATION_RATE:

            change = random.uniform(
                -MUTATION_STRENGTH,
                MUTATION_STRENGTH
            )

            mutated[gene] += change

            mutated[gene] = max(
                0.0,
                min(1.0, mutated[gene])
            )

    return mutated

# evolution

def create_next_generation(evaluated_population):

    survivors = select_survivors(
        evaluated_population
    )

    survivor_chromosomes = []

    for individual in survivors:

        survivor_chromosomes.append(
            individual["chromosome"].copy()
        )

    offspring_count = (
        POP_SIZE - len(survivor_chromosomes)
    )

    offspring = create_offspring(
        survivors,
        offspring_count
    )

    next_population = (
        survivor_chromosomes
        + offspring
    )

    return next_population


def population_average_cost(evaluated_population):

    total = sum(
        individual["average_cost"]
        for individual in evaluated_population
    )

    return total / len(evaluated_population)


def population_rescue_rate(evaluated_population):

    total_runs = (
        len(evaluated_population)
        * EVALUATION_RUNS
    )

    successful_runs = sum(
        individual["successful_runs"]
        for individual in evaluated_population
    )

    return successful_runs / total_runs


def run_evolution(generations=GENERATIONS):

    population = create_population()

    history = []

    best_ever = None

    print("\n" + "=" * 70)
    print("EVOLUTION STARTED")
    print("=" * 70)

    for generation in range(generations + 1):

        evaluated = evaluate_population(
            population
        )

        best = evaluated[0]

        average_cost = population_average_cost(
            evaluated
        )

        rescue_rate = population_rescue_rate(
            evaluated
        )

        if (
            best_ever is None
            or best["average_cost"]
            < best_ever["average_cost"]
        ):

            best_ever = {
                "generation": generation,
                "average_cost": best["average_cost"],
                "chromosome": best["chromosome"].copy(),
                "successful_runs": best["successful_runs"]
            }

        history.append({
            "generation": generation,
            "best_cost": best["average_cost"],
            "best_so_far": best_ever["average_cost"],
            "average_cost": average_cost,
            "rescue_rate": rescue_rate
        })

        print(
            f"Generation {generation:02} | "
            f"Best: {best['average_cost']:.2f} | "
            f"Best so far: {best_ever['average_cost']:.2f} | "
            f"Average: {average_cost:.2f} | "
            f"Rescue rate: {rescue_rate * 100:.1f}%"
        )

        if generation == generations:
            break

        population = create_next_generation(
            evaluated
        )

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)

    print(
        f"Best found in generation "
        f"{best_ever['generation']}"
    )

    print(
        f"Best cost: "
        f"{best_ever['average_cost']:.2f}"
    )

    print("Best chromosome:")

    for gene in GENES:

        print(
            f"  {gene}: "
            f"{best_ever['chromosome'][gene]:.4f}"
        )

    return (
        best_ever["chromosome"].copy(),
        history,
        best_ever
    )

def save_evolution_history(history, filename="evolution_history.csv"):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "generation",
            "best_cost",
            "best_so_far",
            "average_cost",
            "rescue_rate"
        ])

        for row in history:

            writer.writerow([
                row["generation"],
                row["best_cost"],
                row["best_so_far"],
                row["average_cost"],
                row["rescue_rate"]
            ])

    print(
        f"\nEvolution history saved to {filename}"
    )

def validate_best_chromosome(
    best_chromosome,
    runs=FINAL_VALIDATION_RUNS
):

    average_cost, results = evaluate_chromosome(
        best_chromosome,
        runs=runs
    )

    successful_runs = sum(
        1 for result in results
        if result["rescued"]
    )

    average_steps = sum(
        result["steps"]
        for result in results
    ) / runs

    average_repeated = sum(
        result["repeated"]
        for result in results
    ) / runs

    rescue_rate = successful_runs / runs

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Runs: {runs}"
    )

    print(
        f"Successful rescues: "
        f"{successful_runs}/{runs}"
    )

    print(
        f"Rescue rate: "
        f"{rescue_rate * 100:.1f}%"
    )

    print(
        f"Average cost: "
        f"{average_cost:.2f}"
    )

    print(
        f"Average steps: "
        f"{average_steps:.2f}"
    )

    print(
        f"Average repeated visits: "
        f"{average_repeated:.2f}"
    )

    return {
        "runs": runs,
        "successful_runs": successful_runs,
        "rescue_rate": rescue_rate,
        "average_cost": average_cost,
        "average_steps": average_steps,
        "average_repeated": average_repeated
    }
def save_best_result(
    best_ever,
    validation,
    filename="best_result.txt"
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("BEST EVOLVED CHROMOSOME\n")
        file.write("=" * 40 + "\n")

        file.write(
            f"Generation found: "
            f"{best_ever['generation']}\n"
        )

        file.write(
            f"Training cost: "
            f"{best_ever['average_cost']:.2f}\n\n"
        )

        file.write("Chromosome:\n")

        for gene in GENES:

            file.write(
                f"{gene}: "
                f"{best_ever['chromosome'][gene]:.6f}\n"
            )

        file.write("\nFINAL VALIDATION\n")
        file.write("=" * 40 + "\n")

        file.write(
            f"Runs: {validation['runs']}\n"
        )

        file.write(
            f"Successful rescues: "
            f"{validation['successful_runs']}\n"
        )

        file.write(
            f"Rescue rate: "
            f"{validation['rescue_rate'] * 100:.1f}%\n"
        )

        file.write(
            f"Average cost: "
            f"{validation['average_cost']:.2f}\n"
        )

        file.write(
            f"Average steps: "
            f"{validation['average_steps']:.2f}\n"
        )

        file.write(
            f"Average repeated visits: "
            f"{validation['average_repeated']:.2f}\n"
        )

    print(
        f"Best result saved to {filename}"
    )

# graphs

def save_evolution_graphs(history):

    generations = [
        row["generation"]
        for row in history
    ]

    best_costs = [
        row["best_cost"]
        for row in history
    ]

    best_so_far = [
        row["best_so_far"]
        for row in history
    ]

    average_costs = [
        row["average_cost"]
        for row in history
    ]

    rescue_rates = [
        row["rescue_rate"] * 100
        for row in history
    ]

    # cost graph
    plt.figure(figsize=(10, 6))

    plt.plot(
        generations,
        best_costs,
        label="Best cost per generation"
    )

    plt.plot(
        generations,
        best_so_far,
        label="Best cost found so far"
    )

    plt.plot(
        generations,
        average_costs,
        label="Population average cost"
    )

    plt.xlabel("Generation")
    plt.ylabel("Cost")

    plt.title(
        "Evolution of Rescue Agent Cost"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "evolution_cost.png",
        dpi=200
    )

    plt.close()

    # rescue rate graph
    plt.figure(figsize=(10, 6))

    plt.plot(
        generations,
        rescue_rates,
        marker="o"
    )

    plt.xlabel("Generation")
    plt.ylabel("Successful rescues (%)")

    plt.title(
        "Rescue Success Rate by Generation"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "rescue_rate.png",
        dpi=200
    )

    plt.close()

    print(
        "Graphs saved to "
        "evolution_cost.png and rescue_rate.png"
    )

# drawing

def draw_grid():

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                screen,
                CELL_COLOR,
                rect
            )

            pygame.draw.rect(
                screen,
                GRID_COLOR,
                rect,
                1
            )


def draw_scent_field():

    if not show_scent:
        return

    overlay = pygame.Surface(
        (GRID_WIDTH, GRID_HEIGHT),
        pygame.SRCALPHA
    )

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            strength = scent_strength(col, row)

            if strength <= 0:
                continue

            alpha = int(strength * 120)

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                overlay,
                (70, 170, 255, alpha),
                rect
            )

    screen.blit(overlay, (0, 0))


def draw_obstacles():

    for x, y in obstacles:

        rect = pygame.Rect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )

        pygame.draw.rect(
            screen,
            OBSTACLE_COLOR,
            rect
        )


def draw_victim():

    x, y = victim_pos

    rect = pygame.Rect(
        x * CELL_SIZE,
        y * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
    )

    pygame.draw.rect(
        screen,
        VICTIM_COLOR,
        rect
    )


def draw_dog():

    x, y = dog_pos

    rect = pygame.Rect(
        x * CELL_SIZE,
        y * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
    )

    pygame.draw.rect(
        screen,
        DOG_COLOR,
        rect
    )


def draw_text():

    font = pygame.font.SysFont(None, 26)

    instructions = (
        "SPACE: step   A: auto   H: scent   "
        "E: evaluate   P: population   "
        "G: evolve   R: reset"
    )

    text = font.render(
        instructions,
        True,
        TEXT_COLOR
    )

    screen.blit(text, (12, HEIGHT - 28))

    stats_font = pygame.font.SysFont(None, 24)

    stats = (
        f"Steps: {step_count}   "
        f"Repeated: {repeated_visits}   "
        f"Distance: {manhattan_distance()}   "
        f"Cost: {calculate_cost():.1f}"
    )

    stats_text = stats_font.render(
        stats,
        True,
        TEXT_COLOR
    )

    screen.blit(
        stats_text,
        (12, 10)
    )

    if rescued:

        big_font = pygame.font.SysFont(
            None,
            48
        )

        message = big_font.render(
            "Rescue complete!",
            True,
            TEXT_COLOR
        )

        message_rect = message.get_rect(
            center=(WIDTH // 2, 45)
        )

        screen.blit(
            message,
            message_rect
        )

    elif run_finished:

        big_font = pygame.font.SysFont(
            None,
            48
        )

        message = big_font.render(
            "Search failed",
            True,
            TEXT_COLOR
        )

        message_rect = message.get_rect(
            center=(WIDTH // 2, 45)
        )

        screen.blit(
            message,
            message_rect
        )

# main loop

running = True

while running:

    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                reset_simulation()

            elif event.key == pygame.K_h:
                show_scent = not show_scent

            elif event.key == pygame.K_e:
                average_cost, results = evaluate_chromosome(
                    chromosome,
                    runs=3
                )

                print("\nChromosome evaluation")

                print(chromosome)

                for i, result in enumerate(results, start=1):

                    print(
                        f"Run {i}: "
                        f"rescued={result['rescued']}, "
                        f"steps={result['steps']}, "
                        f"repeated={result['repeated']}, "
                        f"distance={result['distance']}, "
                        f"cost={result['cost']}"
                    )

                print(
                    f"Average cost: {average_cost:.2f}"
                )

            elif event.key == pygame.K_p:

                current_population = create_population()

                evaluated_population = evaluate_population(
                    current_population
                )

                print_population_results(
                    evaluated_population
                )

                survivors = select_survivors(
                    evaluated_population
                )

                print_survivors(
                    survivors
                )

                print_parent_pairs(
                    survivors,
                    pairs=4
                )
                offspring = create_offspring(
                    survivors,
                    offspring_count=POP_SIZE - len(survivors)
                )

                print_offspring(
                    offspring
                )

            elif event.key == pygame.K_g:
                (
                    best_evolved_chromosome,
                    evolution_history,
                    best_ever
                ) = run_evolution(
                    generations=GENERATIONS
                )

                save_evolution_history(
                    evolution_history
                )
                
                save_evolution_graphs(
                    evolution_history
                )

                validation = validate_best_chromosome(
                    best_evolved_chromosome,
                    runs=FINAL_VALIDATION_RUNS
                )

                save_best_result(
                    best_ever,
                    validation
                )

                chromosome = (
                    best_evolved_chromosome.copy()
                )

                reset_simulation()

                print(
                    "\nBest evolved dog loaded "
                    "into visual simulation."
                )

                print(
                    "Press A to watch it search!"
                )

            elif event.key == pygame.K_a:
                if not run_finished:
                    auto_mode = not auto_mode

            elif not run_finished:

                if event.key == pygame.K_SPACE:
                    autonomous_step()

                elif event.key == pygame.K_UP:
                    move_dog(0, -1)

                elif event.key == pygame.K_DOWN:
                    move_dog(0, 1)

                elif event.key == pygame.K_LEFT:
                    move_dog(-1, 0)

                elif event.key == pygame.K_RIGHT:
                    move_dog(1, 0)

    # automatic movement
    if (
        auto_mode
        and not rescued
        and current_time - last_auto_move >= AUTO_DELAY
    ):

        autonomous_step()

        last_auto_move = current_time

    if dog_pos == victim_pos:
        rescued = True
        run_finished = True
        auto_mode = False

    elif step_count >= MAX_STEPS:
        run_finished = True
        auto_mode = False

    # drawing
    screen.fill(BACKGROUND)

    draw_grid()

    draw_scent_field()

    draw_obstacles()
    draw_victim()
    draw_dog()

    draw_text()

    pygame.display.flip()

    clock.tick(60)


pygame.quit()
sys.exit()