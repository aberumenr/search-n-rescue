import pygame
import sys
import random
import csv
import matplotlib.pyplot as plt

pygame.init()

# config

GRID_COLS = 24
GRID_ROWS = 16
CELL_SIZE = 48

GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE

PANEL_WIDTH = 360

WIDTH = GRID_WIDTH + PANEL_WIDTH
HEIGHT = GRID_HEIGHT

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

WORLD_RECT = pygame.Rect(0, 0, GRID_WIDTH, GRID_HEIGHT)
PANEL_RECT = pygame.Rect(GRID_WIDTH, 0, PANEL_WIDTH, HEIGHT)

GRID_SIZE = 20

POP_SIZE = 16
EVALUATION_RUNS = 5

FINAL_VALIDATION_RUNS = 30

GENERATIONS = 30

sarge_image = pygame.image.load("sargeant.png").convert_alpha()
victim_image = pygame.image.load("victim.png").convert_alpha()

fullscreen = False

pygame.display.set_caption("SAR-GEANT")

clock = pygame.time.Clock()

# colors

BACKGROUND = (225, 232, 242)

SNOW_COLOR = (239, 245, 250)
GRID_COLOR = (211, 222, 233)

SNOW_DOT_1 = (230, 237, 244)
SNOW_DOT_2 = (226, 234, 242)
SNOW_DRIFT = (245, 248, 252, 80)

SUCCESS_COLOR = (217, 108, 72)

VICTIM_HEAD = (233, 205, 182)
VICTIM_COAT = (88, 109, 152)
VICTIM_SCARF = (235, 177, 92)
VICTIM_OUTLINE = (57, 70, 98)

PANEL_COLOR = (34, 43, 56)
PANEL_SECTION_COLOR = (44, 55, 70)

TEXT_DARK = (22, 28, 36)
TEXT_LIGHT = (245, 248, 252)
TEXT_MUTED = (190, 200, 212)

ACCENT_BLUE = (111, 168, 220)
ACCENT_CYAN = (130, 210, 235)
ACCENT_RED = (210, 78, 78)
ACCENT_GREEN = (83, 176, 118)
ACCENT_ORANGE = (218, 140, 74)

OBSTACLE_COLOR = (120, 130, 142)
OBSTACLE_SHADOW = (95, 105, 118)

DOG_COLOR = (152, 94, 62)
DOG_OUTLINE = (72, 44, 28)
VEST_COLOR = (189, 42, 42)

VICTIM_COLOR = (200, 62, 62)
FOOTPRINT_COLOR = (150, 160, 176)

DOG_CREAM = (232, 229, 204)
DOG_BROWN = (134, 92, 66)
DOG_DARK = (72, 54, 50)
DOG_NOSE = (45, 35, 35)
DOG_VEST = (182, 54, 54)
DOG_SHADOW = (180, 184, 192)

SARGE_SPRITE = [
    "........................",
    "..............oooo......",
    ".............obbbbo.....",
    "............obwbbbeo....",
    "...........obwwwbbno....",
    "...........obcwcccbo....",
    "......ooooobcccvvvbo....",
    "...ooobbbbbccccvvvvbo...",
    "..obbbbbbbcccccvvvvbo...",
    ".obbbccccccccccccccbo...",
    ".obccccccccccccccccbo...",
    "obccccccccccccccccccbo..",
    "obccccccccccccccccccbo..",
    ".obcccccoovvvvooccoo....",
    "..obccoo..ovvvo..occo...",
    "...obco...occo...occo...",
    "...oooo...oooo...oooo...",
    "........................",
    "........................"
]

SARGE_PALETTE = {
    "o": (62, 45, 39),      # dark outline
    "c": (238, 231, 205),   # cream fur
    "w": (250, 246, 225),   # white blaze
    "b": (139, 86, 55),     # brown markings
    "v": (181, 49, 52),     # rescue vest
    "e": (46, 35, 32),      # eye
    "n": (38, 30, 29)       # nose
}

# world

START_POSITION = [1, 1]

dog_pos = START_POSITION.copy()
victim_pos = [19, 13]

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
show_controls = False

dog_facing_right = True

step_count = 0
repeated_visits = 0
run_finished = False
show_grid = False

MAX_STEPS = 200
FAILURE_PENALTY = 500

last_auto_move = 0
AUTO_DELAY = 150

DOG_NAME = "Sargeant"

trail_positions = []

# basic map functions

def valid_position(x, y):

    if x < 0 or x >= GRID_COLS:
        return False

    if y < 0 or y >= GRID_ROWS:
        return False

    if (x, y) in obstacles:
        return False

    return True

def record_visit(position):

    global repeated_visits
    global trail_positions

    key = tuple(position)

    if key in visit_count:
        repeated_visits += 1

    visit_count[key] = visit_count.get(key, 0) + 1

    trail_positions.append(key)

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
    global dog_facing_right

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

    if best_x > dog_pos[0]:
        dog_facing_right = True

    elif best_x < dog_pos[0]:
        dog_facing_right = False

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
    global dog_facing_right

    new_x = dog_pos[0] + dx
    new_y = dog_pos[1] + dy

    if valid_position(new_x, new_y):

        if dx > 0:
            dog_facing_right = True

        elif dx < 0:
            dog_facing_right = False

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
    global trail_positions

    dog_pos = START_POSITION.copy()

    rescued = False
    auto_mode = False
    dog_facing_right = True

    visit_count = {}
    trail_positions = []

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

    if not show_grid:
        return

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                screen,
                GRID_COLOR,
                rect,
                1
            )

def draw_panel_divider():

    pygame.draw.line(
        screen,
        (55, 65, 80),
        (GRID_WIDTH, 0),
        (GRID_WIDTH, HEIGHT),
        2
    )

def draw_panel():

    panel_rect = pygame.Rect(
        GRID_WIDTH,
        0,
        PANEL_WIDTH,
        HEIGHT
    )

    pygame.draw.rect(
        screen,
        PANEL_COLOR,
        panel_rect
    )

def draw_panel_section(y, height, title):

    section_rect = pygame.Rect(
        GRID_WIDTH + 16,
        y,
        PANEL_WIDTH - 32,
        height
    )

    pygame.draw.rect(
        screen,
        PANEL_SECTION_COLOR,
        section_rect,
        border_radius=10
    )

    title_font = pygame.font.SysFont(None, 28)

    title_text = title_font.render(
        title,
        True,
        TEXT_LIGHT
    )

    screen.blit(
        title_text,
        (GRID_WIDTH + 28, y + 12)
    )

    return section_rect

def draw_scent_field():

    if not show_scent:
        return

    overlay = pygame.Surface(
        (GRID_WIDTH, GRID_HEIGHT),
        pygame.SRCALPHA
    )

    for row in range(GRID_ROWS):

        for col in range(GRID_COLS):

            strength = scent_strength(col, row)

            if strength <= 0:
                continue

            alpha = int(strength * 85)

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                overlay,
                (120, 215, 255, alpha),
                rect
            )

    screen.blit(overlay, (0, 0))

def draw_pawprints():

    if len(trail_positions) < 2:
        return

    max_steps = 24

    # figure out where the visible trail begins
    start_index = max(
        0,
        len(trail_positions) - max_steps
    )

    # don't draw underneath Sargeant
    recent_positions = trail_positions[
        start_index:-1
    ]

    if not recent_positions:
        return

    for actual_step, (x, y) in enumerate(
        recent_positions,
        start=start_index
    ):

        # age within the currently visible trail
        age_index = (
            actual_step - start_index
        )

        fade_ratio = (
            (age_index + 1)
            / len(recent_positions)
        )

        # old prints fade quickly
        fade_ratio = fade_ratio ** 2

        alpha = int(
            8 + 120 * fade_ratio
        )

        color = (
            FOOTPRINT_COLOR[0],
            FOOTPRINT_COLOR[1],
            FOOTPRINT_COLOR[2],
            alpha
        )

        paw_surface = pygame.Surface(
            (CELL_SIZE, CELL_SIZE),
            pygame.SRCALPHA
        )

        # use ACTUAL step number
        # so old prints never flip later
        if actual_step % 2 == 0:

            paws = [
                (15, 16),
                (31, 32)
            ]

        else:

            paws = [
                (31, 16),
                (15, 32)
            ]

        for px, py in paws:

            pygame.draw.ellipse(
                paw_surface,
                color,
                (
                    px - 4,
                    py - 3,
                    8,
                    6
                )
            )

            pygame.draw.circle(
                paw_surface,
                color,
                (
                    px - 3,
                    py - 4
                ),
                1
            )

            pygame.draw.circle(
                paw_surface,
                color,
                (
                    px,
                    py - 5
                ),
                1
            )

            pygame.draw.circle(
                paw_surface,
                color,
                (
                    px + 3,
                    py - 4
                ),
                1
            )

        screen.blit(
            paw_surface,
            (
                x * CELL_SIZE,
                y * CELL_SIZE
            )
        )

def draw_obstacles():

    for x, y in obstacles:

        rect = pygame.Rect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )

        inner_rect = pygame.Rect(
            x * CELL_SIZE + 3,
            y * CELL_SIZE + 4,
            CELL_SIZE - 6,
            CELL_SIZE - 8
        )

        pygame.draw.rect(
            screen,
            OBSTACLE_SHADOW,
            rect,
            border_radius=6
        )

        pygame.draw.rect(
            screen,
            OBSTACLE_COLOR,
            inner_rect,
            border_radius=6
        )

def draw_victim():

    x, y = victim_pos

    cell_x = x * CELL_SIZE
    cell_y = y * CELL_SIZE

    draw_width = 44
    draw_height = 44

    scaled_victim = pygame.transform.scale(
        victim_image,
        (draw_width, draw_height)
    )

    offset_x = (CELL_SIZE - draw_width) // 2
    offset_y = (CELL_SIZE - draw_height) // 2

    screen.blit(
        scaled_victim,
        (
            cell_x + offset_x,
            cell_y + offset_y
        )
    )

def draw_pixel_sprite(base_x, base_y, sprite, palette, pixel_size=2):

    for row_index, row in enumerate(sprite):

        for col_index, char in enumerate(row):

            if char == ".":
                continue

            color = palette[char]

            rect = pygame.Rect(
                base_x + col_index * pixel_size,
                base_y + row_index * pixel_size,
                pixel_size,
                pixel_size
            )

            pygame.draw.rect(
                screen,
                color,
                rect
            )

def draw_dog():

    x, y = dog_pos

    cell_x = x * CELL_SIZE
    cell_y = y * CELL_SIZE

    draw_width = 58
    draw_height = 58

    scaled_sarge = pygame.transform.scale(
        sarge_image,
        (draw_width, draw_height)
    )

    if not dog_facing_right:
        scaled_sarge = pygame.transform.flip(
            scaled_sarge,
            True,
            False
        )

    offset_x = (CELL_SIZE - draw_width) // 2
    offset_y = (CELL_SIZE - draw_height) // 2

    screen.blit(
        scaled_sarge,
        (
            cell_x + offset_x,
            cell_y + offset_y
        )
    )

def draw_text():

    title_font = pygame.font.SysFont(None, 38)
    subtitle_font = pygame.font.SysFont(None, 24)
    body_font = pygame.font.SysFont(None, 24)
    hint_font = pygame.font.SysFont(None, 22)
    big_font = pygame.font.SysFont(None, 42)

    # header
    title = title_font.render(
        "Search & Rescue",
        True,
        TEXT_LIGHT
    )

    subtitle = subtitle_font.render(
        f"Dog: {DOG_NAME}",
        True,
        TEXT_MUTED
    )

    screen.blit(
        title,
        (GRID_WIDTH + 24, 18)
    )

    screen.blit(
        subtitle,
        (GRID_WIDTH + 24, 56)
    )

    # run status
    draw_panel_section(
        95,
        205,
        "Run Status"
    )

    mode_text = (
        "Auto"
        if auto_mode
        else "Manual"
    )

    if rescued:

        status_text = "Rescue complete"
        status_color = SUCCESS_COLOR

    elif run_finished:

        status_text = "Search failed"
        status_color = ACCENT_RED

    else:

        status_text = "Searching"
        status_color = ACCENT_CYAN

    lines = [
        ("Mode", mode_text),
        ("Status", status_text),
        ("Steps", str(step_count)),
        ("Repeated", str(repeated_visits)),
        ("Distance", str(manhattan_distance())),
        ("Cost", f"{calculate_cost():.1f}")
    ]

    y = 138

    for label, value in lines:

        label_text = body_font.render(
            f"{label}:",
            True,
            TEXT_MUTED
        )

        value_text = body_font.render(
            value,
            True,
            status_color
            if label == "Status"
            else TEXT_LIGHT
        )

        screen.blit(
            label_text,
            (GRID_WIDTH + 28, y)
        )

        screen.blit(
            value_text,
            (GRID_WIDTH + 165, y)
        )

        y += 27

    # genome
    draw_panel_section(
        325,
        190,
        "Genome"
    )

    genes = [
        ("Scent", chromosome["scent"]),
        ("Explore", chromosome["exploration"]),
        ("Obstacle", chromosome["obstacle"]),
        ("Memory", chromosome["memory"])
    ]

    y = 370

    for label, value in genes:

        label_text = body_font.render(
            f"{label}:",
            True,
            TEXT_MUTED
        )

        value_text = body_font.render(
            f"{value:.4f}",
            True,
            TEXT_LIGHT
        )

        screen.blit(
            label_text,
            (GRID_WIDTH + 28, y)
        )

        screen.blit(
            value_text,
            (GRID_WIDTH + 165, y)
        )

        y += 31

    # bottom hint
    hint = hint_font.render(
        "ESC  -  controls",
        True,
        TEXT_MUTED
    )

    hint_rect = hint.get_rect(
        center=(
            GRID_WIDTH + PANEL_WIDTH // 2,
            HEIGHT - 28
        )
    )

    screen.blit(
        hint,
        hint_rect
    )

    # result overlay
    if rescued:

        message = big_font.render(
            "Rescue complete!",
            True,
            SUCCESS_COLOR
        )

        message_rect = message.get_rect(
            center=(
                GRID_WIDTH // 2,
                34
            )
        )

        screen.blit(
            message,
            message_rect
        )

    elif run_finished:

        message = big_font.render(
            "Search failed",
            True,
            ACCENT_RED
        )

        message_rect = message.get_rect(
            center=(
                GRID_WIDTH // 2,
                34
            )
        )

        screen.blit(
            message,
            message_rect
        )

def draw_controls_overlay():

    if not show_controls:
        return

    overlay = pygame.Surface(
        (GRID_WIDTH, GRID_HEIGHT),
        pygame.SRCALPHA
    )

    overlay.fill(
        (18, 25, 35, 185)
    )

    screen.blit(
        overlay,
        (0, 0)
    )

    box_width = 390
    box_height = 500

    box_x = (
        GRID_WIDTH // 2
        - box_width // 2
    )

    box_y = (
        GRID_HEIGHT // 2
        - box_height // 2
    )

    box = pygame.Rect(
        box_x,
        box_y,
        box_width,
        box_height
    )

    pygame.draw.rect(
        screen,
        PANEL_COLOR,
        box,
        border_radius=16
    )

    pygame.draw.rect(
        screen,
        PANEL_SECTION_COLOR,
        box,
        2,
        border_radius=16
    )

    title_font = pygame.font.SysFont(
        None,
        38
    )

    body_font = pygame.font.SysFont(
        None,
        26
    )

    muted_font = pygame.font.SysFont(
        None,
        22
    )

    title = title_font.render(
        "Controls",
        True,
        TEXT_LIGHT
    )

    screen.blit(
        title,
        (
            box_x + 28,
            box_y + 25
        )
    )

    controls = [
        ("A", "Start / stop automatic search"),
        ("SPACE", "Take one search step"),
        ("H", "Toggle scent visualization"),
        ("T", "Toggle grid"),
        ("E", "Evaluate current chromosome"),
        ("P", "Test initial population"),
        ("G", "Run genetic algorithm"),
        ("R", "Reset simulation"),
        ("F11", "Toggle fullscreen")
    ]

    y = box_y + 85

    for key, description in controls:

        key_text = body_font.render(
            key,
            True,
            ACCENT_CYAN
        )

        description_text = body_font.render(
            description,
            True,
            TEXT_LIGHT
        )

        screen.blit(
            key_text,
            (
                box_x + 30,
                y
            )
        )

        screen.blit(
            description_text,
            (
                box_x + 115,
                y
            )
        )

        y += 38

    close_text = muted_font.render(
        "Press ESC to return",
        True,
        TEXT_MUTED
    )

    close_rect = close_text.get_rect(
        center=(
            box_x + box_width // 2,
            box_y + box_height - 28
        )
    )

    screen.blit(
        close_text,
        close_rect
    )

def toggle_fullscreen():

    global display
    global fullscreen
    global windowed_size

    fullscreen = not fullscreen

    if fullscreen:

        windowed_size = (
            display.get_width(),
            display.get_height()
        )

        display = pygame.display.set_mode(
            (0, 0),
            pygame.FULLSCREEN
        )

    else:

        display = pygame.display.set_mode(
            windowed_size,
            pygame.RESIZABLE
        )

# main loop

running = True

while running:

    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                show_controls = not show_controls
                if show_controls:
                    auto_mode = False
            
            elif event.key == pygame.K_F11:
                toggle_fullscreen()    

            elif show_controls:
                continue  # ignore other keys while controls overlay is shown
  
            elif event.key == pygame.K_r:
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

            elif event.key == pygame.K_t:
                show_grid = not show_grid

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

        if event.type == pygame.VIDEORESIZE:
            if not fullscreen:

                windowed_size = (
                    event.w,
                    event.h
                )

                display = pygame.display.set_mode(
                    windowed_size,
                    pygame.RESIZABLE
                )

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
    screen.fill(BACKGROUND, WORLD_RECT)
    screen.fill(PANEL_COLOR, PANEL_RECT)

    draw_grid()
    draw_scent_field()
    draw_pawprints()

    draw_obstacles()
    draw_victim()
    draw_dog()

    draw_panel()
    draw_text()

    draw_controls_overlay()

    pygame.display.flip()

    clock.tick(60)


pygame.quit()
sys.exit()