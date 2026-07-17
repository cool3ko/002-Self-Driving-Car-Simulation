"""Self-Driving Car Simulation - Main entry point.

Evolves a population of AI-controlled cars with a genetic algorithm.
Each car reads sensor rays into a small neural network that outputs
forward/left/right/reverse controls. A car that crashes into the road
edges or an obstacle is marked damaged and stops updating; a car that
covers GOAL_DISTANCE is marked finished and earns a bonus for however
much time it had left, rewarding speed as well as distance. Once every
car in a generation has crashed or finished, the fittest brains are
bred and mutated into a new generation, and the obstacle course gets
a little denser than the last.
"""

import random
import sys

import pygame

from car import Car
from genetic_algorithm import create_population, evolve_generation
from obstacle import Obstacle
from road import Road

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 600
POPULATION_SIZE = 100
LANE_COUNT = 3
LAYER_SIZES = [7, 8, 4]  # 7 sensor rays -> 8 hidden neurons -> 4 controls
MAX_GENERATION_FRAMES = 3600  # ~60s at 60fps; ends generations where cars idle instead of crashing
GOAL_DISTANCE = 10_000  # distance a car must cover to "finish" the course
TIME_BONUS_PER_FRAME = 2  # extra fitness per frame left on the clock when a car finishes


def obstacle_gaps_for_generation(generation):
    """Shrink obstacle spacing as generations pass, so the course gets
    harder over time. Gaps bottom out so it never becomes unbeatable.

    Args:
        generation: Current generation number (1-based)

    Returns:
        tuple: (min_gap, max_gap) to pass to generate_obstacles
    """
    step = min(generation - 1, 30)
    min_gap = max(120, 250 - step * 4)
    max_gap = max(220, 450 - step * 6)
    return min_gap, max_gap


def generate_obstacles(road, count=40, start_y=-200, min_gap=250, max_gap=450):
    """Scatter obstacles down the road, one per gap, in a random lane.

    Args:
        road: Road the obstacles are placed on
        count: Number of obstacles to generate
        start_y: World y coordinate of the first obstacle
        min_gap: Minimum vertical spacing between obstacles
        max_gap: Maximum vertical spacing between obstacles

    Returns:
        list: Obstacle instances
    """
    obstacles = []
    y = start_y
    for _ in range(count):
        lane = random.randrange(road.lane_count)
        x = road.get_lane_center(lane)
        obstacles.append(Obstacle(x, y, width=30, height=30))
        y -= random.randint(min_gap, max_gap)
    return obstacles


def build_course(road, generation):
    """Generate an obstacle course for the given generation, dense
    enough to reach past GOAL_DISTANCE no matter how tight the gaps get.

    Args:
        road: Road the obstacles are placed on
        generation: Current generation number (1-based)

    Returns:
        list: Obstacle instances
    """
    min_gap, max_gap = obstacle_gaps_for_generation(generation)
    avg_gap = (min_gap + max_gap) / 2
    count = int((GOAL_DISTANCE + 1500) / avg_gap) + 5
    return generate_obstacles(road, count=count, min_gap=min_gap, max_gap=max_gap)


def spawn_generation(road, brains):
    """Create a fresh set of AI cars, one per brain, at the start line.

    Args:
        road: Road the cars start on
        brains: List of Brain objects to assign to the new cars

    Returns:
        list: Car instances
    """
    cars = []
    for brain in brains:
        car = Car(road.get_lane_center(LANE_COUNT // 2), 100, controller="AI")
        car.brain = brain
        cars.append(car)
    return cars


def main():
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Driving Car Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    road = Road(WINDOW_WIDTH / 2, WINDOW_WIDTH * 0.6, lane_count=LANE_COUNT)

    generation = 1
    obstacles = build_course(road, generation)

    brains = create_population(POPULATION_SIZE, LAYER_SIZES)
    cars = spawn_generation(road, brains)
    generation_frame = 0

    running = True
    force_next_gen = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                force_next_gen = True

        for car in cars:
            car.update(road.borders, obstacles)
        generation_frame += 1

        for car in cars:
            if not car.damaged and not car.finished and car.fitness >= GOAL_DISTANCE:
                car.finished = True
                car.fitness += (MAX_GENERATION_FRAMES - generation_frame) * TIME_BONUS_PER_FRAME

        active = [car for car in cars if not car.damaged and not car.finished]
        best_car = max(cars, key=lambda car: car.fitness)
        timed_out = generation_frame >= MAX_GENERATION_FRAMES

        if not active or timed_out or force_next_gen:
            finishers = sum(1 for car in cars if car.finished)
            reason = "forced" if force_next_gen and active and not timed_out else "natural"
            print(
                f"Generation {generation} best fitness: {best_car.fitness:.0f}"
                f"  ({finishers}/{len(cars)} reached the goal, {reason} end)"
            )
            force_next_gen = False
            brains = evolve_generation(cars, elite_count=2, mutation_amount=0.15)
            cars = spawn_generation(road, brains)
            generation += 1
            generation_frame = 0
            obstacles = build_course(road, generation)
            continue

        offset_y = WINDOW_HEIGHT * 0.7 - best_car.y

        window.fill((0, 0, 0))
        road.draw(window, offset_y)
        for obstacle in obstacles:
            obstacle.draw(window, offset_y)

        goal_y = 100 - GOAL_DISTANCE + offset_y
        pygame.draw.line(
            window, (255, 215, 0), (road.left, goal_y), (road.right, goal_y), 4
        )

        for car in cars:
            if car.damaged:
                continue
            color = (0, 220, 120) if car is best_car else (0, 130, 200)
            car.draw(window, offset_y, color=color)
        if best_car.sensor:
            best_car.sensor.draw(window, offset_y)

        hud = font.render(
            f"Generation {generation}   Active {len(active)}/{len(cars)}   "
            f"Best fitness {best_car.fitness:.0f} / goal {GOAL_DISTANCE}",
            True, (255, 255, 255)
        )
        window.blit(hud, (10, 10))

        hint = font.render("Press N to skip to the next generation", True, (180, 180, 180))
        window.blit(hint, (10, WINDOW_HEIGHT - 26))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
