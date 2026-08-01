# Self-Driving Car Simulation

A 2D simulation where a population of AI-controlled cars learns to drive down a
lane-marked road, dodging obstacles, purely through evolution — no backpropagation,
no training data, no hand-coded driving rules.

Each car's "brain" is a small neural network fed by sensor rays; a genetic algorithm
breeds and mutates the fittest brains from one generation into the next, and the
obstacle course gets denser as the population improves.

Code lives in [`self-driving-car/`](self-driving-car).

## How it works

- **Sensing**: each car casts 7 sensor rays out in a fan and measures how far each
  one travels before hitting a road edge or obstacle (`sensor.py`).
- **Brain**: those 7 readings feed a fully-connected network (`7 → 8 → 4`) with a
  step activation, outputting forward/left/right/reverse controls (`neural_network.py`).
- **Fitness**: distance travelled down the road, with a time bonus added if a car
  covers the full goal distance before the generation's frame limit — so evolution
  rewards speed as well as survival (`main.py`).
- **Evolution**: once every car in a generation has crashed, finished, or the frame
  limit is hit, the fittest brains are carried over (elitism) and the rest of the
  next generation is bred from them with mutation (`genetic_algorithm.py`).
- **Difficulty ramp**: obstacle spacing shrinks generation over generation (down to a
  floor so the course never becomes unbeatable), so later generations are tested
  against a harder course than earlier ones.

## Reproducibility

Obstacle courses are seeded (`--seed`), so a given seed regenerates the exact same
course sequence across runs — useful for comparing different brains or settings
against an identical course rather than a randomly luckier or harder one. Each
run's seed is logged alongside every row in the training log, so historical results
can always be traced back to the course sequence that produced them.

## Running it

```bash
cd self-driving-car
python main.py                 # random seed, printed at startup
python main.py --seed 12345    # reproducible course sequence
python main.py --clear-logs    # wipe logs/training_log.csv before starting
```

Requires Python 3 with `pygame` and `numpy` installed. Press **N** during a run to
force-skip to the next generation.

## Training log & analysis

Every generation appends one row per car to `self-driving-car/logs/training_log.csv`
(generation, fitness, distance, frames alive, speed stats, turning behaviour,
damaged/finished flags, final position, and the course seed used). This is
explored in `self-driving-car/logs/Data_analysis.ipynb`.

## Project structure

```
self-driving-car/
  main.py               - simulation loop, CLI args, course generation, logging
  car.py                 - car physics, collision, fitness tracking
  sensor.py               - ray casting for the car's inputs
  neural_network.py       - Brain / Level (feed-forward network, step activation)
  genetic_algorithm.py    - population creation, selection, breeding, mutation
  road.py                  - lane geometry
  obstacle.py              - obstacle collision boxes
  utils.py                 - shared helpers
  logs/                     - training_log.csv + Data_analysis.ipynb
```
