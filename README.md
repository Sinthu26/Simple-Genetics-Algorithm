# Simple Genetic Algorithm
Test Functions: De Jong's Sphere Function, Rosenbrock's Valley, Himmelblau's Function

## Requirements
- Python 3.11 or newer

## Installation
Run in terminal:
    pip install matplotlib

## How to Run
    python genetic_algorithm.py

The program runs experiments on all three objective functions, averaged over 5 independent
runs each. Results are printed to the terminal and the following convergence plots are saved:
- convergence_plot.png       (De Jong's Sphere — 4 parameter variation experiments)
- convergence_rosenbrock.png (Rosenbrock's Valley — baseline configuration)
- convergence_himmelblau.png (Himmelblau's Function — baseline configuration)

## Experiments (De Jong's Sphere)
Four configurations are tested to study the effect of parameter variations:
1. Baseline          — pop=100, mut=0.01, cross=0.8
2. Large Population  — pop=200, mut=0.01, cross=0.8
3. High Mutation     — pop=100, mut=0.05, cross=0.8
4. Low Crossover     — pop=100, mut=0.01, cross=0.4

## Folder Structure
Simple Genetics Algorithm/
├── genetic_algorithm.py
├── README.md
└── .gitignore
