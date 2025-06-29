# Quantum Computing and Black Hole Scrambling

This project explores the quantum computational complexity of black hole information scrambling. It contains a short research paper and a Python simulation demonstrating entanglement growth in random circuits. The code relies on [Qiskit](https://qiskit.org/) for simulation.

## Contents

- `paper/black_hole_scrambling.tex` - LaTeX document presenting the theory, simulation approach, and results.
- `src/random_scrambling.py` - Python script that builds a random circuit and tracks entanglement entropy.

## Running the Simulation

1. Install the dependencies

```bash
pip install -r requirements.txt
```

2. Execute the script

```bash
python src/random_scrambling.py --qubits 6 --depth 20
```

The script prints the bipartite von Neumann entropy after each circuit layer.

## Compiling the Paper

To build the PDF version of the paper you can run:

```bash
pdflatex paper/black_hole_scrambling.tex
```

Any standard LaTeX distribution should work.
