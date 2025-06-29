import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy


def random_scrambling_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """Create a random circuit with single- and two-qubit gates."""
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng()
    for _ in range(depth):
        # random single qubit rotations
        for q in range(n_qubits):
            theta, phi, lam = rng.random(3) * 2 * np.pi
            qc.u(theta, phi, lam, q)
        # random pairs for entangling gates
        indices = list(range(n_qubits))
        rng.shuffle(indices)
        for a, b in zip(indices[::2], indices[1::2]):
            qc.cx(a, b)
        qc.barrier()
    return qc


def measure_entropy(state: Statevector, n_qubits: int) -> float:
    half = n_qubits // 2
    reduced = partial_trace(state, list(range(half, n_qubits)))
    return entropy(reduced, base=2)


def simulate(qubits: int, depth: int):
    qc = QuantumCircuit(qubits)
    entropies = []
    state = Statevector.from_label('0' * qubits)
    for layer in range(depth):
        qc_layer = random_scrambling_circuit(qubits, 1)
        state = state.evolve(qc_layer)
        s = measure_entropy(state, qubits)
        entropies.append(s)
        print(f"Layer {layer+1}: entropy = {s:.4f}")
    return entropies


def main():
    parser = argparse.ArgumentParser(description="Random scrambling simulation")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=10, help="Circuit depth")
    args = parser.parse_args()
    simulate(args.qubits, args.depth)


if __name__ == "__main__":
    main()
