import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt


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
    entropies = []
    state = Statevector.from_label('0' * qubits)
    for layer in range(depth):
        qc_layer = random_scrambling_circuit(qubits, 1)
        state = state.evolve(qc_layer)
        s = measure_entropy(state, qubits)
        entropies.append(s)
        print(f"Layer {layer+1}: entropy = {s:.4f}")
    return entropies


def plot_entropies(entropies, path="entropy.png"):
    plt.figure()
    plt.plot(range(1, len(entropies) + 1), entropies, marker="o")
    plt.xlabel("Circuit depth")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved entropy plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Random scrambling simulation")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=10, help="Circuit depth")
    parser.add_argument("--plot", action="store_true", help="Generate entropy plot")
    parser.add_argument("--backend", type=str, default=None, help="IBMQ backend name")
    args = parser.parse_args()
    entropies = simulate(args.qubits, args.depth)
    if args.plot:
        plot_entropies(entropies)
    if args.backend:
        try:
            from qiskit_ibm_provider import IBMProvider
        except ImportError:
            print("qiskit-ibm-provider not installed. Cannot run on hardware.")
            return
        provider = IBMProvider()
        backend = provider.get_backend(args.backend)
        qc = random_scrambling_circuit(args.qubits, args.depth)
        qc.measure_all()
        tqc = transpile(qc, backend)
        job = backend.run(tqc)
        result = job.result()
        counts = result.get_counts()
        plot_histogram(counts).savefig("hardware_counts.png")
        print("Execution complete. Counts saved to hardware_counts.png")


if __name__ == "__main__":
    main()
