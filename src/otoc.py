import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, Operator
from random_scrambling import random_scrambling_circuit


def otoc_value(n_qubits: int, depth: int) -> float:
    """Compute a simple OTOC for Z_0 and X_1."""
    U = random_scrambling_circuit(n_qubits, depth)
    state = Statevector.from_label('0' * n_qubits)
    W = Operator(Pauli('X'))
    V = Operator(Pauli('Z'))
    # evolve W
    U_op = Operator(U)
    W_t = U_op.adjoint() @ (W.expand(Operator(np.eye(2**(n_qubits-1))))) @ U_op
    V_full = V.expand(Operator(np.eye(2**(n_qubits-1))))
    O = W_t @ V_full @ W_t.adjoint() @ V_full
    return np.real(state.expectation_value(O))


def main():
    parser = argparse.ArgumentParser(description="Compute simple OTOC")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    val = otoc_value(args.qubits, args.depth)
    print(f"OTOC value: {val:.4f}")


if __name__ == "__main__":
    main()
