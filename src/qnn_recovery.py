import argparse
import numpy as np
import torch
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from random_scrambling import random_scrambling_circuit


def build_model(n_qubits: int, depth: int):
    scramble = random_scrambling_circuit(n_qubits, depth)
    ansatz = RealAmplitudes(n_qubits, reps=1)
    circuit = scramble.compose(ansatz)
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=[],
        weight_params=ansatz.parameters,
        observables=[Statevector.from_label('0'*n_qubits).to_operator()]
    )
    return TorchConnector(qnn)


def compute_fidelity(original: Statevector, recovered: Statevector) -> float:
    return state_fidelity(original, recovered)


def train(qubits: int, depth: int, epochs: int):
    model = build_model(qubits, depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    target = torch.tensor([[1.0]])
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(torch.empty(0))
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    params = list(model.parameters())[0].detach().numpy()
    scramble = random_scrambling_circuit(qubits, depth)
    ansatz = RealAmplitudes(qubits, reps=1)
    ansatz = ansatz.assign_parameters(params)
    circuit = scramble.compose(ansatz)
    final_state = Statevector.from_label('0'*qubits).evolve(circuit)
    fidelity = compute_fidelity(Statevector.from_label('0'*qubits), final_state)
    print(f"Fidelity: {fidelity:.4f}")


def main():
    parser = argparse.ArgumentParser(description="QNN information recovery")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.qubits, args.depth, args.epochs)


if __name__ == "__main__":
    main()
