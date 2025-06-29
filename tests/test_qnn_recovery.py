from qiskit.quantum_info import Statevector
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from qnn_recovery import compute_fidelity


def test_fidelity_identity():
    psi = Statevector.from_label('00')
    assert compute_fidelity(psi, psi) == 1.0


def test_fidelity_orthogonal():
    psi = Statevector.from_label('00')
    phi = Statevector.from_label('11')
    assert abs(compute_fidelity(psi, phi)) < 1e-9
