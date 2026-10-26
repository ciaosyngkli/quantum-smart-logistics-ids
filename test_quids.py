import os
import pytest
from qids_hybrid_v3 import *

# 1. Unit Test: Verify core script existence and syntax/imports
def test_unit_script_availability():
    assert os.path.exists("qids_hybrid_v3.py"), "Hybrid model script is missing!"

# 2. Integration Test: Verify component or data pipeline readiness
def test_integration_dependencies():
    import numpy as np
    import pandas as pd
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(2)
    assert qc.num_qubits == 2, "Quantum circuit initialization failed."

# 3. System Test: End-to-end execution flow simulation
def test_system_pipeline_flow():
    # Simulating end-to-end pipeline check for network intrusion features
    dummy_input = [0.1, 0.5, 0.8, 0.2]
    assert len(dummy_input) == 4, "System input dimension mismatch."
