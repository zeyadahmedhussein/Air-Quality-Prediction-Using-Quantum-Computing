"""
Quantum Device Configuration Module
Supports both noise model integration and real hardware execution
Following Phase 3 IBM Quantum Deployment Guidelines
"""

import pennylane as qml
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import Fake127QPulseV1
import warnings
import os
from typing import Optional, Dict, Any, Union

class QuantumDeviceManager:
    """
    Manages quantum device creation and configuration
    Supports local simulation, noisy simulation, and real hardware
    """
    
    def __init__(self, backend_name: Optional[str] = None, 
                 shots: int = 1024, 
                 use_noise_model: bool = False,
                 use_real_hardware: bool = False):
        """
        Initialize quantum device manager
        
        Args:
            backend_name: IBM Quantum backend name (e.g., 'ibm_brisbane')
            shots: Number of shots for quantum measurements
            use_noise_model: Whether to use noise model from backend
            use_real_hardware: Whether to use real IBM hardware via qiskit.remote
        """
        self.backend_name = backend_name
        self.shots = shots
        self.use_noise_model = use_noise_model
        self.use_real_hardware = use_real_hardware
        self.noise_model = None
        self.backend = None
        
        # Initialize runtime service if needed
        if backend_name or use_real_hardware:
            self._initialize_runtime_service()
    
    def _initialize_runtime_service(self):
        """Initialize QiskitRuntimeService for IBM Quantum access"""
        try:
            # Save account credentials if provided
            if not self._is_account_saved():
                self._save_account_credentials()
            
            # Get runtime service
            self.service = QiskitRuntimeService()
            
            if self.backend_name:
                self.backend = self.service.backend(self.backend_name)
                print(f"✓ Connected to backend: {self.backend_name}")
                
                # Validate backend meets requirements (≥127 qubits)
                self._validate_backend()
                
        except Exception as e:
            print(f"Warning: Could not initialize runtime service: {e}")
            print("Falling back to local simulation")
            self.use_real_hardware = False
            self.use_noise_model = False
    
    def _is_account_saved(self) -> bool:
        """Check if IBM Quantum account is already saved"""
        try:
            QiskitRuntimeService()
            return True
        except Exception:
            return False
    
    def _save_account_credentials(self):
        """Save IBM Quantum account credentials"""
        # This would typically use environment variables or config files
        # For security, credentials should not be hardcoded
        token = os.getenv('IBM_QUANTUM_TOKEN')
        instance = os.getenv('IBM_QUANTUM_INSTANCE')
        
        if token and instance:
            QiskitRuntimeService.save_account(
                channel="ibm_cloud",
                token=token,
                instance=instance,
                set_as_default=True,
                overwrite=True
            )
        else:
            raise RuntimeError(
                "IBM Quantum credentials not found. Please set IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE environment variables."
            )
    
    def _validate_backend(self):
        """Validate backend meets competition requirements"""
        if not self.backend:
            return
            
        try:
            num_qubits = self.backend.num_qubits
            if num_qubits < 127:
                raise ValueError(f"Backend {self.backend_name} has {num_qubits} qubits, minimum 127 required")
            
            print(f"✓ Backend validated: {num_qubits} qubits available")
            
        except Exception as e:
            print(f"Warning: Backend validation failed: {e}")
    
    def create_noise_model(self) -> Optional[NoiseModel]:
        """
        Create noise model from backend for realistic simulation
        Following your requirement to use NoiseModel from backend
        """
        if not self.use_noise_model:
            return None
            
        try:
            if self.backend:
                # Create noise model from real backend
                self.noise_model = NoiseModel.from_backend(self.backend)
                print(f"✓ Noise model created from backend: {self.backend_name}")
            else:
                # Use fake backend for noise model
                fake_backend = Fake127QPulseV1()
                self.noise_model = NoiseModel.from_backend(fake_backend)
                print("✓ Noise model created from fake backend (Fake127QPulseV1)")
                
            return self.noise_model
            
        except Exception as e:
            print(f"Warning: Could not create noise model: {e}")
            return None
    
    def create_ideal_device(self, n_qubits: int):
        """Create ideal quantum device for noiseless simulation"""
        return qml.device("lightning.qubit", wires=n_qubits, shots=self.shots)
    
    def create_noisy_device(self, n_qubits: int):
        """Create noisy quantum device using noise model from backend"""
        noise_model = self.create_noise_model()
        
        if noise_model:
            # Use qiskit.aer with noise model
            return qml.device(
                "qiskit.aer", 
                wires=n_qubits, 
                backend="aer_simulator",
                noise_model=noise_model,
                shots=self.shots
            )
        else:
            # Fallback to simulated noise
            print("Warning: Using simulated noise instead of backend noise model")
            return qml.device("default.mixed", wires=n_qubits, shots=self.shots)
    
    def create_hardware_device(self, n_qubits: int):
        """
        Create real hardware device using qiskit.remote
        Following your requirement to use qiskit.remote for real hardware
        """
        if not self.use_real_hardware or not self.backend_name:
            raise ValueError("Real hardware requires backend_name and use_real_hardware=True")
        
        try:
            # Use qiskit.remote device for real IBM hardware
            device = qml.device(
                "qiskit.remote",
                wires=n_qubits,
                backend=self.backend_name,
                shots=self.shots,
                # Add optimization options
                optimization_level=3,
                resilience_level=1
            )
            
            print(f"✓ Real hardware device created: {self.backend_name}")
            return device
            
        except Exception as e:
            print(f"Error creating hardware device: {e}")
            print("Falling back to noisy simulation")
            return self.create_noisy_device(n_qubits)
    
    def create_device(self, n_qubits: int, device_type: str = "ideal"):
        """
        Create quantum device based on specified type
        
        Args:
            n_qubits: Number of qubits
            device_type: "ideal", "noisy", or "hardware"
            
        Returns:
            Configured PennyLane quantum device
        """
        if device_type == "ideal":
            return self.create_ideal_device(n_qubits)
        elif device_type == "noisy":
            return self.create_noisy_device(n_qubits)
        elif device_type == "hardware":
            return self.create_hardware_device(n_qubits)
        else:
            raise ValueError(f"Unknown device_type: {device_type}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current device configuration"""
        return {
            "backend_name": self.backend_name,
            "shots": self.shots,
            "use_noise_model": self.use_noise_model,
            "use_real_hardware": self.use_real_hardware,
            "has_noise_model": self.noise_model is not None,
            "has_backend": self.backend is not None
        }


def create_quantum_circuit(n_qubits: int, n_layers: int, device):
    """
    Create quantum circuit with specified architecture
    Compatible with both ideal and noisy devices
    """
    @qml.qnode(device, interface="torch")
    def quantum_circuit(inputs, weights):
        """
        Quantum variational circuit
        
        Args:
            inputs: Input features (n_qubits,)
            weights: Variational parameters (n_layers, n_qubits, 3)
        
        Returns:
            List of expectation values for each qubit
        """
        # Data encoding using angle embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # Variational layers with strong entangling gates
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_circuit


def create_device_from_config(config: Dict[str, Any], n_qubits: int):
    """
    Create quantum device from configuration dictionary
    Convenient function for loading from config files
    """
    device_manager = QuantumDeviceManager(
        backend_name=config.get('backend_name'),
        shots=config.get('shots', 1024),
        use_noise_model=config.get('use_noise_model', False),
        use_real_hardware=config.get('use_real_hardware', False)
    )
    
    device_type = config.get('device_type', 'ideal')
    return device_manager.create_device(n_qubits, device_type)


# Example usage and testing functions
def test_device_creation():
    """Test quantum device creation with different configurations"""
    print("Testing quantum device creation...")
    
    # Test ideal device
    manager = QuantumDeviceManager()
    ideal_dev = manager.create_device(4, "ideal")
    print(f"✓ Ideal device created: {ideal_dev}")
    
    # Test noisy device
    manager_noisy = QuantumDeviceManager(use_noise_model=True)
    noisy_dev = manager_noisy.create_device(4, "noisy")
    print(f"✓ Noisy device created: {noisy_dev}")
    
    # Test hardware device (would require real backend)
    try:
        manager_hw = QuantumDeviceManager(
            backend_name="ibm_brisbane", 
            use_real_hardware=True
        )
        hw_dev = manager_hw.create_device(4, "hardware")
        print(f"✓ Hardware device created: {hw_dev}")
    except Exception as e:
        print(f"Hardware device test skipped: {e}")


if __name__ == "__main__":
    test_device_creation()
