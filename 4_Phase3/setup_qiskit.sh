#!/bin/bash

set -e  # Exit on any error

echo "=== Qiskit IBM Runtime Setup Script ==="
echo ""

# Step 1: Create virtual environment
echo "1. Creating Python virtual environment..."
python3 -m venv .venv
echo " Virtual environment created"

# Step 2: Activate virtual environment
echo ""
echo "2. Activating virtual environment..."
source .venv/bin/activate
echo " Virtual environment activated"

# Step 3: Install requirements
echo ""
echo "3. Installing requirements from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo " Error: requirements.txt not found in current directory"
    exit 1
fi
pip install -r requirements.txt
echo " Requirements installed"

# Step 4: Get IBM token from user
echo ""
echo "4. Setting up IBM Quantum token..."
echo "Please enter your IBM Quantum token:"
echo "(You can get it from https://quantum-computing.ibm.com/)"
read -s IBM_TOKEN  # -s flag hides the input
echo ""

if [ -z "$IBM_TOKEN" ]; then
    echo " Error: No token provided"
    exit 1
fi

# Step 5: Save the token using Python
echo "5. Saving IBM Quantum token..."
python3 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_cloud', token='$IBM_TOKEN')
print(' Token saved successfully')
"

# Step 6: Verify the connection
echo ""
echo "6. Verifying connection to IBM Quantum..."
python3 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
try:
    service = QiskitRuntimeService()
    backends = [b.name for b in service.backends()]
    print(' Connection successful!')
    print(f'Available backends: {len(backends)}')
    print('Backend names:', backends[:5] if len(backends) > 5 else backends)
    if len(backends) > 5:
        print(f'... and {len(backends) - 5} more backends')
except Exception as e:
    print(f' Connection failed: {e}')
    exit(1)
"

echo ""
echo "=== Setup Complete! ==="
echo "Your Qiskit IBM Runtime environment is ready to use."
echo "Remember to activate the virtual environment before using:"
echo "  source .venv/bin/activate"
