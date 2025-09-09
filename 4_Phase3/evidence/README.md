# Evidence Directory

This directory contains evidence and documentation for NILE Competition submission runs.

## Structure

- `job_ids.csv`: Tracks all quantum job executions with timestamps, backend info, and job IDs
- `logs/`: Contains execution logs and debug information
- `screenshots/`: Screenshots from IBM Quantum Portal showing completed jobs
- `README.md`: This documentation file

## Job IDs CSV Format

The `job_ids.csv` file tracks quantum job executions with the following columns:

- `timestamp`: ISO timestamp of job submission
- `backend`: Name of the quantum backend used
- `job_id`: Unique job identifier from the quantum provider
- `mode`: Either "physical" or "simulator" 
- `shots`: Number of shots used for the job
- `notes`: Additional notes about the run

## Screenshots

Screenshots should be saved in the `screenshots/` directory and should include:

1. Job completion confirmation from IBM Quantum Portal
2. Circuit visualization 
3. Job details showing backend, shots, and execution time
4. Results summary

## Logs

The `logs/` directory contains:

- Execution logs from quantum prediction runs
- Error logs if any issues occurred
- Performance metrics and timing information
- Debug output from quantum circuits

This evidence is required to validate that quantum jobs were actually executed on physical quantum hardware as specified in the competition requirements.
