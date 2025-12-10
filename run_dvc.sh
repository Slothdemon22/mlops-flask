#!/bin/bash
# Activate the virtual environment
source /root/mlops-flask/venv/bin/activate

# Navigate to project directory
cd /root/mlops-flask

# Run the DVC pipeline
dvc repro
