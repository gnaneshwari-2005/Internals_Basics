# MLOps Lab CIE Project

## Overview
This project builds a machine learning pipeline to predict code review turnaround time.

## Tasks Completed
- Task 1: Data processing and model training
- Task 2: FastAPI deployment
- Task 3: Logging system
- Task 4: Documentation and reproducibility

## Project Structure
- data/ → datasets
- src/ → train.py, api.py
- models/ → trained model
- results/ → JSON outputs
- logs/ → logging file

## Setup Instructions
1. Create virtual environment
2. Install requirements:
   pip install -r requirements.txt

## Run Training
python src/train.py

## Run API
uvicorn src.api:app --reload --port 9000

## API Endpoint
POST /predict

Example Input:
{
  "pr_lines_changed": 1039,
  "reviewer_load": 5,
  "file_count": 22,
  "is_critical_path": 1
}

## Output
{
  "prediction": 18.92
}

## Logs
Logs are stored in:
logs/app.log