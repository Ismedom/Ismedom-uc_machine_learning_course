# Power Consumption Analysis and Forecasting

This project provides a comprehensive solution for analyzing power consumption patterns, predicting future usage, and optimizing energy costs. It consists of a Next.js frontend, FastAPI backend, and machine learning components for time series forecasting.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
   - [Backend (FastAPI)](#1-backend-fastapi)
   - [Frontend (Next.js)](#2-frontend-nextjs)
4. [Training the AI Model](#3-training-the-ai-model)
5. [Running the Application](#running-the-application)
6. [API Endpoints](#api-endpoints)
7. [License](#license)

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- pip (Python package manager)

## Project Structure

```
.
├── frontend/               # Next.js frontend application
├── services/               # Backend business logic and services
├── routes/                 # FastAPI route handlers
├── ai_models/              # ML models and data processing
├── data/                   # Data files and datasets
├── tests/                  # Test files
├── main.py                # FastAPI application entry point
└── requirements.txt        # Python dependencies
```

## Setup and Installation

### 1. Backend (FastAPI)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd us_ai_model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Linux/Mac
   .\venv\Scripts\activate # On Windows: 
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Frontend (Next.js)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   Create a `.env` file in the frontend directory:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

## 3. Training the AI Model

To train the power consumption forecasting model:

1. Ensure you have download the model by run this command 
```bash
 pip install -r requirements.txt
 ```

2. Run the training script:
   ```bash
   python3 model.py
   ```

   Available training options:
   ```bash
   python -m model.train \
     --data-path data/consumption_data.csv \
     --model-output models/power_forecast_model.pkl \
     --epochs 100 \
     --batch-size 32 \
     --learning-rate 0.001
   ```

3. The trained model will be saved to the specified output path and automatically used by the API.

## Running the Application

1. Start the FastAPI backend:
   ```bash
   python3 main.py
   ```
if you use uvicorn run this command :
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

2. In a separate terminal, start the Next.js frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Access the application at `http://localhost:3000`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# Ismedom-uc_machine_learning_course
