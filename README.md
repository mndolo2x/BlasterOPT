# BlastOpt Botswana

AI-Driven Drilling & Blasting Optimization Platform for Open-Pit Diamond Mining

**Repository:** https://github.com/mndolo2x/BlasterOPT

## Overview

BlastOpt Botswana is a comprehensive AI-powered platform designed for optimizing drilling and blasting operations in open-pit diamond mines, specifically tailored for Botswana's mining operations (Debswana Jwaneng/Orapa pits).

## Features

- **Machine Learning Models**: Random Forest, Gradient Boosting, and Linear Regression for blast prediction
- **Genetic Algorithm Optimization**: Differential evolution for parameter optimization
- **Digital Twin Technology**: Fragmentation modeling and downstream integration
- **Safety Compliance**: Real-time safety checks and regulatory compliance
- **Agent-Based Interface**: AI assistant with voice interaction (English/Setswana)
- **Multi-Objective Optimization**: Pareto front analysis for trade-off decisions
- **Uncertainty Quantification**: Ensemble methods and physics-informed neural networks
- **Equipment Integration**: Connectivity to Sandvik, Epiroc drilling systems
- **Bilingual Support**: English and Setswana language support

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd jules_session_2175162121400205044 (1)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Main Streamlit application
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
├── src/                       # Source modules
│   ├── config.py             # Configuration settings
│   ├── synthetic_data.py     # Synthetic data generator
│   ├── data_ingestion.py    # Data loading and preprocessing
│   ├── models.py             # ML models and registry
│   ├── predict.py            # Prediction functions
│   ├── optimize.py           # Optimization algorithms
│   ├── visualize.py          # Visualization functions
│   ├── report.py             # Report generation
│   ├── explainability.py     # Model explanations
│   ├── domain/               # Domain models
│   ├── services/             # Business services
│   └── agent/                # AI agent modules
├── models/                    # Model registry
├── data/                      # Data directory
│   ├── audit/                # Audit logs
│   └── processed/            # Processed data
└── tests/                     # Unit tests
```

## Usage

### Dashboard
Access the main dashboard to view blast data, statistics, and historical records.

### Data Ingestion
Upload real mine data or generate synthetic data for testing and training.

### ML Manager
Train and evaluate machine learning models with cross-validation.

### Predictor
Make single blast design predictions with safety reports and explainability.

### Optimizer
Use genetic algorithms to optimize blast parameters subject to safety constraints.

### Agent Interface
Interact with the AI assistant for guided blast design recommendations.

## Configuration

Set the `DEMO_MODE` environment variable:
- `DEMO_MODE=true`: Use simulated data (default)
- `DEMO_MODE=false`: Connect to live hardware streams

## Safety & Compliance

The platform includes comprehensive safety checks:
- Ground vibration (PPV) monitoring
- Flyrock distance prediction
- Fragmentation analysis
- Regulatory compliance verification
- Certified blaster approval workflow

## Development

### Running Tests
```bash
pytest tests/
```

### Startup Diagnostics
The application runs automatic diagnostics on startup to verify all subsystems are operational.

## License

This project is part of the BlastOpt Botswana initiative for diamond mining optimization.

## Support

For technical support, please contact the BlastOpt engineering team.
