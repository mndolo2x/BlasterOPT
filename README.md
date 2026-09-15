# BlastOpt Botswana 🇧🇼💥

**BlastOpt Botswana** is an AI-powered blasting design, fragmentation modeling, ground vibration (PPV) prediction, and genetic algorithm optimization suite tailored for open-pit mining operations in Botswana (e.g., Jwaneng, Orapa, Karowe, Letlhakane).

---

## 🌟 Key Features

1. **Synthetic & Historical Data Ingestion**:
   - Generates high-fidelity physics-guided synthetic blasting data based on Kuz-Ram fragmentation, USBM ground vibration attenuation, empirical flyrock, and mining cost equations.
   - Cleans, validates, and engineers domain features (Powder Factor, Energy Factor, Spacing/Burden Ratio).

2. **Machine Learning Predictive Engine**:
   - Trains Multi-output ensemble ML models (Random Forest, XGBoost, Ridge/Linear Regression) to accurately predict mean fragment size ($d_{50}$), ground vibration (PPV), flyrock distance, and drilling & blasting cost per ton ($/t).
   - Features cross-validation evaluation and model artifact persistence.

3. **Genetic Algorithm Blast Parameter Optimization**:
   - Differential Evolution & Genetic Optimization to find optimal blast geometry parameters (Burden, Spacing, Stemming, Powder Factor).
   - Minimizes costs while adhering strictly to vibration and flyrock safety thresholds.

4. **Interactive Streamlit Web Dashboard**:
   - **Dashboard & Explorer**: Interactive data tables, summary KPIs, and distributions.
   - **Data Ingestion**: Dataset generation and custom CSV upload pipeline.
   - **Model Manager**: Train, evaluate, compare, and save ML models.
   - **Predictor**: Real-time predictions with interactive Kuz-Ram size distribution curve viewer.
   - **Optimizer**: Multi-objective and single-objective parameter optimization with convergence plots.
   - **2D Blast Pattern Visualizer**: Interactive hole layout with initiation delay sequences and timing visualization.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

### Running Tests

```bash
python3 -m pytest tests/
```

---

## 📁 Repository Structure

```
blastopt-botswana/
├── app.py                    # Streamlit main dashboard application
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py     # Data loading, cleaning, validation & feature engineering
│   ├── synthetic_data.py     # Physics-guided synthetic blast data generator
│   ├── models.py             # Machine learning model training, evaluation & persistence
│   ├── predict.py            # Prediction wrappers with physics fallbacks
│   ├── optimize.py           # Genetic algorithm & differential evolution parameter optimizer
│   └── visualize.py          # Interactive Plotly & Matplotlib visualization utilities
├── data/
│   ├── raw/                  # Original uploaded blast datasets
│   └── processed/            # Cleaned, feature-engineered datasets
├── models/                   # Saved ML model artifacts (.joblib)
├── notebooks/                # Exploratory notebooks
├── tests/                    # Unit testing suite
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Domain Physics & Models
- **Fragmentation**: Kuz-Ram Model ($d_{50} = A \cdot K^{-0.8} \cdot Q^{1/6} \cdot (115/E)^{19/30}$)
- **Vibration (PPV)**: USBM Scale Distance Law ($PPV = K_{vib} \cdot (SD)^{-\beta}$)
- **Flyrock**: Empirical scaled charge equations ($L_{fly} = K_{fly} \cdot (Q^{2/3} / B)$)
- **Cost**: Combined drilling, explosive product, initiation systems, and scaling cost per ton.

---

## 📜 License
MIT License. Developed for advanced mining analytics and blast optimization in Botswana.
