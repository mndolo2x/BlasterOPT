# BlastOpt Botswana 🇧🇼💥

**BlastOpt Botswana** is an end-to-end, AI-powered blasting design, fragmentation modeling, ground vibration prediction, and genetic algorithm optimization software suite engineered for open-pit mining operations in Botswana (e.g., Jwaneng Mine, Orapa Mine, Karowe Mine, Letlhakane Mine).

---

## 📖 Project Description

In open-pit diamond and hard-rock mining, drilling and blasting are critical primary operations. Drilling and blasting activities account for up to **20% of total mining costs per tonne**, making Mine-to-Mill optimization economically crucial. Suboptimal blast designs lead to coarse rock fragmentation, excessive ground vibration (Peak Particle Velocity / PPV posing environmental and structural risks), hazardous flyrock projections, and inflated operational expenditure.

### 🏭 Downstream Mining & Processing Impact
- **Crusher Throughput:** Coarse fragmentation causes bridging, mechanical jamming, and unexpected downtime in primary gyratory/jaw crushers.
- **Energy Consumption:** Suboptimal fragment size distribution drastically increases electrical kWh/tonne energy consumption in downstream SAG and ball mills.
- **Liner & Equipment Wear:** Large boulders accelerate abrasive wear on crusher liners, conveyor belts, and grinding media.

### 📐 Quadratic Penalty Model for Crusher Performance
Crusher performance and operational cost penalties are modeled quadratically relative to target mean fragment size ($d_{50, \text{target}}$):
$$\text{Penalty}_{crusher} = k \cdot (d_{50} - d_{50, \text{target}})^2$$
Deviations above target cause severe physical jamming and liner wear, while extreme fine deviations increase packing and screen blinding.

### 💨 Airblast Sensitivity Hierarchy
- **Stemming Length ($T$) — Most Sensitive Parameter:** Stemming length directly confines explosive detonation gases within the blast hole. Inadequate stemming results in premature gas venting to atmosphere, triggering high-intensity airblast overpressure shockwaves ($dBL$).
- **Spacing ($S$) — Least Sensitive Parameter:** Hole spacing governs stress wave superposition in the rock mass rather than direct gas venting into atmosphere, making it the least sensitive parameter for airblast overpressure generation.

### 🧠 Knowledge Transfer & Skill Shortage Mitigation
Botswana's mining industry faces periodic engineering skill shortages and staff rotations across operations (e.g. Debswana Jwaneng and Orapa mines). The **Similar Blast Recommender** (`src/recommender.py`) uses $k$-nearest neighbors matching to allow junior blasters and newly rotated engineers to search historical blast logs, examine actual achieved outcomes ($d_{50}$, PPV, flyrock, cost), and learn from historical lessons learned.

### 📊 Real Mine Data Ingestion vs. Synthetic Physics Data
Production blast deployment requires real mine production logs (`load_real_blast_data()` in `src/data_ingestion.py`). Synthetic physics datasets provide idealized empirical baseline estimates, whereas real mine production logs capture site-specific geological heterogeneity, structural joint orientations ($JWF$), bench water conditions, and actual measured seismograph waveforms.

### 📡 Real-Time MWD Telemetry & Closed-Loop Control
Drill rig MWD sensors (`src/mwd_ingestion.py`) capture real-time ROP, torque, WOB, and Specific Energy of Drilling ($SED$). Ingested via MQTT, MWD data enables closed-loop adaptation of charging plans (variable bulk explosive density and deck charges) based on as-drilled geometry and rock hardness transitions.

### 💎 3D Digital Twin of the Bench
Constructs a 3D spatial digital twin (`src/digital_twin.py`) integrating geological block models, as-drilled drillhole trajectories, and joint set orientations. Simulates full fragmentation size distributions ($d_{10}, d_{50}, d_{80}$, Rosin-Rammler $n, x_c$) and feeds downstream value models predicting shovel productivity, truck payload, crusher throughput, and specific grinding energy.

### 🚜 Direct-to-Drill Telematics (ISO 15143-3 / AEMP 2.0)
Connects BlastOpt directly to Sandvik (My Sandvik) and Epiroc (Certiq) drill rigs via ISO 15143-3 (`src/drill_connectivity.py`). Pushes 3D drill patterns to rig cabin displays and pulls as-drilled telemetry, eliminating paper handoffs and manual operator entry errors.

### ⚡ Electronic Detonator Integration (AEL / BME / Orica)
Integrates field-to-cloud initiation workflows (`src/detonator_integration.py`) for AEL IntelliShot, BME AXXIS, and Orica i-kon III systems. Validates timing sequences against regulatory vibration ($PPV \le 10.0$ mm/s) and airblast ($dBL \le 120$ dB) limits, and logs downloadable post-blast firing confirmations.

### 📱 React Native Mobile Field App (`mobile/`)
Field app for pit operators, drillers, and blasters in remote bench locations with limited cellular connectivity. Features offline-first local storage, action sync queue replay upon reconnection, input data validation, and Role-Based Access Control (Blaster, Engineer, Supervisor).

**BlastOpt Botswana** bridges domain mining physics with advanced multi-target Machine Learning (Random Forest, XGBoost, Ridge) and Genetic Algorithm / Differential Evolution optimization. The application enables mining engineers to:
- Predict mean fragment size ($d_{50}$), uniformity index ($n$), ground vibration (PPV), flyrock distance, and unit operational cost ($/t) in real time.
- Generate and visualize interactive Kuz-Ram Rosin-Rammler fragmentation size distribution curves with customizable characteristic size ($x_c$) and uniformity index ($n$) controls.
- Automatically optimize blast geometry (Burden, Spacing, Stemming, Powder Factor) to minimize total drilling & blasting costs while satisfying strict vibration, flyrock, and fragmentation constraints.
- Generate and export professional downloadable PDF summary reports containing top recommended blast designs and embedded fragmentation curves.
- Simulate 2D blast hole patterns with initiation delay timing sequences.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+ installed on your system.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/blastopt-botswana.git
   cd blastopt-botswana
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation by running unit tests:**
   ```bash
   python3 -m pytest tests/
   ```

---

## 🏃 Running the Application

To launch the multi-module Streamlit interactive dashboard:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your web browser.

---

## 📱 Application Modules & Pages

The Streamlit web dashboard features **6 interactive modules** accessible via the sidebar navigation:

1. **📊 Dashboard & Data Explorer (`Module 1`)**
   - High-level KPI overview (Total Blast Events, Mean $d_{50}$, Average Powder Factor, Average PPV).
   - Interactive data table viewer with filtering by mine site, rock type, and explosive type.
   - Exploratory distribution histograms and correlation matrices for blasting variables.

2. **⚙️ Data Ingestion & Synthetic Generator (`Module 2`)**
   - Physics-guided synthetic blast dataset generator with customizable sample size ($N$).
   - Custom CSV file upload pipeline with schema validation, missing value imputation, outlier clipping, and automated feature engineering.
   - Cleaned dataset download export functionality.

3. **🧠 ML Model Manager (`Module 3`)**
   - Multi-output ML model training suite (Random Forest, XGBoost, Ridge Regression).
   - 5-Fold Cross-Validation evaluation metrics ($R^2$, RMSE, MAE) across all 4 target variables.
   - Model comparative analysis charts, feature importances ranking, and model artifact persistence (`models/best_fragmentation_model.pkl`).

4. **🎯 Predictor & Kuz-Ram Curve (`Module 4`)**
   - Single-blast performance prediction input interface with ML inference and physics-fallback logic.
   - Interactive Plotly **Kuz-Ram Fragmentation Size Distribution Curve** viewer.
   - Real-time sliders for Tuning Uniformity Index ($n$) and Characteristic Size ($x_c$) parameters.

5. **⚡ Genetic Algorithm Optimizer (`Module 5`)**
   - Differential Evolution optimization engine minimizing cost ($/t) subject to PPV, flyrock, and $d_{50}$ fragmentation constraints.
   - Top 5 Recommended Blast Designs summary table.
   - Objective function convergence history plot.
   - One-click **Download Optimization PDF Report** button exporting clean PDF summaries with embedded Rosin-Rammler plots.

6. **📐 2D Blast Pattern & Delays (`Module 6`)**
   - Interactive 2D spatial layout generator for blast holes.
   - Configurable row-by-row and hole-by-hole initiation delay timing sequence simulation.

---

## 📊 Synthetic Data Generator & ML Models

### Physics-Guided Synthetic Data Generator (`src/synthetic_data.py`)
Generates realistic Botswana mining datasets incorporating key empirical domain physics:
- **Kuz-Ram Fragmentation Model**: Mean fragment size $d_{50} = A \cdot K^{-0.8} \cdot Q^{1/6} \cdot (115 / RWS)^{19/30}$
- **Rosin-Rammler Uniformity Index**: $n = (2.2 - 14 \cdot B / d) \cdot (1 + (S/B - 1)/2) \cdot (L / H)$
- **USBM Scaled Distance Ground Vibration**: $PPV = K_{vib} \cdot \left(\frac{D}{\sqrt{Q}}\right)^{-\beta}$
- **Empirical Scaled Charge Flyrock Distance**: $L_{fly} = K_{fly} \cdot \frac{Q^{2/3}}{B} \cdot \left(\frac{h_{stem}}{B}\right)^{-0.5}$
- **Operational Mining Cost Model**: Combined drilling, explosive product, initiation systems, and rock tonnage scaling.

### Machine Learning Engine (`src/models.py` & `src/predict.py`)
- **Algorithms**: Multi-output Random Forest, XGBoost Regressor, and Ridge Regression.
- **Engineered Domain Features**:
  - Spacing-to-Burden Ratio ($S/B$)
  - Stiffness Ratio ($H/B$)
  - Scaled Distance ($SD = D / \sqrt{Q}$)
  - Energy Factor ($MJ/m^3$)
  - Interaction Terms: Powder Factor $\times$ Burden (`pf_burden_interaction`), Spacing $\times$ Stemming (`spacing_stemming_interaction`).
- **Hyperparameter Tuning**: `GridSearchCV` optimization for fragmentation prediction ($d_{50}$) persisted to `models/best_fragmentation_model.pkl`.

---

## 🔮 Future Work

Potential future enhancements for BlastOpt Botswana include:
- **3D Spatial Blast & Block Model Integration**: Import 3D block models (Grade, Rock Quality Designation RQD) for spatially variable blast design.
- **Drone Aerial Image Fragmentation Analysis**: Computer vision integration (OpenCV) to measure post-blast fragmentation size distribution directly from drone orthomosaics.
- **Seismograph IoT Telemetry Pipeline**: Real-time streaming ingestion of near-field seismograph PPV waveforms via MQTT/Kafka.
- **Multi-Objective Pareto Frontier Optimization**: NSGA-II genetic algorithm to explicitly map the Pareto trade-off boundary between cost minimization and fragmentation fine-tuning.

---

## 📜 License
MIT License. Developed for advanced mining analytics, blast design optimization, and geotechnical safety in Botswana.
