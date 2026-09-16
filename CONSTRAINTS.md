# BlastOpt Botswana Constraints & Domain Bounds

## Mining Domain Constraints
- **Site Context:** Open-pit diamond mining operations in Botswana (Jwaneng and Orapa mines).
- **Bench Height ($H$):** 10.0 to 18.0 meters (Standard bench height = 15.0 m).
- **Hole Diameter ($d$):** 150 to 311 mm (Standard production blast diameter = 250 mm).
- **Burden ($B$):** 3.0 to 8.0 meters.
- **Spacing ($S$):** 4.0 to 10.0 meters.
- **Stemming Length ($T$):** 2.5 to 6.0 meters.
- **Powder Factor ($PF$):** 0.30 to 1.20 kg/m³.
- **Sub-drilling ($J$):** 0.5 to 2.5 meters.

## Blast Safety Bounds
- **Peak Particle Velocity (PPV):** Maximum threshold 10.0 mm/s at nearest sensitive mine boundary / pit wall infrastructure.
- **Airblast Overpressure ($dBL$):** Maximum limit 120 dB.
- **Flyrock Distance:** Maximum allowable flyrock range 250 meters.

## Performance Bounds & Downstream Processing
- **Mean Fragmentation ($P_{50}$ / $d_{50}$):** Target range 10.0 cm to 35.0 cm for optimal primary crusher performance.
- **Crusher Throughput Target:** Range 1,800 to 3,200 tonnes per hour (tph).

### Downstream Comminution & Processing Impact
- **Primary Crusher Throughput:** Coarse boulders cause bridging and mechanical jamming in primary gyratory/jaw crushers, forcing downtime and reducing throughput.
- **Grinding Energy Consumption:** Suboptimal fragment size distributions increase kWh/tonne power requirements in downstream SAG and ball mills (governed by Bond Work Index scaling).
- **Liner & Equipment Wear:** Boulders and non-uniform feed accelerate liner wear and media consumption in crushers and grinding mills.

### Quadratic Penalty Model for Crusher Performance
Crusher throughput efficiency and operational cost penalization follow a quadratic relationship relative to target mean fragment size ($d_{50, \text{target}}$):
$$\text{Penalty} = k_{crush} \cdot (d_{50} - d_{50, \text{target}})^2$$
- Deviating above target ($d_{50} > d_{50, \text{target}}$) causes physical bridging, feed blockages, and severe mechanical wear.
- Deviating below target ($d_{50} < d_{50, \text{target}}$) causes packing, screen blinding, and energy inefficiency in primary crushing stages.

### Airblast Sensitivity Hierarchy
- **Stemming ($T$) — Most Sensitive:** Stemming length and quality directly confine explosive gases. Insufficient stemming allows high-pressure gas venting to atmosphere, creating intense airblast overpressure shockwaves ($dBL$).
- **Spacing ($S$) — Least Sensitive:** Spacing governs inter-hole stress wave interaction and fragment movement within the rock mass, having negligible direct impact on atmospheric shockwave generation.

### Economic Significance of Mine-to-Mill Optimization
- **Drill & Blast Cost Share:** Drilling and blasting operations account for up to **20% of total mining costs per tonne**.
- **Mine-to-Mill Balance:** Comminution (crushing and grinding) accounts for over 50% of total mine electrical energy. Small, targeted increases in D&B expenditure (powder factor tuning) yield disproportionately large energy savings, throughput gains, and reduced equipment wear downstream.

### Knowledge Transfer & Skill Shortage Mitigation
- **Industry Skill Shortage:** Botswana's mining sector faces periodic engineering skill shortages and frequent staff rotations between mine sites (Jwaneng, Orapa, Letlhakane, Karowe).
- **Nearest-Neighbors Historical Lookup:** The Similar Blast Recommender system (`src/recommender.py`) allows junior blasters and newly rotated engineers to query past historical blast logs with similar geometry/rock parameters, review achieved outcomes, and leverage historical 'lessons learned' to reduce operational risk.

### Real Data vs. Synthetic Data Domain Context
- **Synthetic Data Limitations:** Synthetic datasets generated from empirical equations (e.g. Kuz-Ram, USBM) rely on idealized assumptions and uniform rock mass properties.
- **Real Production Data Necessity:** Production blast logs from mine operations capture site-specific geological heterogeneity, structural discontinuities (joints, faults, bedding planes), Joint Wall Factor ($JWF$), bench groundwater saturation, explosive product degradation, and actual measured seismograph waveforms. Ingesting real mine data when data-sharing agreements are active is critical to calibrate ML models for production deployment and minimize generalization error on site.

### Measure-While-Drilling (MWD) & Real-Time Adaptive Blast Designer (Model 1)
- **MWD Sensor Ingestion:** MWD telemetry from smart drill rigs (Epiroc, Sandvik) measures penetration rate (ROP, m/hr), torque (N·m), weight-on-bit (WOB, kg), air pressure (bar), and Specific Energy of Drilling ($SED$).
- **Dynamic In-Flight Adaptation (`src/realtime_adaptive.py`):** Replaces static pre-blast assumptions with dynamic per-hole charging plan adjustments (`adjust_charging_plan`). Hard Kimberlite strata (low ROP) triggers powder factor increases (+15%), while high torque/vibration ground triggers stemming length increases (+0.5 m).
- **Risk Controller & Immutable Audit Log:** The automated Risk Controller (`risk_controller`) evaluates modified designs against regulatory limits ($PPV \le 10.0$ mm/s, $dBL \le 120$ dB). All recommendations and operator overrides are immutably logged to a local SQLite database (`audit_log.db`).

### 3D Digital Twin of the Bench & Mine-to-Mill Value
- **3D Spatial Digital Twin:** The Digital Twin module (`src/digital_twin.py`) constructs a 3D spatial representation of the bench incorporating geological block models (rock mass rating, Kimberlite vs Granite boundaries, joint set spacing), as-drilled hole trajectories, and structural discontinuities.
- **Downstream Mine-to-Mill Linking:** Simulated fragmentation distributions ($d_{10}, d_{50}, d_{80}$, Rosin-Rammler $n, x_c$) directly predict downstream excavator productivity (t/h), truck fill factors (%), primary crusher throughput (t/h), specific grinding energy (kWh/t), and total operating cost per tonne ($/t).

### Direct-to-Drill Rig Telematics & ISO 15143-3 Standard
- **ISO 15143-3 (AEMP 2.0) Standard:** Direct-to-drill connectivity (`src/drill_connectivity.py`) implements the ISO 15143-3 standard for telematics data exchange across Sandvik (My Sandvik) and Epiroc (Certiq) drill rigs.
- **Manual Data Entry Elimination:** Pushing 3D drill patterns directly from BlastOpt to drill rig cabin displays eliminates manual paper handoffs, USB transfers, and operator collar coordinate entry errors.

### Electronic Detonator Field-to-Cloud Integration
- **Precision Electronic Initiation:** Electronic detonator integration (`src/detonator_integration.py`) supports AEL IntelliShot, BME AXXIS, and Orica i-kon systems, providing $\pm 0.1$ ms timing accuracy for vibration wave cancellation and fragmentation optimization.
- **Regulatory Sequence Validation:** Pre-blast validation (`validate_sequence`) enforces Botswana Department of Mines environmental guidelines (minimum inter-hole delay $\ge 8$ ms, inter-row delay $\ge 25$ ms, $PPV \le 10.0$ mm/s, $dBL \le 120$ dB).

### Offline-First Architecture & Write-Ahead Log (WAL) Pattern
- **Write-Ahead Log (WAL) Resilience:** In remote open-pit mining benches with limited cellular connectivity, all local field modifications (hole measurements, design edits) are enqueued to persistent disk storage (`WriteAheadLog` in `src/offline_sync.py`) prior to transmission.
- **Exponential Backoff SyncManager:** When connectivity is restored, `SyncManager` replays pending WAL items using exponential backoff retries ($delay = initial \cdot backoff^{attempt}$) and resolves concurrent modification conflicts (`resolve_conflicts`).

### Botswana Regulatory Compliance Framework
- **Mines, Quarries, Works and Machinery Act (Cap. 44:02):** Mandates legal environmental compliance and public safety limits. Maximum allowable ground vibration $PPV \le 10.0$ mm/s (minimum detectable $0.1$ mm/s), airblast noise overpressure $dBL \le 120$ dB, and maximum flyrock range $\le 250$ m.
- **Data Protection Act of Botswana:** Mandates secure handling and anonymization of site-specific geological block models and blasting telematics.
- **Automated Compliance Engine (`src/regulatory.py`):** Configurable via `data/processed/regulatory_limits.json` to generate downloadable official PDF submission reports for Department of Mines audits.

### Mobile App Offline-First Architecture & RBAC
- **Offline-First Criticality:** Remote pit benches in Botswana open-pit operations (Jwaneng, Orapa, Karowe) experience limited or intermittent cellular coverage. The field app (`mobile/`) utilizes local JSON storage and an offline action queue to ensure drillers and blasters can log hole measurements without network connection.
- **Role-Based Access Control (RBAC):**
  - **Driller / Blaster:** Views approved blast designs, logs measured hole depth, charge weight, and stemming length.
  - **Mining Engineer:** Approves/edits pattern designs, reviews field logs, and configures optimization constraints.
  - **Supervisor / Manager:** Reviews shift summaries, oversees fleet status, and monitors real-time safety alerts.

### Setswana Language Support & Internationalization (i18n)
- **Local Accessibility:** Setswana is the national language of Botswana, spoken natively by over 90% of the population. Internationalization (`src/i18n.py` and `mobile/src/i18n.js`) allows blasters and pit operators to toggle between English and Setswana interfaces.
- **Mining Terminology Adaptations:**
  - *Drilling & Blasting:* "Thunyako le Go Tlhaba Garane"
  - *Powder Factor:* "Sekala sa Dithunyane"
  - *Ground Vibration (PPV):* "Tshikinyego ya Mmu"
  - *Flyrock:* "Mabopo a a Fofang"

### Debswana Enterprise Systems Integration (SAP, Deswik, Surpac)
- **Data Silo Elimination:** Direct API integration (`src/integrations.py`) connects BlastOpt with Debswana's core enterprise systems to eliminate manual spreadsheet exports:
  - **SAP ERP:** Ingests explosives $/kg and drilling rates $/m; pushes actual post-blast costs per tonne ($/t) back to cost centers.
  - **Deswik CAD:** Ingests 3D bench polygons and 3D hole collar locations; pushes optimized pattern geometries.
  - **GEOVIA Surpac:** Ingests 3D geological block models (rock types, Kimberlite pipe contacts, Bond Work Index hardness); pushes predicted $d_{50}$ fragmentation overlays.

### Physics-Informed Neural Network (PINN, Model 2) & Uncertainty Quantification
- **Architecture:** 12 input features, 4 hidden layers (128, 256, 256, 128) with Dropout(0.1), and 3 output heads (fragmentation $d_{50}$, PPV, airblast) (`src/pinn.py`).
- **Soft Loss Terms:** $L_{\text{total}} = L_{\text{data}} + \lambda_1 L_{\text{kuzram}} + \lambda_2 L_{\text{usbm}}$.
- **Monte Carlo Dropout Uncertainty:** Uses Monte Carlo dropout inference (`predict_with_uncertainty`) to estimate epistemic uncertainty, returning 95% confidence intervals and Out-Of-Distribution (OOD) risk alerts.

### Multi-Objective Pareto Optimizer (Model 3) & NSGA-II
- **Multi-Objective Trade-Off Surface:** Model 3 (`src/pareto_optimizer.py`) implements NSGA-II genetic algorithm optimization across 5 competing objectives (minimizing $d_{80}$ fragmentation, minimizing PPV vibration, minimizing airblast dBL, minimizing cost $/t$, and maximizing primary crusher throughput $t/h$).
- **Normalized Weighted Sum Selection:** Allows mine engineers to customize objective importance weights and select recommended designs from the Pareto front with plain-English trade-off explanations (`generate_trade_off_explanation`).

## Fallback Mechanisms
- When trained PyTorch ML artifacts (`.pkl` / `.pt`) are unavailable, systems must seamlessly fallback to physics-based formulations (Kuz-Ram, USBM, Langerfors-Kihlström flyrock equation).
