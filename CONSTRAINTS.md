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

## Fallback Mechanisms
- When trained PyTorch ML artifacts (`.pkl` / `.pt`) are unavailable, systems must seamlessly fallback to physics-based formulations (Kuz-Ram, USBM, Langerfors-Kihlström flyrock equation).
