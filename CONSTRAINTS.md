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

## Performance Bounds
- **Mean Fragmentation ($P_{50}$):** Target range 10.0 cm to 35.0 cm for optimal crusher throughput.
- **Crusher Throughput Target:** Range 1,800 to 3,200 tonnes per hour (tph).

## Fallback Mechanisms
- When trained PyTorch ML artifacts (`.pkl` / `.pt`) are unavailable, systems must seamlessly fallback to physics-based formulations (Kuz-Ram, USBM, Langerfors-Kihlström flyrock equation).
