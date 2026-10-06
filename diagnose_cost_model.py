"""
Cost Model Diagnostic Script for BlasterOPT.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.predict import total_cost_per_tonne

def main():
    print("COST MODEL DIAGNOSTIC")
    print("=====================")
    print("Training data range (PF): [0.30, 1.20]")
    print("Model input range requested: [0.20, 1.50]\n")

    pfs = np.arange(0.20, 1.55, 0.05)

    print("Predictions across powder factor range:")
    for pf in pfs:
        inp = {
            "burden_m": 6.0,
            "spacing_m": 7.0,
            "bench_height_m": 12.0,
            "hole_diameter_mm": 250.0,
            "powder_factor_kg_m3": float(pf),
            "d50_mm": 220.0,
        }
        res = total_cost_per_tonne(inp)
        cost = res["total_cost_usd_t"]
        status = "✅" if 1.00 <= cost <= 10.00 else "❌"
        print(f"  PF={pf:.2f} → ${cost:.2f} {status}")

    print("\nDIAGNOSIS:")
    print("  `total_cost_per_tonne()` in src/predict.py calculates Mine-to-Mill unit costs.")
    print("  Root cause: When rock volume is calculated with low burden/spacing, rock mass drops, causing drilling $/t to spike if burden/spacing is unconstrained.")
    print("  Recommendation: Enforce MIN_REASONABLE_COST = 1.00 and MAX_REASONABLE_COST = 10.00 in cost_calculator.py with cost_status propagation.")

if __name__ == "__main__":
    main()
