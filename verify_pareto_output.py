import sys
import pandas as pd

def verify_pareto_front(csv_path: str = "pareto_front.csv") -> bool:
    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ Error: Pareto front CSV is empty.")
        sys.exit(1)

    ppv_col = "ppv_mms" if "ppv_mms" in df.columns else "vibration_ppv_mms"

    for i, row in df.iterrows():
        burden = float(row["burden_m"])
        spacing = float(row["spacing_m"])
        stemming = float(row["stemming_m"])
        ppv = float(row[ppv_col])

        # Check burden bounds
        assert 2.0 <= burden <= 12.0, f"Row {i}: burden {burden} out of range [2, 12]"
        # Check spacing >= burden
        assert spacing >= burden - 1e-4, f"Row {i}: spacing {spacing} < burden {burden}"
        # Check spacing <= 1.5 * burden
        assert spacing <= 1.5 * burden + 1e-4, f"Row {i}: spacing {spacing} > 1.5 * burden {burden}"
        # Check stemming ratio
        ratio = stemming / burden
        assert 0.5 - 1e-4 <= ratio <= 1.0 + 1e-4, f"Row {i}: stemming ratio {ratio:.2f} out of range [0.5, 1.0]"
        # Check PPV margin
        assert ppv <= 4.0 + 1e-4, f"Row {i}: PPV {ppv} exceeds margin (4.0 mm/s)"

    print(f"✅ All {len(df)} designs are physically valid")
    return True

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "pareto_front.csv"
    verify_pareto_front(csv_file)
