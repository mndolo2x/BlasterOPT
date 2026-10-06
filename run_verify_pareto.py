import pandas as pd

df = pd.read_csv("pareto_front.csv")

assert df["cost_per_tonne_usd"].max() < 20.0, f"Max cost {df['cost_per_tonne_usd'].max()}"
assert df["cost_per_tonne_usd"].min() > 1.0, f"Min cost {df['cost_per_tonne_usd'].min()}"

coarse = df[df["d80_mm"] > 300]["cost_per_tonne_usd"].mean()
fine = df[df["d80_mm"] < 250]["cost_per_tonne_usd"].mean()

print(f"✅ Cost range: ${df['cost_per_tonne_usd'].min():.2f} – ${df['cost_per_tonne_usd'].max():.2f}/t")
if not pd.isna(fine) and not pd.isna(coarse):
    assert coarse > fine, "Coarse fragmentation should cost more than fine"
    assert coarse / fine < 3.0, f"Cost ratio too extreme: {coarse / fine:.1f}×"
    print(f"   Fine avg: ${fine:.2f}, Coarse avg: ${coarse:.2f}, Ratio: {coarse/fine:.2f}×")
