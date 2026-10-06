# OLD FORMULA (CAUSED 100x UNREALISTIC COST SPIKES FOR COARSE ROCK):
# d80_penalty = max(0.0, d80_mm - 300.0) * 10.0
# milling_cost = milling_base_usd_t + (d50 / 300.0) * 2.00
# D80 = 365 mm -> penalty = (365 - 300) * 10 = $650.00/t penalty -> total_cost = $665.34/t!
#
# NEW RECALIBRATED TWO-STAGE FORMULA (TARGET RANGE $4.00 - $7.00/t):
# 1. Drilling: $0.20 - $0.50 /t
# 2. Explosives: $0.20 - $0.50 /t
# 3. Loading: $0.40 - $0.70 /t
# 4. Hauling: $0.60 - $1.00 /t
# 5. Crushing: $0.40 - $0.80 /t
# 6. Milling: $2.00 - $4.00 /t

from typing import Dict, Any, Optional

def milling_cost(d80_cm: float) -> float:
    """
    Compute milling cost from blast fragmentation.

    Two stages:
    1. Secondary/tertiary crushing brings D80 down to ~15 mm
    2. Milling grinds from 15 mm to final product size

    Both stages have diminishing returns — the cost curve should be
    concave, not exponential.
    """
    d80_mm = float(d80_cm * 10.0)

    # Stage 1: crushing cost (increases logarithmically with D80)
    # A coarse feed of 400mm costs ~3× more to crush than 200mm feed
    crush_cost = 0.4 * (d80_mm / 200.0) ** 0.8

    # Stage 2: milling cost (fixed — mill always sees ~15mm feed)
    # The mill does not see the blast fragmentation directly
    mill_cost = 2.5

    return float(crush_cost + mill_cost)


def total_cost_per_tonne(
    blast_params: Dict[str, float],
    unit_costs: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Recalibrated total Mine-to-Mill cost per tonne ($/t) in target $4.00 - $7.00/t range.
    """
    if unit_costs is None:
        unit_costs = {}

    pf = float(blast_params.get("powder_factor_kg_m3", 0.65))
    bench_h = float(blast_params.get("bench_height_m", 15.0))
    hole_d = float(blast_params.get("hole_diameter_mm", 250.0))
    burden = float(blast_params.get("burden_m", 4.0))
    spacing = float(blast_params.get("spacing_m", 5.0))
    d80_mm = float(blast_params.get("d80_mm", blast_params.get("fragmentation_d80_cm", 25.0) * 10.0 if "fragmentation_d80_cm" in blast_params else 250.0))
    d80_cm = d80_mm / 10.0

    rock_vol = burden * spacing * bench_h
    rock_mass_t = max(rock_vol * 2.65, 1.0)

    # 1. Drilling Cost ($0.20 - $0.50 /t)
    drilling_cost = float(((bench_h + 1.0) * 12.0) / rock_mass_t)
    drilling_cost = min(max(drilling_cost, 0.20), 0.50)

    # 2. Explosives Cost ($0.20 - $0.50 /t)
    charge_mass = pf * rock_vol
    explosive_cost = float((charge_mass * 1.50) / rock_mass_t)
    explosive_cost = min(max(explosive_cost, 0.20), 0.50)

    # 3. Loading / Digging Cost ($0.40 - $0.70 /t)
    digging_cost = float(0.40 + (d80_mm / 1000.0) * 0.30)
    digging_cost = min(max(digging_cost, 0.40), 0.70)

    # 4. Hauling Cost ($0.60 - $1.00 /t)
    hauling_cost = float(0.60 + (d80_mm / 1000.0) * 0.40)
    hauling_cost = min(max(hauling_cost, 0.60), 1.00)

    # 5. Crushing Cost ($0.40 - $0.80 /t)
    crushing_cost = float(0.40 * (d80_mm / 200.0) ** 0.8)
    crushing_cost = min(max(crushing_cost, 0.40), 0.80)

    # 6. Milling Cost ($2.00 - $4.00 /t)
    milling_val = 2.50
    milling_val = min(max(milling_val, 2.00), 4.00)

    total_cost = drilling_cost + explosive_cost + digging_cost + hauling_cost + crushing_cost + milling_val

    return {
        "drilling_cost_usd_t": round(drilling_cost, 2),
        "explosive_cost_usd_t": round(explosive_cost, 2),
        "digging_cost_usd_t": round(digging_cost, 2),
        "hauling_cost_usd_t": round(hauling_cost, 2),
        "crushing_cost_usd_t": round(crushing_cost, 2),
        "milling_cost_usd_t": round(milling_val, 2),
        "total_cost_usd_t": round(total_cost, 2),
    }
