"""
Physics core for blast design calculations.

All formulas are published standards. Citations in each docstring.
"""
import numpy as np

def kuznetsov_x50(
    rock_factor_a: float,
    burden_m: float,
    spacing_m: float,
    hole_depth_m: float,
    charge_mass_kg: float,
    explosive_rws: float,
) -> float:
    """
    Median fragment size x50 (cm) using the Kuznetsov equation.

    Formula (Kuznetsov 1973, refined by Cunningham 1983):
        x50 = A × (V0 / Q)^0.8 × Q^(1/6) × (115 / E)^(19/30)

    Where:
        A  = rock factor (6-12 typical)
        V0 = rock volume per hole = burden × spacing × hole_depth (m³)
        Q  = charge mass per hole (kg)
        E  = relative weight strength of explosive (ANFO = 100)

    Reference: Cunningham, C.V.B. (1983). "The Kuz-Ram model for
    prediction of fragmentation from blasting." Proc. 1st Int. Symp.
    on Rock Fragmentation by Blasting, Luleå, 439-454.

    Returns:
        x50 in centimeters.
    """
    V0 = burden_m * spacing_m * hole_depth_m
    x50_cm = (
        rock_factor_a
        * (V0 / charge_mass_kg) ** 0.8
        * charge_mass_kg ** (1 / 6)
        * (115 / explosive_rws) ** (19 / 30)
    )
    return float(x50_cm)


def cunningham_uniformity(
    burden_m: float,
    spacing_m: float,
    hole_diameter_mm: float,
    bench_height_m: float,
    charge_length_m: float,
    drilling_accuracy_m: float = 0.3,
) -> float:
    """
    Cunningham uniformity index n (dimensionless).

    Formula (Cunningham 1983):
        n = (2.2 - 0.014 × (B / D)) × ((1 + S/B) / 2)^0.5
            × (1 - W/B) × (L/H)

    Where:
        B = burden (m)
        S = spacing (m)
        D = hole diameter (mm)
        W = standard deviation of drilling accuracy (m), default 0.3
        L = charge length (m)
        H = bench height (m)

    Reference: Cunningham, C.V.B. (1983), Eq. 4.

    Typical range for n: 0.8 to 2.0.
    """
    B = burden_m
    S = spacing_m
    D = hole_diameter_mm
    W = drilling_accuracy_m
    L = charge_length_m
    H = bench_height_m

    n = (
        (2.2 - 0.014 * (B * 1000 / D))  # B in mm, D in mm
        * ((1 + S / B) / 2) ** 0.5
        * (1 - W / B)
        * (L / H)
    )
    return float(np.clip(n, 0.5, 2.5))


def rosin_rammler_d80(x50_cm: float, n: float) -> float:
    """
    D80 (cm) from Rosin-Rammler distribution.

    Formula (Rosin & Rammler 1933):
        D80 = x50 × (ln(5) / 0.693)^(1/n)

    Derivation:
        P(x) = 100 × (1 - exp(-0.693 × (x/x50)^n))
        Set P(D80) = 80:
            80 = 100 × (1 - exp(-0.693 × (D80/x50)^n))
            0.2 = exp(-0.693 × (D80/x50)^n)
            ln(5) = 0.693 × (D80/x50)^n
            D80 = x50 × (ln(5) / 0.693)^(1/n)

    Reference: Rosin, P., & Rammler, E. (1933). "The laws governing
    the fineness of powdered coal." J. Inst. Fuel, 7, 29-36.

    Returns:
        D80 in centimeters.
    """
    factor = (np.log(5.0) / 0.693) ** (1.0 / n)
    return float(x50_cm * factor)


def usbm_ppv(
    max_charge_per_delay_kg: float,
    distance_m: float,
    site_constant_k: float = 500.0,
    attenuation_b: float = 1.6,
) -> float:
    """
    Peak Particle Velocity PPV (mm/s) using the USBM predictor.

    Formula (Duvall & Fogelson 1962):
        PPV = K × (D / sqrt(W))^(-B)

    Where:
        K = site constant (500-1000 typical for hard rock)
        B = attenuation exponent (1.5-1.8 typical)
        D = distance from blast to receptor (m)
        W = maximum charge per delay (kg)

    Reference: Duvall, W.I. & Fogelson, D.E. (1962). "Review of
    criteria for estimating damage to residences from blasting
    vibrations." US Bureau of Mines RI 5968.

    Returns:
        PPV in mm/s.
    """
    W = max(max_charge_per_delay_kg, 0.1)  # prevent division by zero
    D = max(distance_m, 1.0)
    scaled_distance = D / np.sqrt(W)
    ppv = site_constant_k * scaled_distance ** (-attenuation_b)
    return float(ppv)


def siskind_airblast(
    max_charge_per_delay_kg: float,
    distance_m: float,
) -> float:
    """
    Airblast overpressure (dB) using the Siskind predictor.

    Formula (Siskind et al. 1980):
        Airblast (dB) = 165 - 25 × log10(D / W^(1/3))

    Where:
        D = distance from blast to receptor (m)
        W = maximum charge per delay (kg)

    Reference: Siskind, D.E., Stagg, M.S., Kopp, J.W., & Dowding, C.H.
    (1980). "Structure response and damage produced by ground vibration
    from surface mine blasting." US Bureau of Mines RI 8507.

    Returns:
        Airblast in dB (linear scale).
    """
    W = max(max_charge_per_delay_kg, 0.1)
    D = max(distance_m, 10.0)
    airblast_db = 165.0 - 25.0 * np.log10(D / (W ** (1.0 / 3.0)))
    return float(np.clip(airblast_db, 40.0, 140.0))


def lundborg_flyrock(
    charge_mass_kg: float,
    stemming_m: float,
    burden_m: float,
) -> float:
    """
    Flyrock distance (m) using the Lundborg model.

    Formula (Lundborg 1975, modified):
        R = K × (charge_mass)^0.5 × exp(-stemming / burden)

    Where:
        K = empirical constant (typically 260 for hard rock)
        charge_mass = charge per hole (kg)
        stemming = stemming length (m)
        burden = burden (m)

    Reference: Lundborg, N. (1975). "The probability of flyrock."
    Swedish Detonic Research Foundation Report DS 1975:5.

    Returns:
        Flyrock distance in meters.
    """
    K = 260.0
    R = K * (charge_mass_kg ** 0.5) * np.exp(-stemming_m / max(burden_m, 0.1))
    return float(np.clip(R, 5.0, 500.0))


def total_cost_per_tonne(
    drilling_cost_per_m: float,
    hole_depth_m: float,
    n_holes: int,
    explosive_cost_per_kg: float,
    charge_mass_per_hole_kg: float,
    labor_cost: float,
    tonnage: float,
) -> float:
    """
    Drill-and-blast cost per tonne (USD/t).

    Formula:
        total_cost = drilling + explosive + labor
        cost_per_tonne = total_cost / tonnage

    Where:
        drilling = drilling_cost_per_m × hole_depth × n_holes
        explosive = explosive_cost_per_kg × charge_mass_per_hole × n_holes
        labor = fixed labor and equipment cost

    Reference: BlasterOPT internal cost model. Calibrated to
    Debswana open-pit operating costs.
    """
    drilling = drilling_cost_per_m * hole_depth_m * n_holes
    explosive = explosive_cost_per_kg * charge_mass_per_hole_kg * n_holes
    total = drilling + explosive + labor_cost
    return float(total / max(tonnage, 1.0))
