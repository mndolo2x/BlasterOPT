"""
Advanced Timing Design Module for BlastOpt Botswana.
Provides electronic detonator timing design, optimization, and analysis.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class DetonatorType(Enum):
    """Types of detonators."""
    ELECTRONIC = "electronic"
    SHOCK_TUBE = "shock_tube"
    DETONATING_CORD = "detonating_cord"
    NON_ELECTRIC = "non_electric"


class DelayType(Enum):
    """Types of delay timing."""
    SURFACE_DELAY = "surface_delay"
    IN_HOLE_DELAY = "in_hole_delay"
    COMBINED = "combined"


@dataclass
class Detonator:
    """Represents a single detonator."""
    detonator_id: str
    hole_id: str
    detonator_type: DetonatorType
    delay_ms: int
    in_hole_delay_ms: int = 0
    surface_delay_ms: int = 0
    nominal_delay_ms: int = 0
    actual_delay_ms: int = 0
    tolerance_ms: int = 0
    status: str = "programmed"
    serial_number: Optional[str] = None


@dataclass
class TimingSequence:
    """Represents a complete timing sequence for a blast."""
    sequence_id: str
    blast_id: str
    detonators: List[Detonator]
    initiation_pattern: str  # "row_by_row", "V_initiation", "echelon", "custom"
    base_time_ms: int = 0
    created_at: str = ""
    version: int = 1


@dataclass
class TimingEffect:
    """Results of timing effect analysis."""
    fragmentation_improvement: float
    vibration_reduction: float
    face_burst_quality: float
    timing_efficiency: float
    overall_score: float


class TimingDesigner:
    """Designer for electronic detonator timing sequences."""

    def __init__(self):
        """Initialize the timing designer."""
        self.detonator_counter = 0

    def design_row_by_row_timing(
        self,
        holes: List,
        row_delay_ms: int = 42,
        hole_delay_ms: int = 17,
        start_delay_ms: int = 0
    ) -> List[Detonator]:
        """
        Design row-by-row timing sequence.

        Args:
            holes: List of blast holes
            row_delay_ms: Delay between rows (ms)
            hole_delay_ms: Delay between holes in row (ms)
            start_delay_ms: Starting delay (ms)

        Returns:
            List of detonators with timing
        """
        detonators = []

        # Group holes by row (assuming holes are ordered)
        current_row = 0
        row_holes = []

        # Simple row grouping based on Y coordinates
        if holes:
            y_coords = [hole.y for hole in holes]
            unique_y = sorted(set(y_coords))

            for hole in holes:
                row_index = unique_y.index(hole.y)
                delay = start_delay_ms + (row_index * row_delay_ms)

                detonator = self._create_detonator(
                    hole=hole,
                    delay_ms=delay,
                    in_hole_delay_ms=0,
                    surface_delay_ms=delay
                )
                detonators.append(detonator)

        return detonators

    def design_v_initiation_timing(
        self,
        holes: List,
        center_hole_delay_ms: int = 0,
        delay_increment_ms: int = 17,
        max_delay_ms: int = 500
    ) -> List[Detonator]:
        """
        Design V-initiation timing sequence.

        Args:
            holes: List of blast holes
            center_hole_delay_ms: Delay for center hole (ms)
            delay_increment_ms: Delay increment per step (ms)
            max_delay_ms: Maximum delay (ms)

        Returns:
            List of detonators with V-timing
        """
        detonators = []

        if not holes:
            return detonators

        # Find center point
        x_coords = [hole.x for hole in holes]
        y_coords = [hole.y for hole in holes]
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        # Calculate distance from center for each hole
        for hole in holes:
            distance = np.sqrt((hole.x - center_x)**2 + (hole.y - center_y)**2)

            # Calculate delay based on distance
            delay = center_hole_delay_ms + int(distance * delay_increment_ms / 10)
            delay = min(delay, max_delay_ms)

            detonator = self._create_detonator(
                hole=hole,
                delay_ms=delay,
                in_hole_delay_ms=0,
                surface_delay_ms=delay
            )
            detonators.append(detonator)

        return detonators

    def design_echelon_timing(
        self,
        holes: List,
        base_delay_ms: int = 0,
        row_delay_ms: int = 42,
        echelon_delay_ms: int = 17,
        direction: str = "forward"
    ) -> List[Detonator]:
        """
        Design echelon timing sequence.

        Args:
            holes: List of blast holes
            base_delay_ms: Base delay (ms)
            row_delay_ms: Delay between rows (ms)
            echelon_delay_ms: Echelon delay increment (ms)
            direction: "forward" or "backward"

        Returns:
            List of detonators with echelon timing
        """
        detonators = []

        if not holes:
            return detonators

        # Group holes by row
        y_coords = [hole.y for hole in holes]
        unique_y = sorted(set(y_coords))

        for hole in holes:
            row_index = unique_y.index(hole.y)

            # Calculate echelon delay based on position in row
            x_coords_in_row = [h.x for h in holes if h.y == hole.y]
            hole_index_in_row = sorted(x_coords_in_row).index(hole.x)

            if direction == "forward":
                echelon_delay = hole_index_in_row * echelon_delay_ms
            else:
                echelon_delay = (len(x_coords_in_row) - 1 - hole_index_in_row) * echelon_delay_ms

            total_delay = base_delay_ms + (row_index * row_delay_ms) + echelon_delay

            detonator = self._create_detonator(
                hole=hole,
                delay_ms=total_delay,
                in_hole_delay_ms=0,
                surface_delay_ms=total_delay
            )
            detonators.append(detonator)

        return detonators

    def design_combined_delay_timing(
        self,
        holes: List,
        in_hole_delay_ms: int = 200,
        surface_delay_ms: int = 42,
        row_delay_ms: int = 17
    ) -> List[Detonator]:
        """
        Design combined in-hole and surface delay timing.

        Args:
            holes: List of blast holes
            in_hole_delay_ms: In-hole delay (ms)
            surface_delay_ms: Surface delay (ms)
            row_delay_ms: Delay between surface delays (ms)

        Returns:
            List of detonators with combined timing
        """
        detonators = []

        if not holes:
            return detonators

        # Group holes by row
        y_coords = [hole.y for hole in holes]
        unique_y = sorted(set(y_coords))

        for hole in holes:
            row_index = unique_y.index(hole.y)

            # Surface delay varies by row
            surface_delay = surface_delay_ms + (row_index * row_delay_ms)

            # Total delay is surface + in-hole
            total_delay = surface_delay + in_hole_delay_ms

            detonator = self._create_detonator(
                hole=hole,
                delay_ms=total_delay,
                in_hole_delay_ms=in_hole_delay_ms,
                surface_delay_ms=surface_delay
            )
            detonators.append(detonator)

        return detonators

    def _create_detonator(
        self,
        hole,
        delay_ms: int,
        in_hole_delay_ms: int,
        surface_delay_ms: int
    ) -> Detonator:
        """
        Create a detonator object.

        Args:
            hole: Blast hole
            delay_ms: Total delay (ms)
            in_hole_delay_ms: In-hole delay (ms)
            surface_delay_ms: Surface delay (ms)

        Returns:
            Detonator object
        """
        self.detonator_counter += 1
        detonator_id = f"DET{self.detonator_counter:03d}"

        return Detonator(
            detonator_id=detonator_id,
            hole_id=hole.hole_id,
            detonator_type=DetonatorType.ELECTRONIC,
            delay_ms=delay_ms,
            in_hole_delay_ms=in_hole_delay_ms,
            surface_delay_ms=surface_delay_ms,
            nominal_delay_ms=delay_ms,
            actual_delay_ms=delay_ms,
            tolerance_ms=2,  # Electronic detonators typically have ±2ms tolerance
            status="programmed"
        )


class TimingEffectAnalyzer:
    """Analyzer for timing effects on fragmentation and vibration."""

    def __init__(self):
        """Initialize the timing effect analyzer."""
        pass

    def analyze_fragmentation_effect(
        self,
        timing_sequence: TimingSequence,
        burden_m: float,
        spacing_m: float,
        rock_factor_A: float = 8.0
    ) -> Dict[str, Any]:
        """
        Analyze timing effect on fragmentation.

        Args:
            timing_sequence: Timing sequence
            burden_m: Burden distance (m)
            spacing_m: Spacing distance (m)
            rock_factor_A: Rock factor

        Returns:
            Dictionary with fragmentation analysis
        """
        detonators = timing_sequence.detonators

        if not detonators:
            return {"error": "No detonators in sequence"}

        # Calculate timing statistics
        delays = [det.delay_ms for det in detonators]
        min_delay = min(delays)
        max_delay = max(delays)
        avg_delay = np.mean(delays)
        std_delay = np.std(delays)

        # Calculate timing efficiency
        # Optimal timing should have consistent spacing between delays
        if len(delays) > 1:
            delay_gaps = np.diff(sorted(delays))
            avg_gap = np.mean(delay_gaps)
            timing_uniformity = 1.0 - (np.std(delay_gaps) / avg_gap) if avg_gap > 0 else 0.0
        else:
            timing_uniformity = 1.0

        # Fragmentation improvement factor based on timing
        # Better timing improves fragmentation by 5-15%
        fragmentation_improvement = 0.05 + (0.10 * timing_uniformity)

        # Adjust d50 based on timing improvement
        base_d50 = 0.8 * rock_factor_A * (burden_m * spacing_m) ** 0.5
        improved_d50 = base_d50 * (1.0 - fragmentation_improvement)

        return {
            "min_delay_ms": min_delay,
            "max_delay_ms": max_delay,
            "avg_delay_ms": avg_delay,
            "std_delay_ms": std_delay,
            "timing_uniformity": timing_uniformity,
            "fragmentation_improvement": fragmentation_improvement,
            "base_d50_mm": base_d50,
            "improved_d50_mm": improved_d50,
            "d50_reduction_percent": fragmentation_improvement * 100
        }

    def analyze_vibration_effect(
        self,
        timing_sequence: TimingSequence,
        max_charge_per_delay_kg: float,
        monitoring_distance_m: float,
        site_constant_K: float = 1140
    ) -> Dict[str, Any]:
        """
        Analyze timing effect on ground vibration.

        Args:
            timing_sequence: Timing sequence
            max_charge_per_delay_kg: Maximum charge per delay (kg)
            monitoring_distance_m: Monitoring distance (m)
            site_constant_K: Site constant

        Returns:
            Dictionary with vibration analysis
        """
        detonators = timing_sequence.detonators

        if not detonators:
            return {"error": "No detonators in sequence"}

        # Calculate effective charge per delay based on timing
        # Electronic detonators can reduce effective charge per delay
        delays = [det.delay_ms for det in detonators]

        # Group detonators by delay windows (±17ms typically)
        delay_groups = self._group_by_delay_window(delays, window_ms=17)

        # Calculate max charge per delay based on timing
        max_charge_per_effective_delay = max_charge_per_delay_kg / max(1, len(delay_groups))

        # Calculate PPV with timing reduction
        ppv_untimed = site_constant_K * (max_charge_per_delay_kg ** 0.5) / (monitoring_distance_m ** 1.5)
        ppv_timed = site_constant_K * (max_charge_per_effective_delay ** 0.5) / (monitoring_distance_m ** 1.5)

        # Vibration reduction due to timing
        vibration_reduction = (ppv_untimed - ppv_timed) / ppv_untimed if ppv_untimed > 0 else 0.0

        return {
            "ppv_untimed_mms": ppv_untimed,
            "ppv_timed_mms": ppv_timed,
            "vibration_reduction_percent": vibration_reduction * 100,
            "max_charge_per_effective_delay_kg": max_charge_per_effective_delay,
            "num_delay_groups": len(delay_groups),
            "timing_effectiveness": 1.0 - (len(delay_groups) / len(detonators))
        }

    def _group_by_delay_window(self, delays: List[int], window_ms: int = 17) -> List[List[int]]:
        """
        Group delays by time windows.

        Args:
            delays: List of delay values
            window_ms: Window size for grouping (ms)

        Returns:
            List of delay groups
        """
        if not delays:
            return []

        sorted_delays = sorted(delays)
        groups = []
        current_group = [sorted_delays[0]]

        for delay in sorted_delays[1:]:
            if delay - current_group[0] <= window_ms:
                current_group.append(delay)
            else:
                groups.append(current_group)
                current_group = [delay]

        groups.append(current_group)
        return groups

    def analyze_face_burst_quality(
        self,
        timing_sequence: TimingSequence,
        bench_geometry,
        burden_m: float
    ) -> Dict[str, Any]:
        """
        Analyze face burst quality based on timing.

        Args:
            timing_sequence: Timing sequence
            bench_geometry: Bench geometry
            burden_m: Burden distance

        Returns:
            Dictionary with face burst analysis
        """
        detonators = timing_sequence.detonators

        if not detonators:
            return {"error": "No detonators in sequence"}

        # Analyze timing pattern
        delays = [det.delay_ms for det in detonators]

        # Check for proper face relief timing
        # Face relief holes should fire first with proper delays
        sorted_delays = sorted(delays)

        # Calculate timing progression
        if len(sorted_delays) > 1:
            delay_progression = np.diff(sorted_delays)
            avg_progression = np.mean(delay_progression)
            progression_uniformity = 1.0 - (np.std(delay_progression) / avg_progression) if avg_progression > 0 else 0.0
        else:
            progression_uniformity = 1.0

        # Face burst quality score
        # Based on timing uniformity and progression
        face_burst_quality = 0.5 * progression_uniformity + 0.5 * (1.0 - (std_delay := np.std(delays)) / (np.mean(delays) if np.mean(delays) > 0 else 1))

        # Check burden relief
        burden_adequacy = min(1.0, burden_m / (bench_geometry.bench_height_m * 0.8))

        return {
            "face_burst_quality": face_burst_quality,
            "timing_uniformity": progression_uniformity,
            "burden_adequacy": burden_adequacy,
            "avg_delay_progression_ms": avg_progression if len(sorted_delays) > 1 else 0,
            "min_delay_ms": min(delays),
            "max_delay_ms": max(delays),
            "recommendation": self._get_face_burst_recommendation(face_burst_quality, burden_adequacy)
        }

    def _get_face_burst_recommendation(self, quality: float, adequacy: float) -> str:
        """Get face burst recommendation."""
        if quality > 0.8 and adequacy > 0.8:
            return "Excellent timing and burden relief"
        elif quality > 0.6 and adequacy > 0.6:
            return "Good timing, consider minor adjustments"
        elif quality > 0.4:
            return "Fair timing, review progression"
        else:
            return "Poor timing, redesign recommended"


class TimingOptimizer:
    """Optimizer for timing sequences."""

    def __init__(self):
        """Initialize the timing optimizer."""
        self.designer = TimingDesigner()
        self.analyzer = TimingEffectAnalyzer()

    def optimize_for_fragmentation(
        self,
        holes: List,
        bench_geometry,
        burden_m: float,
        spacing_m: float,
        max_delay_ms: int = 500,
        timing_pattern: str = "v_initiation"
    ) -> Dict[str, Any]:
        """
        Optimize timing for fragmentation improvement.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            burden_m: Burden distance
            spacing_m: Spacing distance
            max_delay_ms: Maximum delay
            timing_pattern: Timing pattern to use

        Returns:
            Optimization results
        """
        # Generate timing sequence
        if timing_pattern == "row_by_row":
            detonators = self.designer.design_row_by_row_timing(holes)
        elif timing_pattern == "v_initiation":
            detonators = self.designer.design_v_initiation_timing(holes, max_delay_ms=max_delay_ms)
        elif timing_pattern == "echelon":
            detonators = self.designer.design_echelon_timing(holes)
        else:
            detonators = self.designer.design_row_by_row_timing(holes)

        # Create timing sequence
        sequence = TimingSequence(
            sequence_id="OPT_FRAG",
            blast_id="TEMP",
            detonators=detonators,
            initiation_pattern=timing_pattern,
            created_at=datetime.now().isoformat()
        )

        # Analyze fragmentation effect
        frag_analysis = self.analyzer.analyze_fragmentation_effect(
            sequence, burden_m, spacing_m
        )

        return {
            "timing_sequence": sequence,
            "fragmentation_analysis": frag_analysis,
            "optimization_pattern": timing_pattern,
            "num_detonators": len(detonators)
        }

    def optimize_for_vibration(
        self,
        holes: List,
        max_charge_per_delay_kg: float,
        monitoring_distance_m: float,
        max_delay_ms: int = 500,
        target_ppv_mms: float = 10.0
    ) -> Dict[str, Any]:
        """
        Optimize timing for vibration reduction.

        Args:
            holes: List of blast holes
            max_charge_per_delay_kg: Max charge per delay
            monitoring_distance_m: Monitoring distance
            max_delay_ms: Maximum delay
            target_ppv_mms: Target PPV

        Returns:
            Optimization results
        """
        # Generate timing with optimal spacing for vibration
        detonators = self.designer.design_row_by_row_timing(
            holes,
            row_delay_ms=42,
            hole_delay_ms=17
        )

        # Create timing sequence
        sequence = TimingSequence(
            sequence_id="OPT_VIB",
            blast_id="TEMP",
            detonators=detonators,
            initiation_pattern="row_by_row",
            created_at=datetime.now().isoformat()
        )

        # Analyze vibration effect
        vib_analysis = self.analyzer.analyze_vibration_effect(
            sequence, max_charge_per_delay_kg, monitoring_distance_m
        )

        # Check if target achieved
        target_achieved = vib_analysis["ppv_timed_mms"] <= target_ppv_mms

        return {
            "timing_sequence": sequence,
            "vibration_analysis": vib_analysis,
            "target_ppv_mms": target_ppv_mms,
            "target_achieved": target_achieved,
            "num_detonators": len(detonators)
        }

    def optimize_multi_objective(
        self,
        holes: List,
        bench_geometry,
        burden_m: float,
        spacing_m: float,
        max_charge_per_delay_kg: float,
        monitoring_distance_m: float,
        max_delay_ms: int = 500
    ) -> Dict[str, Any]:
        """
        Multi-objective optimization for fragmentation and vibration.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            burden_m: Burden distance
            spacing_m: Spacing distance
            max_charge_per_delay_kg: Max charge per delay
            monitoring_distance_m: Monitoring distance
            max_delay_ms: Maximum delay

        Returns:
            Multi-objective optimization results
        """
        # Test different timing patterns
        patterns = ["row_by_row", "v_initiation", "echelon"]
        results = []

        for pattern in patterns:
            # Generate timing
            if pattern == "row_by_row":
                detonators = self.designer.design_row_by_row_timing(holes)
            elif pattern == "v_initiation":
                detonators = self.designer.design_v_initiation_timing(holes, max_delay_ms=max_delay_ms)
            else:
                detonators = self.designer.design_echelon_timing(holes)

            sequence = TimingSequence(
                sequence_id=f"OPT_{pattern.upper()}",
                blast_id="TEMP",
                detonators=detonators,
                initiation_pattern=pattern,
                created_at=datetime.now().isoformat()
            )

            # Analyze effects
            frag_analysis = self.analyzer.analyze_fragmentation_effect(
                sequence, burden_m, spacing_m
            )
            vib_analysis = self.analyzer.analyze_vibration_effect(
                sequence, max_charge_per_delay_kg, monitoring_distance_m
            )

            results.append({
                "pattern": pattern,
                "timing_sequence": sequence,
                "fragmentation_analysis": frag_analysis,
                "vibration_analysis": vib_analysis,
                "combined_score": frag_analysis["fragmentation_improvement"] + vib_analysis["vibration_reduction"]
            })

        # Select best pattern
        best_result = max(results, key=lambda x: x["combined_score"])

        return {
            "best_pattern": best_result["pattern"],
            "best_timing_sequence": best_result["timing_sequence"],
            "best_fragmentation_analysis": best_result["fragmentation_analysis"],
            "best_vibration_analysis": best_result["vibration_analysis"],
            "all_results": results
        }


class TimingErrorAnalyzer:
    """Analyzer for timing error tolerance and effects."""

    def __init__(self):
        """Initialize the timing error analyzer."""
        pass

    def analyze_timing_error_effects(
        self,
        timing_sequence: TimingSequence,
        error_std_ms: float = 2.0
    ) -> Dict[str, Any]:
        """
        Analyze effects of timing errors.

        Args:
            timing_sequence: Timing sequence
            error_std_ms: Standard deviation of timing error (ms)

        Returns:
            Dictionary with error analysis
        """
        detonators = timing_sequence.detonators

        if not detonators:
            return {"error": "No detonators in sequence"}

        # Simulate timing errors
        actual_delays = []
        for det in detonators:
            # Add random error (normal distribution)
            error = np.random.normal(0, error_std_ms)
            actual_delay = det.nominal_delay_ms + error
            actual_delays.append(actual_delay)

        # Calculate impact on timing groups
        nominal_groups = self.analyzer._group_by_delay_window([det.delay_ms for det in detonators])
        actual_groups = self.analyzer._group_by_delay_window(actual_delays)

        # Calculate group disruption
        group_disruption = abs(len(nominal_groups) - len(actual_groups)) / max(1, len(nominal_groups))

        # Calculate delay variance
        delay_variance = np.var(actual_delays)

        return {
            "error_std_ms": error_std_ms,
            "nominal_groups": len(nominal_groups),
            "actual_groups": len(actual_groups),
            "group_disruption": group_disruption,
            "delay_variance": delay_variance,
            "tolerance_adequacy": 1.0 - group_disruption,
            "recommendation": self._get_error_recommendation(group_disruption, error_std_ms)
        }

    def _group_by_delay_window(self, delays: List[int], window_ms: int = 17) -> List[List[int]]:
        """Group delays by time windows."""
        if not delays:
            return []

        sorted_delays = sorted(delays)
        groups = []
        current_group = [sorted_delays[0]]

        for delay in sorted_delays[1:]:
            if delay - current_group[0] <= window_ms:
                current_group.append(delay)
            else:
                groups.append(current_group)
                current_group = [delay]

        groups.append(current_group)
        return groups

    def _get_error_recommendation(self, disruption: float, error_std: float) -> str:
        """Get error tolerance recommendation."""
        if disruption < 0.1 and error_std < 2.0:
            return "Excellent tolerance, errors within acceptable range"
        elif disruption < 0.3:
            return "Acceptable tolerance, minor timing group disruption"
        elif disruption < 0.5:
            return "Marginal tolerance, consider increasing delay windows"
        else:
            return "Poor tolerance, redesign timing or improve detonator quality"
