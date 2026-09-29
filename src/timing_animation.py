"""
Initiation Sequence Animation Module for BlastOpt Botswana.
Provides animated visualization of blast timing sequences.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
from src.timing_design import TimingSequence, Detonator, TimingEffectAnalyzer


class SequenceAnimator:
    """Animator for blast initiation sequences."""

    def __init__(self):
        """Initialize the sequence animator."""
        self.analyzer = TimingEffectAnalyzer()

    def create_timing_animation(
        self,
        timing_sequence: TimingSequence,
        animation_duration_ms: int = 2000,
        frame_interval_ms: int = 50
    ) -> go.Figure:
        """
        Create animated timing sequence visualization.

        Args:
            timing_sequence: Timing sequence
            animation_duration_ms: Total animation duration (ms)
            frame_interval_ms: Frame interval (ms)

        Returns:
            Plotly figure with animation
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        # Extract coordinates and timing
        x_coords = [det.hole_id for det in detonators]  # Will map to actual coordinates
        y_coords = [det.delay_ms for det in detonators]

        # Create scatter plot for timing sequence
        fig = go.Figure()

        # Add holes as scatter points
        for det in detonators:
            fig.add_trace(go.Scatter(
                x=[det.hole_id],
                y=[det.delay_ms],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name=det.hole_id,
                text=f"{det.hole_id}: {det.delay_ms}ms",
                showlegend=False
            ))

        # Add animation frames
        frames = []
        time_points = np.arange(0, animation_duration_ms, frame_interval_ms)

        for t in time_points:
            frame_data = []
            for det in detonators:
                # Check if this detonator should be "fired" at this time
                if det.delay_ms <= t:
                    frame_data.append({
                        'x': [det.hole_id],
                        'y': [det.delay_ms],
                        'marker': {'size': 20, 'color': 'red'},
                        'mode': 'markers'
                    })
                else:
                    frame_data.append({
                        'x': [det.hole_id],
                        'y': [det.delay_ms],
                        'marker': {'size': 15, 'color': 'blue'},
                        'mode': 'markers'
                    })

            frames.append(go.Frame(data=[trace for trace in frame_data]))

        fig.update_layout(
            title='Blast Initiation Sequence Animation',
            xaxis_title='Hole ID',
            yaxis_title='Delay (ms)',
            height=600,
            width=1000
        )

        return fig

    def create_3d_timing_animation(
        self,
        timing_sequence: TimingSequence,
        hole_coordinates: Dict[str, tuple],  # hole_id -> (x, y, z)
        animation_duration_ms: int = 2000,
        frame_interval_ms: int = 50
    ) -> go.Figure:
        """
        Create 3D animated timing sequence.

        Args:
            timing_sequence: Timing sequence
            hole_coordinates: Dictionary mapping hole IDs to 3D coordinates
            animation_duration_ms: Total animation duration
            frame_interval_ms: Frame interval

        Returns:
            Plotly 3D figure with animation
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        fig = go.Figure()

        # Create animation frames
        frames = []
        time_points = np.arange(0, animation_duration_ms, frame_interval_ms)

        for t in time_points:
            # Create frame data
            fired_holes = []
            unfired_holes = []

            for det in detonators:
                if det.hole_id in hole_coordinates:
                    x, y, z = hole_coordinates[det.hole_id]

                    if det.delay_ms <= t:
                        fired_holes.append({
                            'x': [x],
                            'y': [y],
                            'z': [z],
                            'marker': {'size': 10, 'color': 'red'},
                            'mode': 'markers',
                            'name': det.hole_id
                        })
                    else:
                        unfired_holes.append({
                            'x': [x],
                            'y': [y],
                            'z': [z],
                            'marker': {'size': 8, 'color': 'blue'},
                            'mode': 'markers',
                            'name': det.hole_id
                        })

            # Add traces for this frame
            frame_traces = []

            for hole_data in fired_holes:
                frame_traces.append(go.Scatter3d(**hole_data, showlegend=False))

            for hole_data in unfired_holes:
                frame_traces.append(go.Scatter3d(**hole_data, showlegend=False))

            frames.append(go.Frame(data=frame_traces))

        # Initial frame (all unfired)
        initial_traces = []
        for det in detonators:
            if det.hole_id in hole_coordinates:
                x, y, z = hole_coordinates[det.hole_id]
                initial_traces.append(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    name=det.hole_id,
                    showlegend=False
                ))

        fig.update_layout(
            title='3D Blast Initiation Sequence Animation',
            scene=dict(
                xaxis_title='East (m)',
                yaxis_title='North (m)',
                zaxis_title='Elevation (m)',
                aspectmode='data'
            ),
            height=800,
            width=1200
        )

        return fig

    def create_face_burst_animation(
        self,
        timing_sequence: TimingSequence,
        hole_coordinates: Dict[str, tuple],
        bench_face_coordinates: List[tuple],
        animation_duration_ms: int = 2000
    ) -> go.Figure:
        """
        Create face burst animation showing relief progression.

        Args:
            timing_sequence: Timing sequence
            hole_coordinates: Hole coordinates
            bench_face_coordinates: Bench face coordinates
            animation_duration_ms: Animation duration

        Returns:
            Plotly figure with face burst animation
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        fig = go.Figure()

        # Create animation frames
        frames = []

        # Plot bench face
        face_x = [coord[0] for coord in bench_face_coordinates]
        face_y = [coord[1] for coord in bench_face_coordinates]
        face_z = [coord[2] for coord in bench_face_coordinates]

        fig.add_trace(go.Scatter3d(
            x=face_x, y=face_y, z=face_z,
            mode='lines',
            line=dict(width=5, color='orange'),
            name='Bench Face'
        ))

        # Create animation frames
        time_points = np.arange(0, animation_duration_ms, 100)

        for t in time_points:
            frame_traces = [go.Scatter3d(  # Keep bench face
                x=face_x, y=face_y, z=face_z,
                mode='lines',
                line=dict(width=5, color='orange'),
                name='Bench Face',
                showlegend=False
            )]

            # Add holes based on timing
            for det in detonators:
                if det.hole_id in hole_coordinates:
                    x, y, z = hole_coordinates[det.hole_id]

                    if det.delay_ms <= t:
                        # Show fired hole with face burst line
                        frame_traces.append(go.Scatter3d(
                            x=[x, x + 5],  # Show burst direction
                            y=[y, y],
                            z=[z, z - 2],
                            mode='lines',
                            line=dict(width=3, color='red'),
                            name=f'{det.hole_id} Burst',
                            showlegend=False
                        ))

            frames.append(go.Frame(data=frame_traces))

        fig.update_layout(
            title='Face Burst Progression Animation',
            scene=dict(
                xaxis_title='East (m)',
                yaxis_title='North (m)',
                zaxis_title='Elevation (m)',
                aspectmode='data'
            ),
            height=800,
            width=1200
        )

        return fig

    def create_timing_statistics_dashboard(
        self,
        timing_sequence: TimingSequence
    ) -> go.Figure:
        """
        Create comprehensive timing statistics dashboard.

        Args:
            timing_sequence: Timing sequence

        Returns:
            Plotly figure with statistics
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        # Extract timing data
        delays = [det.delay_ms for det in detonators]
        in_hole_delays = [det.in_hole_delay_ms for det in detonators]
        surface_delays = [det.surface_delay_ms for det in detonators]

        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delay Distribution', 'Delay Timeline',
                          'In-Hole vs Surface Delay', 'Delay Statistics'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Delay distribution histogram
        fig.add_trace(go.Histogram(x=delays, name='Delays', nbinsx=20), row=1, col=1)

        # Delay timeline
        fig.add_trace(go.Scatter(
            x=list(range(len(delays))),
            y=sorted(delays),
            mode='lines+markers',
            name='Timeline',
            line=dict(color='blue', width=2)
        ), row=1, col=2)

        # In-hole vs surface delay scatter
        fig.add_trace(go.Scatter(
            x=surface_delays,
            y=in_hole_delays,
            mode='markers',
            name='In-Hole vs Surface',
            marker=dict(size=8, color='green')
        ), row=2, col=1)

        # Delay statistics bar chart
        stats = {
            'Min Delay': min(delays),
            'Max Delay': max(delays),
            'Avg Delay': np.mean(delays),
            'Std Delay': np.std(delays)
        }

        fig.add_trace(go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            name='Statistics'
        ), row=2, col=2)

        fig.update_layout(
            title='Timing Sequence Statistics Dashboard',
            height=800,
            width=1200,
            showlegend=False
        )

        return fig


class WaveformGenerator:
    """Generator for vibration waveforms based on timing."""

    def __init__(self):
        """Initialize the waveform generator."""
        pass

    def generate_ppv_waveform(
        self,
        timing_sequence: TimingSequence,
        monitoring_distance_m: float,
        site_constant_K: float = 1140,
        max_charge_kg: float = 500.0
    ) -> go.Figure:
        """
        Generate simulated PPV waveform based on timing sequence.

        Args:
            timing_sequence: Timing sequence
            monitoring_distance_m: Monitoring distance
            site_constant_K: Site constant
            max_charge_kg: Maximum charge

        Returns:
            Plotly figure with waveform
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        # Generate time series
        max_delay = max(det.delay_ms for det in detonators)
        time_series = np.arange(0, max_delay + 500, 10)  # 10ms intervals

        # Calculate PPV for each time point
        ppv_values = []

        for t in time_series:
            # Calculate effective charge at this time
            effective_charge = 0.0

            for det in detonators:
                # If detonator fires around this time, contribute to PPV
                if abs(det.delay_ms - t) <= 50:  # 50ms window
                    effective_charge += max_charge_kg / len(detonators)

            # Calculate PPV using scaled distance formula
            if effective_charge > 0:
                ppv = site_constant_K * (effective_charge ** 0.5) / (monitoring_distance_m ** 1.5)
            else:
                ppv = 0.0

            ppv_values.append(ppv)

        # Create waveform plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_series,
            y=ppv_values,
            mode='lines',
            line=dict(color='blue', width=2),
            name='PPV Waveform'
        ))

        # Add timing markers
        for det in detonators:
            fig.add_vline(
                x=det.delay_ms,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{det.hole_id}"
            )

        fig.update_layout(
            title='Simulated PPV Waveform Based on Timing',
            xaxis_title='Time (ms)',
            yaxis_title='PPV (mm/s)',
            height=500,
            width=1000
        )

        return fig

    def generate_fragmentation_timeline(
        self,
        timing_sequence: TimingSequence,
        fragmentation_progression: List[float]
    ) -> go.Figure:
        """
        Generate fragmentation progression timeline.

        Args:
            timing_sequence: Timing sequence
            fragmentation_progression: List of fragmentation values over time

        Returns:
            Plotly figure with timeline
        """
        detonators = timing_sequence.detonators

        if not detonators:
            fig = go.Figure()
            fig.update_layout(title="No detonators in sequence")
            return fig

        # Generate time points
        max_delay = max(det.delay_ms for det in detonators)
        time_points = np.arange(0, max_delay + 500, 50)

        # Interpolate fragmentation progression
        if len(fragmentation_progression) > 0:
            # Simple linear interpolation
            frag_times = np.linspace(0, max_delay, len(fragmentation_progression))
            frag_values = np.interp(
                np.linspace(0, 100, len(time_points)),
                np.linspace(0, 100, len(fragmentation_progression)),
                fragmentation_progression
            )
        else:
            # Default linear progression
            frag_values = np.linspace(0, 100, len(time_points))

        # Create timeline plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_points,
            y=frag_values,
            mode='lines+markers',
            line=dict(color='green', width=2),
            name='Fragmentation %'
        ))

        # Add timing markers
        for det in detonators:
            fig.add_vline(
                x=det.delay_ms,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{det.hole_id}"
            )

        fig.update_layout(
            title='Fragmentation Progression Timeline',
            xaxis_title='Time (ms)',
            yaxis_title='Fragmentation (%)',
            height=500,
            width=1000
        )

        return fig


def create_timing_summary_report(
    timing_sequence: TimingSequence,
    fragmentation_analysis: Dict[str, Any],
    vibration_analysis: Dict[str, Any],
    face_burst_analysis: Dict[str, Any]
) -> str:
    """
    Create comprehensive timing summary report.

    Args:
        timing_sequence: Timing sequence
        fragmentation_analysis: Fragmentation analysis results
        vibration_analysis: Vibration analysis results
        face_burst_analysis: Face burst analysis results

    Returns:
        String report
    """
    report = "=" * 60 + "\n"
    report += "TIMING DESIGN SUMMARY REPORT\n"
    report += "=" * 60 + "\n\n"

    report += f"Sequence ID: {timing_sequence.sequence_id}\n"
    report += f"Initiation Pattern: {timing_sequence.initiation_pattern}\n"
    report += f"Number of Detonators: {len(timing_sequence.detonators)}\n"
    report += f"Created: {timing_sequence.created_at}\n\n"

    report += "-" * 60 + "\n"
    report += "FRAGMENTATION ANALYSIS\n"
    report += "-" * 60 + "\n"
    report += f"Base d50: {fragmentation_analysis.get('base_d50_mm', 0):.1f} mm\n"
    report += f"Improved d50: {fragmentation_analysis.get('improved_d50_mm', 0):.1f} mm\n"
    report += f"D50 Reduction: {fragmentation_analysis.get('d50_reduction_percent', 0):.1f}%\n"
    report += f"Timing Uniformity: {fragmentation_analysis.get('timing_uniformity', 0):.3f}\n\n"

    report += "-" * 60 + "\n"
    report += "VIBRATION ANALYSIS\n"
    report += "-" * 60 + "\n"
    report += f"Untimed PPV: {vibration_analysis.get('ppv_untimed_mms', 0):.2f} mm/s\n"
    report += f"Timed PPV: {vibration_analysis.get('ppv_timed_mms', 0):.2f} mm/s\n"
    report += f"Vibration Reduction: {vibration_analysis.get('vibration_reduction_percent', 0):.1f}%\n"
    report += f"Timing Effectiveness: {vibration_analysis.get('timing_effectiveness', 0):.3f}\n\n"

    report += "-" * 60 + "\n"
    report += "FACE BURST ANALYSIS\n"
    report += "-" * 60 + "\n"
    report += f"Face Burst Quality: {face_burst_analysis.get('face_burst_quality', 0):.3f}\n"
    report += f"Burden Adequacy: {face_burst_analysis.get('burden_adequacy', 0):.3f}\n"
    report += f"Recommendation: {face_burst_analysis.get('recommendation', 'N/A')}\n\n"

    report += "=" * 60 + "\n"
    report += "END OF REPORT\n"
    report += "=" * 60 + "\n"

    return report
