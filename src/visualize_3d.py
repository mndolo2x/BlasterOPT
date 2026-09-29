"""
3D Visualization Module for BlastOpt Botswana.
Provides interactive 3D visualization for blast patterns and designs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
from src.blast_pattern_3d import BlastHole, BenchGeometry, BlastPattern3D, PatternType
from src.timing_design import TimingSequence, Detonator, TimingDesigner


class PatternVisualizer3D:
    """3D visualizer for blast patterns."""

    def __init__(self):
        """Initialize the 3D visualizer."""
        self.color_palette = {
            'collar': 'red',
            'toe': 'blue',
            'hole': 'green',
            'bench_surface': 'lightblue',
            'face': 'orange',
            'deck1': 'red',
            'deck2': 'blue',
            'deck3': 'green'
        }

    def plot_3d_pattern(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        show_bench: bool = True,
        show_face: bool = True,
        show_deviations: bool = False,
        show_annotations: bool = True
    ) -> go.Figure:
        """
        Create 3D visualization of blast pattern.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            show_bench: Show bench surface
            show_face: Show bench face
            show_deviations: Show hole deviations
            show_annotations: Show hole annotations

        Returns:
            Plotly 3D figure
        """
        fig = go.Figure()

        # Plot bench surface
        if show_bench:
            self._add_bench_surface(fig, bench_geometry)

        # Plot bench face
        if show_face:
            self._add_bench_face(fig, bench_geometry)

        # Plot holes
        self._add_holes_3d(fig, holes, show_deviations, show_annotations)

        # Set layout
        fig.update_layout(
            title='3D Blast Pattern Design',
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

    def _add_bench_surface(self, fig: go.Figure, bench_geometry: BenchGeometry):
        """
        Add bench surface to 3D plot.

        Args:
            fig: Plotly figure
            bench_geometry: Bench geometry
        """
        # Create bench surface mesh
        x = np.linspace(0, bench_geometry.bench_length_m, 20)
        y = np.linspace(0, bench_geometry.bench_width_m, 20)
        X, Y = np.meshgrid(x, y)

        # Simple flat surface at crest elevation
        Z = np.full_like(X, bench_geometry.crest_elevation)

        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Blues',
            opacity=0.3,
            name='Bench Surface',
            showscale=False
        ))

    def _add_bench_face(self, fig: go.Figure, bench_geometry: BenchGeometry):
        """
        Add bench face to 3D plot.

        Args:
            fig: Plotly figure
            bench_geometry: Bench geometry
        """
        # Create face surface
        x = np.linspace(0, bench_geometry.bench_length_m, 20)
        z = np.linspace(bench_geometry.toe_elevation, bench_geometry.crest_elevation, 20)
        X, Z = np.meshgrid(x, z)

        # Calculate Y based on face angle
        Y = np.full_like(X, bench_geometry.bench_width_m)

        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Oranges',
            opacity=0.5,
            name='Bench Face',
            showscale=False
        ))

    def _add_holes_3d(
        self,
        fig: go.Figure,
        holes: List[BlastHole],
        show_deviations: bool,
        show_annotations: bool
    ):
        """
        Add 3D holes to plot.

        Args:
            fig: Plotly figure
            holes: List of blast holes
            show_deviations: Show deviation vectors
            show_annotations: Show hole ID annotations
        """
        for hole in holes:
            # Plot collar position
            fig.add_trace(go.Scatter3d(
                x=[hole.x],
                y=[hole.y],
                z=[hole.collar_elevation],
                mode='markers',
                marker=dict(size=8, color=self.color_palette['collar']),
                name=f'{hole.hole_id} Collar',
                showlegend=False
            ))

            # Plot toe position
            fig.add_trace(go.Scatter3d(
                x=[hole.x],
                y=[hole.y],
                z=[hole.toe_elevation],
                mode='markers',
                marker=dict(size=6, color=self.color_palette['toe']),
                name=f'{hole.hole_id} Toe',
                showlegend=False
            ))

            # Plot hole as line
            fig.add_trace(go.Scatter3d(
                x=[hole.x, hole.x],
                y=[hole.y, hole.y],
                z=[hole.collar_elevation, hole.toe_elevation],
                mode='lines',
                line=dict(width=4, color=self.color_palette['hole']),
                name=f'{hole.hole_id} Hole',
                showlegend=False
            ))

            # Show deviation vectors if requested
            if show_deviations and (hole.deviation_x != 0 or hole.deviation_y != 0):
                dev_length = hole.hole_length * 0.2  # Show 20% of hole length
                dev_x = dev_length * np.tan(hole.deviation_x)
                dev_y = dev_length * np.tan(hole.deviation_y)

                fig.add_trace(go.Scatter3d(
                    x=[hole.x, hole.x + dev_x],
                    y=[hole.y, hole.y + dev_y],
                    z=[hole.toe_elevation, hole.toe_elevation],
                    mode='lines',
                    line=dict(width=2, color='purple', dash='dash'),
                    name=f'{hole.hole_id} Deviation',
                    showlegend=False
                ))

            # Add annotation if requested
            if show_annotations:
                fig.add_trace(go.Scatter3d(
                    x=[hole.x],
                    y=[hole.y],
                    z=[hole.collar_elevation + 2],  # Offset above collar
                    mode='text',
                    text=[hole.hole_id],
                    textfont=dict(size=10),
                    name=f'{hole.hole_id} Label',
                    showlegend=False
                ))

    def plot_timing_sequence_3d(
        self,
        holes: List[BlastHole],
        timing_ms: List[int],
        bench_geometry: BenchGeometry
    ) -> go.Figure:
        """
        Plot 3D pattern with timing sequence visualization.

        Args:
            holes: List of blast holes
            timing_ms: Timing sequence in milliseconds
            bench_geometry: Bench geometry

        Returns:
            Plotly 3D figure with timing colors
        """
        fig = go.Figure()

        # Normalize timing for color mapping
        max_timing = max(timing_ms) if timing_ms else 1
        min_timing = min(timing_ms) if timing_ms else 0

        # Color map based on timing
        for hole, timing in zip(holes, timing_ms):
            # Calculate color based on timing
            timing_normalized = (timing - min_timing) / (max_timing - min_timing) if max_timing > min_timing else 0

            # Use a color scale from blue (early) to red (late)
            color = self._timing_to_color(timing_normalized)

            # Plot hole with timing color
            fig.add_trace(go.Scatter3d(
                x=[hole.x],
                y=[hole.y],
                z=[hole.collar_elevation],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f'{hole.hole_id} ({timing}ms)',
                text=f'{hole.hole_id}: {timing}ms',
                showlegend=False
            ))

        # Add timing legend
        self._add_timing_legend(fig, min_timing, max_timing)

        fig.update_layout(
            title='3D Blast Pattern with Timing Sequence',
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

    def _timing_to_color(self, normalized_timing: float) -> str:
        """
        Convert normalized timing to color.

        Args:
            normalized_timing: Normalized timing (0-1)

        Returns:
            Color string
        """
        # Simple blue to red gradient
        r = int(255 * normalized_timing)
        b = int(255 * (1 - normalized_timing))
        g = 0
        return f'rgb({r}, {g}, {b})'

    def _add_timing_legend(self, fig: go.Figure, min_timing: int, max_timing: int):
        """
        Add timing legend to figure.

        Args:
            fig: Plotly figure
            min_timing: Minimum timing value
            max_timing: Maximum timing value
        """
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=f'Early: {min_timing}ms',
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'Late: {max_timing}ms',
            showlegend=True
        ))

    def plot_decking_3d(
        self,
        holes: List[BlastHole],
        decking_configs: List[Dict[str, Any]],
        bench_geometry: BenchGeometry
    ) -> go.Figure:
        """
        Plot 3D pattern with decking visualization.

        Args:
            holes: List of blast holes
            decking_configs: List of decking configurations
            bench_geometry: Bench geometry

        Returns:
            Plotly 3D figure with decked holes
        """
        fig = go.Figure()

        # Plot bench surface
        self._add_bench_surface(fig, bench_geometry)

        # Plot decked holes
        for hole, deck_config in zip(holes, decking_configs):
            deck_positions = deck_config.get('deck_positions', [])

            for deck_pos in deck_positions:
                deck_number = deck_pos['deck_number']
                start_depth = deck_pos['start_depth_m']
                end_depth = deck_pos['end_depth_m']

                # Calculate elevation for this deck
                deck_top_elevation = hole.collar_elevation - start_depth
                deck_bottom_elevation = hole.collar_elevation - end_depth

                # Choose color based on deck number
                color = self.color_palette.get(f'deck{deck_number}', 'gray')

                # Plot deck as colored line segment
                fig.add_trace(go.Scatter3d(
                    x=[hole.x, hole.x],
                    y=[hole.y, hole.y],
                    z=[deck_top_elevation, deck_bottom_elevation],
                    mode='lines',
                    line=dict(width=6, color=color),
                    name=f'{hole.hole_id} Deck {deck_number}',
                    showlegend=False
                ))

        fig.update_layout(
            title='3D Blast Pattern with Decking',
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

    def plot_cross_section(
        self,
        holes: List[BlastHole],
        bench_geometry: BenchGeometry,
        section_offset: float = 0.0,
        tolerance: float = 1.0
    ) -> go.Figure:
        """
        Plot 2D cross-section of blast pattern.

        Args:
            holes: List of blast holes
            bench_geometry: Bench geometry
            section_offset: Y coordinate offset for cross-section
            tolerance: Tolerance for including holes in section

        Returns:
            Plotly 2D figure
        """
        fig = go.Figure()

        # Filter holes near the cross-section
        section_holes = [
            hole for hole in holes
            if abs(hole.y - section_offset) <= tolerance
        ]

        # Plot bench profile
        bench_x = [0, bench_geometry.bench_length_m]
        bench_z = [bench_geometry.crest_elevation, bench_geometry.toe_elevation]

        fig.add_trace(go.Scatter(
            x=bench_x,
            y=bench_z,
            mode='lines',
            line=dict(width=3, color='orange'),
            name='Bench Profile'
        ))

        # Plot holes in cross-section
        for hole in section_holes:
            # Plot collar
            fig.add_trace(go.Scatter(
                x=[hole.x],
                y=[hole.collar_elevation],
                mode='markers',
                marker=dict(size=10, color='red'),
                name=f'{hole.hole_id} Collar',
                showlegend=False
            ))

            # Plot hole
            fig.add_trace(go.Scatter(
                x=[hole.x, hole.x],
                y=[hole.collar_elevation, hole.toe_elevation],
                mode='lines',
                line=dict(width=4, color='green'),
                name=f'{hole.hole_id} Hole',
                showlegend=False
            ))

            # Plot toe
            fig.add_trace(go.Scatter(
                x=[hole.x],
                y=[hole.toe_elevation],
                mode='markers',
                marker=dict(size=8, color='blue'),
                name=f'{hole.hole_id} Toe',
                showlegend=False
            ))

        fig.update_layout(
            title=f'Cross-Section at Y = {section_offset:.1f}m',
            xaxis_title='East (m)',
            yaxis_title='Elevation (m)',
            height=600,
            width=1000
        )

        return fig

    def plot_pattern_statistics(
        self,
        holes: List[BlastHole],
        timing_ms: Optional[List[int]] = None
    ) -> go.Figure:
        """
        Plot pattern statistics and metrics.

        Args:
            holes: List of blast holes
            timing_ms: Optional timing sequence

        Returns:
            Plotly figure with statistics
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hole Length Distribution', 'Charge Mass Distribution',
                          'Burden Distribution', 'Spacing Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'histogram'}]]
        )

        # Extract data
        hole_lengths = [hole.hole_length for hole in holes]
        charge_masses = [hole.charge_mass_kg for hole in holes]
        burdens = [hole.burden_m for hole in holes]
        spacings = [hole.spacing_m for hole in holes]

        # Plot histograms
        fig.add_trace(go.Histogram(x=hole_lengths, name='Hole Length'), row=1, col=1)
        fig.add_trace(go.Histogram(x=charge_masses, name='Charge Mass'), row=1, col=2)
        fig.add_trace(go.Histogram(x=burdens, name='Burden'), row=2, col=1)
        fig.add_trace(go.Histogram(x=spacings, name='Spacing'), row=2, col=2)

        fig.update_layout(
            title='Blast Pattern Statistics',
            height=800,
            width=1200,
            showlegend=False
        )

        return fig

    def plot_timing_sequence_3d(
        self,
        holes: List[BlastHole],
        timing_sequence: TimingSequence,
        bench_geometry: BenchGeometry,
        show_timing_colors: bool = True
    ) -> go.Figure:
        """
        Plot 3D pattern with timing sequence visualization.

        Args:
            holes: List of blast holes
            timing_sequence: Timing sequence
            bench_geometry: Bench geometry
            show_timing_colors: Color holes by timing

        Returns:
            Plotly 3D figure with timing colors
        """
        fig = go.Figure()

        # Plot bench surface
        self._add_bench_surface(fig, bench_geometry)

        # Plot bench face
        self._add_bench_face(fig, bench_geometry)

        # Create hole ID to detonator mapping
        hole_detonator_map = {det.hole_id: det for det in timing_sequence.detonators}

        # Normalize timing for color mapping
        if timing_sequence.detonators:
            timings = [det.delay_ms for det in timing_sequence.detonators]
            max_timing = max(timings) if timings else 1
            min_timing = min(timings) if timings else 0

        # Plot holes with timing colors
        for hole in holes:
            if hole.hole_id in hole_detonator_map:
                detonator = hole_detonator_map[hole.hole_id]

                if show_timing_colors:
                    # Color based on timing
                    timing_normalized = (detonator.delay_ms - min_timing) / (max_timing - min_timing) if max_timing > min_timing else 0
                    color = self._timing_to_color(timing_normalized)
                else:
                    color = self.color_palette['hole']

                # Plot collar
                fig.add_trace(go.Scatter3d(
                    x=[hole.x],
                    y=[hole.y],
                    z=[hole.collar_elevation],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'{hole.hole_id} ({detonator.delay_ms}ms)',
                    text=f'{hole.hole_id}: {detonator.delay_ms}ms',
                    showlegend=False
                ))

                # Plot hole
                fig.add_trace(go.Scatter3d(
                    x=[hole.x, hole.x],
                    y=[hole.y, hole.y],
                    z=[hole.collar_elevation, hole.toe_elevation],
                    mode='lines',
                    line=dict(width=4, color=color),
                    name=f'{hole.hole_id} Hole',
                    showlegend=False
                ))

        # Add timing legend
        if show_timing_colors:
            self._add_timing_legend(fig, min_timing, max_timing)

        fig.update_layout(
            title='3D Blast Pattern with Timing Sequence',
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

    def _timing_to_color(self, normalized_timing: float) -> str:
        """
        Convert normalized timing to color.

        Args:
            normalized_timing: Normalized timing (0-1)

        Returns:
            Color string
        """
        # Color gradient from blue (early) to red (late)
        r = int(255 * normalized_timing)
        b = int(255 * (1 - normalized_timing))
        g = 0
        return f'rgb({r}, {g}, {b})'

    def _add_timing_legend(self, fig: go.Figure, min_timing: int, max_timing: int):
        """
        Add timing legend to figure.

        Args:
            fig: Plotly figure
            min_timing: Minimum timing value
            max_timing: Maximum timing value
        """
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=f'Early: {min_timing}ms',
            showlegend=True
        ))

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'Late: {max_timing}ms',
            showlegend=True
        ))


def create_3d_design_summary(
    pattern: BlastPattern3D
) -> Dict[str, Any]:
    """
    Create summary statistics for 3D blast pattern.

    Args:
        pattern: BlastPattern3D object

    Returns:
        Dictionary with summary statistics
    """
    holes = pattern.holes

    # Calculate statistics
    total_holes = len(holes)
    total_charge_mass = sum(hole.charge_mass_kg for hole in holes)
    avg_hole_length = np.mean([hole.hole_length for hole in holes])
    avg_burden = np.mean([hole.burden_m for hole in holes])
    avg_spacing = np.mean([hole.spacing_m for hole in holes])

    # Calculate pattern area
    x_coords = [hole.x for hole in holes]
    y_coords = [hole.y for hole in holes]
    pattern_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

    # Calculate powder factor
    bench_volume = pattern_area * pattern.bench_geometry.bench_height_m
    powder_factor = total_charge_mass / bench_volume if bench_volume > 0 else 0

    return {
        'pattern_id': pattern.pattern_id,
        'pattern_type': pattern.pattern_type.value,
        'total_holes': total_holes,
        'total_charge_mass_kg': total_charge_mass,
        'avg_hole_length_m': avg_hole_length,
        'avg_burden_m': avg_burden,
        'avg_spacing_m': avg_spacing,
        'pattern_area_m2': pattern_area,
        'bench_volume_m3': bench_volume,
        'powder_factor_kg_m3': powder_factor,
        'bench_height_m': pattern.bench_geometry.bench_height_m,
        'bench_width_m': pattern.bench_geometry.bench_width_m,
        'bench_length_m': pattern.bench_geometry.bench_length_m
    }
