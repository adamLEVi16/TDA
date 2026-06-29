"""
Publication-Quality Plotting Configuration
For Master's Thesis Expansion

This module sets up matplotlib and seaborn for professional,
publication-ready figures suitable for journal submission.

Usage:
    from plot_config import setup_plots, COLORS, save_figure

    setup_plots()  # Call once at start of notebook

    # Use COLORS for consistent coloring
    plt.plot(data, color=COLORS['blue'])

    # Save with proper settings
    save_figure('my_plot', formats=['pdf', 'png'])

Author: Adam Levine
Date: January 2026
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Professional color scheme (colorblind-safe)
COLORS = {
    'blue': '#0173B2',      # Primary data, trust
    'orange': '#DE8F05',    # Secondary data, attention
    'green': '#029E73',     # Positive, growth
    'red': '#DC267F',       # Negative, danger
    'purple': '#785EF0',    # Uncertainty, special cases
    'brown': '#CA9161',     # Neutral
    'gray': '#949494',      # Baseline, benchmarks
    'black': '#000000',     # Text, borders
}

def setup_plots(style='publication'):
    """
    Configure matplotlib for publication-quality figures

    Args:
        style (str): One of 'publication', 'presentation', 'poster'
            - publication: Journal paper settings (default)
            - presentation: Slides/Beamer (larger text)
            - poster: Conference poster (very large text)

    Returns:
        None
    """

    # Base settings for all styles
    base_config = {
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],

        'lines.linewidth': 2.0,
        'lines.markersize': 6,

        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,

        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,

        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.edgecolor': 'gray',

        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
    }

    # Style-specific settings
    if style == 'publication':
        style_config = {
            'figure.figsize': (12, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',

            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        }

    elif style == 'presentation':
        style_config = {
            'figure.figsize': (14, 7),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',

            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,

            'lines.linewidth': 3.0,
        }

    elif style == 'poster':
        style_config = {
            'figure.figsize': (16, 9),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',

            'font.size': 18,
            'axes.labelsize': 22,
            'axes.titlesize': 26,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,

            'lines.linewidth': 4.0,
        }

    else:
        raise ValueError(f"Unknown style: {style}. Use 'publication', 'presentation', or 'poster'")

    # Apply settings
    plt.rcParams.update(base_config)
    plt.rcParams.update(style_config)

    # Set colorblind-safe palette
    sns.set_palette("colorblind")

    print(f"✅ {style.capitalize()} plotting mode enabled")
    print(f"   Resolution: {plt.rcParams['savefig.dpi']} DPI")
    print(f"   Format: {plt.rcParams['savefig.format']}")
    print(f"   Font size: {plt.rcParams['font.size']}pt")
    print(f"   Figure size: {plt.rcParams['figure.figsize']}")


def save_figure(filename, fig=None, formats=['pdf', 'png'], output_dir='../figures'):
    """
    Save figure in multiple formats with proper settings

    Args:
        filename (str): Filename without extension
        fig (matplotlib.figure.Figure): Figure to save (default: current figure)
        formats (list): List of formats to save ['pdf', 'png', 'svg']
        output_dir (str): Directory to save figures

    Returns:
        list: Paths to saved files
    """

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use current figure if not specified
    if fig is None:
        fig = plt.gcf()

    saved_files = []

    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"

        # Format-specific settings
        save_kwargs = {
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
        }

        if fmt == 'pdf':
            save_kwargs['dpi'] = 300
        elif fmt == 'png':
            save_kwargs['dpi'] = 300
            save_kwargs['transparent'] = False
        elif fmt == 'svg':
            save_kwargs['transparent'] = True

        fig.savefig(filepath, **save_kwargs)
        saved_files.append(str(filepath))
        print(f"💾 Saved: {filepath}")

    return saved_files


def add_crisis_shading(ax, crisis_periods=None):
    """
    Add crisis period shading to plot

    Args:
        ax (matplotlib.axes.Axes): Axes to add shading to
        crisis_periods (list): List of tuples (start, end, label, color)
            Default uses standard periods from paper

    Returns:
        None
    """

    if crisis_periods is None:
        # Default crisis periods from original paper
        crisis_periods = [
            ('2020-02-01', '2020-04-01', 'COVID\nCrash', '#DC267F'),
            ('2022-01-01', '2022-06-01', 'Fed\nPivot', '#DE8F05'),
            ('2024-01-01', '2024-12-01', 'AI\nVolatility', '#785EF0'),
        ]

    import pandas as pd

    for start, end, label, color in crisis_periods:
        # Add shaded region
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   alpha=0.12, color=color, zorder=0)


def format_percentage_axis(ax, axis='y'):
    """
    Format axis to show percentages

    Args:
        ax (matplotlib.axes.Axes): Axes to format
        axis (str): Which axis to format ('x' or 'y')

    Returns:
        None
    """
    from matplotlib.ticker import FuncFormatter

    def to_percent(y, position):
        return f'{100 * y:.0f}%'

    formatter = FuncFormatter(to_percent)

    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    elif axis == 'x':
        ax.xaxis.set_major_formatter(formatter)


def add_panel_label(ax, label, position='top_left', fontsize=14):
    """
    Add panel label (A, B, C, etc.) to subplot

    Args:
        ax (matplotlib.axes.Axes): Axes to add label to
        label (str): Label text (e.g., 'A', 'B')
        position (str): One of 'top_left', 'top_right'
        fontsize (int): Font size for label

    Returns:
        None
    """

    if position == 'top_left':
        x, y = 0.02, 0.98
        ha, va = 'left', 'top'
    elif position == 'top_right':
        x, y = 0.98, 0.98
        ha, va = 'right', 'top'
    else:
        raise ValueError("position must be 'top_left' or 'top_right'")

    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor='white',
                     edgecolor='gray',
                     alpha=0.8))


# Auto-setup on import (publication mode by default)
setup_plots('publication')
