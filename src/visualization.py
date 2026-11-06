"""
Visualization helpers for the Steam Games project.
Ensures charts comply with Section 3.4 rubric:
'At least two well-labelled plots (title, axes, units, legend).'
Course: CSC17104 - Programming for Data Science, HCMUS.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_theme():
    """
    Apply a consistent, clean, high-contrast academic aesthetic.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.autolayout'] = True


def annotate_bars(ax, fmt: str = '{:.1f}', padding: int = 3):
    """
    Add numeric data labels directly above bar chart patches.
    """
    for p in ax.patches:
        height = p.get_height()
        if height is not None and not (height != height):  # check not nan
            if abs(height) > 0.001:
                ax.annotate(
                    fmt.format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, padding),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='semibold'
                )

