#!/usr/bin/env python3
"""Experiment 2 Visualization Script"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONDITION_COLORS = {
    "Discrete Graded": "#4A5C7A",
    "Interpolated Graded": "#28A745",
    "Visual Baseline": "#808080"
}


def _add_session_markers(ax, unique_timepoints):
    if len(unique_timepoints) >= 4:
        session_boundary = 3.5
        ax.axvline(x=session_boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        ax.text(0.25, 0.05, 'Session 1', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='black', transform=ax.transAxes)
        ax.text(0.75, 0.05, 'Session 2', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='black', transform=ax.transAxes)


def _legend_elements(conditions):
    from matplotlib.lines import Line2D
    legend_elements = []
    for condition in sorted(conditions):
        if condition == 'Interpolated Graded':
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='white',
                                          markeredgecolor=CONDITION_COLORS.get(condition, 'gray'),
                                          markeredgewidth=2, markersize=8,
                                          linestyle='None', label=condition))
        else:
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=CONDITION_COLORS.get(condition, 'gray'),
                                          markeredgecolor=CONDITION_COLORS.get(condition, 'gray'),
                                          markersize=8, linestyle='None', label=condition))
    return legend_elements


def plot_hit_rate(data, plots_dir, show_individuals=False):
    summary = data.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'hit_rate': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_hit_rate', 'sd_hit_rate', 'n_participants']
    summary['se'] = summary['sd_hit_rate'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_hit_rate'] + summary['se']
    summary['lower'] = summary['mean_hit_rate'] - summary['se']

    unique_timepoints = sorted(data['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')

        if show_individuals:
            cond_individual = data[data['condition'] == condition].copy()
            cond_individual['seq_time'] = cond_individual['time_point'].map(timepoint_map)
            for pid in cond_individual['participantID'].unique():
                pid_data = cond_individual[cond_individual['participantID'] == pid].sort_values('seq_time')
                if len(pid_data) > 1:
                    ax.plot(pid_data['seq_time'], pid_data['hit_rate'],
                            color=CONDITION_COLORS.get(condition, 'gray'),
                            alpha=0.25, linewidth=0.7, zorder=0,
                            solid_capstyle='round', solid_joinstyle='round')
                if condition == 'Interpolated Graded':
                    ax.scatter(pid_data['seq_time'], pid_data['hit_rate'],
                               facecolors='white', edgecolors=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, linewidths=1, zorder=0)
                else:
                    ax.scatter(pid_data['seq_time'], pid_data['hit_rate'],
                               color=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, zorder=0, edgecolors='none')

        import matplotlib.colors as mcolors
        condition_color = CONDITION_COLORS.get(condition, 'gray')
        band_color = mcolors.to_rgba(condition_color, alpha=0.2)
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        color=band_color, label='_nolegend_', zorder=1)

        line_color = mcolors.to_rgba(condition_color, alpha=1.0)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_hit_rate'].values,
                    linewidth=2.5, color=line_color, zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_hit_rate'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=line_color,
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_hit_rate'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=line_color, label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('Target Acquisition Rate (targets/min)', fontsize=16)
    ax.set_title('Locate Task Performance\nTarget Acquisition Rate', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if show_individuals else "_no_individuals"
    plt.savefig(plots_dir / f"target_acquisition_rate_over_time{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_selection_error(data, plots_dir, show_individuals=False):
    summary = data.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'mean_selection_error': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_error', 'sd_error', 'n_participants']
    summary['mean_error'] = summary['mean_error'] * 100
    summary['sd_error'] = summary['sd_error'] * 100
    summary['se'] = summary['sd_error'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_error'] + summary['se']
    summary['lower'] = summary['mean_error'] - summary['se']

    unique_timepoints = sorted(data['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')

        if show_individuals:
            cond_individual = data[data['condition'] == condition].copy()
            cond_individual['seq_time'] = cond_individual['time_point'].map(timepoint_map)
            for pid in cond_individual['participantID'].unique():
                pid_data = cond_individual[cond_individual['participantID'] == pid].sort_values('seq_time')
                if len(pid_data) > 1:
                    ax.plot(pid_data['seq_time'], pid_data['mean_selection_error'] * 100,
                            color=CONDITION_COLORS.get(condition, 'gray'),
                            alpha=0.25, linewidth=0.7, zorder=0,
                            solid_capstyle='round', solid_joinstyle='round')
                if condition == 'Interpolated Graded':
                    ax.scatter(pid_data['seq_time'], pid_data['mean_selection_error'] * 100,
                               facecolors='white', edgecolors=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, linewidths=1, zorder=0)
                else:
                    ax.scatter(pid_data['seq_time'], pid_data['mean_selection_error'] * 100,
                               color=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, zorder=0, edgecolors='none')

        import matplotlib.colors as mcolors
        condition_color = CONDITION_COLORS.get(condition, 'gray')
        band_color = mcolors.to_rgba(condition_color, alpha=0.2)
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        color=band_color, label='_nolegend_', zorder=1)

        line_color = mcolors.to_rgba(condition_color, alpha=1.0)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_error'].values,
                    linewidth=2.5, color=line_color, zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_error'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=line_color,
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_error'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=line_color, label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('Selection Error (cm)', fontsize=16)
    ax.set_title('Point Task Performance\nSelection Error', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if show_individuals else "_no_individuals"
    plt.savefig(plots_dir / f"selection_error_over_time{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_da(data, plots_dir, show_individuals=False):
    summary = data.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'composite_da_score': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_da', 'sd_da', 'n_participants']
    summary['se'] = summary['sd_da'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_da'] + summary['se']
    summary['lower'] = summary['mean_da'] - summary['se']

    unique_timepoints = sorted(data['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')

        if show_individuals:
            cond_individual = data[data['condition'] == condition].copy()
            cond_individual['seq_time'] = cond_individual['time_point'].map(timepoint_map)
            for pid in cond_individual['participantID'].unique():
                pid_data = cond_individual[cond_individual['participantID'] == pid].sort_values('seq_time')
                if len(pid_data) > 1:
                    ax.plot(pid_data['seq_time'], pid_data['composite_da_score'],
                            color=CONDITION_COLORS.get(condition, 'gray'),
                            alpha=0.25, linewidth=0.7, zorder=0,
                            solid_capstyle='round', solid_joinstyle='round')
                if condition == 'Interpolated Graded':
                    ax.scatter(pid_data['seq_time'], pid_data['composite_da_score'],
                               facecolors='white', edgecolors=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, linewidths=1, zorder=0)
                else:
                    ax.scatter(pid_data['seq_time'], pid_data['composite_da_score'],
                               color=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, zorder=0, edgecolors='none')

        import matplotlib.colors as mcolors
        condition_color = CONDITION_COLORS.get(condition, 'gray')
        band_color = mcolors.to_rgba(condition_color, alpha=0.2)
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        color=band_color, label='_nolegend_', zorder=1)

        line_color = mcolors.to_rgba(condition_color, alpha=1.0)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_da'].values,
                    linewidth=2.5, color=line_color, zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_da'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=line_color,
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_da'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=line_color, label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('Composite DA (1-7)', fontsize=16)
    ax.set_title('Composite DA', fontsize=18, fontweight='bold')
    ax.set_ylim(1, 7)
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if show_individuals else "_no_individuals"
    plt.savefig(plots_dir / f"combined_da_over_time{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_angular_deviation(data, plots_dir, show_individuals=False):
    summary = data.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'mean_angular_deviation': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_angular', 'sd_angular', 'n_participants']
    summary['se'] = summary['sd_angular'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_angular'] + summary['se']
    summary['lower'] = summary['mean_angular'] - summary['se']

    unique_timepoints = sorted(data['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')

        if show_individuals:
            cond_individual = data[data['condition'] == condition].copy()
            cond_individual['seq_time'] = cond_individual['time_point'].map(timepoint_map)
            for pid in cond_individual['participantID'].unique():
                pid_data = cond_individual[cond_individual['participantID'] == pid].sort_values('seq_time')
                if len(pid_data) > 1:
                    ax.plot(pid_data['seq_time'], pid_data['mean_angular_deviation'],
                            color=CONDITION_COLORS.get(condition, 'gray'),
                            alpha=0.25, linewidth=0.7, zorder=0,
                            solid_capstyle='round', solid_joinstyle='round')
                if condition == 'Interpolated Graded':
                    ax.scatter(pid_data['seq_time'], pid_data['mean_angular_deviation'],
                               facecolors='white', edgecolors=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, linewidths=1, zorder=0)
                else:
                    ax.scatter(pid_data['seq_time'], pid_data['mean_angular_deviation'],
                               color=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, zorder=0, edgecolors='none')

        import matplotlib.colors as mcolors
        condition_color = CONDITION_COLORS.get(condition, 'gray')
        band_color = mcolors.to_rgba(condition_color, alpha=0.2)
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        color=band_color, label='_nolegend_', zorder=1)

        line_color = mcolors.to_rgba(condition_color, alpha=1.0)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_angular'].values,
                    linewidth=2.5, color=line_color, zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_angular'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=line_color,
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_angular'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=line_color, label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('Angular Deviation (degrees)', fontsize=16)
    ax.set_title('Point Task Performance\nAngular Deviation', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if show_individuals else "_no_individuals"
    plt.savefig(plots_dir / f"angular_deviation_over_time{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_depth_error(data, plots_dir, show_individuals=False):
    summary = data.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'mean_depth_error': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_depth', 'sd_depth', 'n_participants']
    summary['mean_depth'] = summary['mean_depth'] * 100
    summary['sd_depth'] = summary['sd_depth'] * 100
    summary['se'] = summary['sd_depth'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_depth'] + summary['se']
    summary['lower'] = summary['mean_depth'] - summary['se']

    unique_timepoints = sorted(data['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')

        if show_individuals:
            cond_individual = data[data['condition'] == condition].copy()
            cond_individual['seq_time'] = cond_individual['time_point'].map(timepoint_map)
            for pid in cond_individual['participantID'].unique():
                pid_data = cond_individual[cond_individual['participantID'] == pid].sort_values('seq_time')
                if len(pid_data) > 1:
                    ax.plot(pid_data['seq_time'], pid_data['mean_depth_error'] * 100,
                            color=CONDITION_COLORS.get(condition, 'gray'),
                            alpha=0.25, linewidth=0.7, zorder=0,
                            solid_capstyle='round', solid_joinstyle='round')
                if condition == 'Interpolated Graded':
                    ax.scatter(pid_data['seq_time'], pid_data['mean_depth_error'] * 100,
                               facecolors='white', edgecolors=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, linewidths=1, zorder=0)
                else:
                    ax.scatter(pid_data['seq_time'], pid_data['mean_depth_error'] * 100,
                               color=CONDITION_COLORS.get(condition, 'gray'),
                               alpha=0.4, s=20, zorder=0, edgecolors='none')

        import matplotlib.colors as mcolors
        condition_color = CONDITION_COLORS.get(condition, 'gray')
        band_color = mcolors.to_rgba(condition_color, alpha=0.2)
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        color=band_color, label='_nolegend_', zorder=1)

        line_color = mcolors.to_rgba(condition_color, alpha=1.0)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_depth'].values,
                    linewidth=2.5, color=line_color, zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_depth'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=line_color,
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_depth'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=line_color, label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('Depth Error (cm)', fontsize=16)
    ax.set_title('Point Task Performance\nDepth Error', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if show_individuals else "_no_individuals"
    plt.savefig(plots_dir / f"depth_error_over_time{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalized_hit_rate(data, plots_dir):
    first_session_data = data[data['session_block'].str.startswith('S1_B1', na=False)].copy()
    condition_baseline = first_session_data.groupby('condition')['hit_rate'].mean().reset_index()
    condition_baseline.columns = ['condition', 'baseline_hit_rate']
    data_norm = data.merge(condition_baseline, on='condition', how='left')
    data_norm['normalized_hit_rate'] = np.where(
        data_norm['baseline_hit_rate'] > 0.01,
        ((data_norm['hit_rate'] - data_norm['baseline_hit_rate']) / data_norm['baseline_hit_rate']) * 100,
        np.nan
    )
    data_norm = data_norm.dropna(subset=['normalized_hit_rate'])
    if len(data_norm) == 0:
        return

    summary = data_norm.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'normalized_hit_rate': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_norm', 'sd_norm', 'n_participants']
    summary['se'] = summary['sd_norm'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_norm'] + summary['se']
    summary['lower'] = summary['mean_norm'] - summary['se']

    unique_timepoints = sorted(data_norm['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        band_alpha = 0.25 if condition == 'Interpolated Graded' else 0.2
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        alpha=band_alpha, color=CONDITION_COLORS.get(condition, 'gray'),
                        label='_nolegend_', zorder=1)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    linewidth=2.5, color=CONDITION_COLORS.get(condition, 'gray'),
                    zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=CONDITION_COLORS.get(condition, 'gray'),
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=CONDITION_COLORS.get(condition, 'gray'),
                    label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    block_6_timepoint = max(unique_timepoints)
    block_6_seq = timepoint_map[block_6_timepoint]
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        block_6_data = cond_data[cond_data['seq_time'] == block_6_seq]
        if len(block_6_data) > 0:
            improvement = block_6_data.iloc[0]['mean_norm']
            y_pos = block_6_data.iloc[0]['mean_norm']
            ax.text(block_6_seq, y_pos + 3, f'{improvement:.0f}%',
                    ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color=CONDITION_COLORS.get(condition, 'gray'))

    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('% improvement from first block performance', fontsize=16)
    ax.set_title('Target Acquisition Rate Improvement', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "target_acquisition_rate_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalized_selection_error(data, plots_dir):
    first_session_data = data[data['session_block'].str.startswith('S1_B1', na=False)].copy()
    condition_baseline = first_session_data.groupby('condition')['mean_selection_error'].mean().reset_index()
    condition_baseline.columns = ['condition', 'baseline_error']
    data_norm = data.merge(condition_baseline, on='condition', how='left')
    data_norm['normalized_error'] = np.where(
        data_norm['baseline_error'] > 0.001,
        -((data_norm['mean_selection_error'] - data_norm['baseline_error']) / data_norm['baseline_error']) * 100,
        np.nan
    )
    data_norm = data_norm.dropna(subset=['normalized_error'])
    if len(data_norm) == 0:
        return

    summary = data_norm.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'normalized_error': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_norm', 'sd_norm', 'n_participants']
    summary['se'] = summary['sd_norm'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_norm'] + summary['se']
    summary['lower'] = summary['mean_norm'] - summary['se']

    unique_timepoints = sorted(data_norm['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        band_alpha = 0.25 if condition == 'Interpolated Graded' else 0.2
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        alpha=band_alpha, color=CONDITION_COLORS.get(condition, 'gray'),
                        label='_nolegend_', zorder=1)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    linewidth=2.5, color=CONDITION_COLORS.get(condition, 'gray'),
                    zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=CONDITION_COLORS.get(condition, 'gray'),
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=CONDITION_COLORS.get(condition, 'gray'),
                    label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    block_6_timepoint = max(unique_timepoints)
    block_6_seq = timepoint_map[block_6_timepoint]
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        block_6_data = cond_data[cond_data['seq_time'] == block_6_seq]
        if len(block_6_data) > 0:
            improvement = block_6_data.iloc[0]['mean_norm']
            y_pos = block_6_data.iloc[0]['mean_norm']
            ax.text(block_6_seq, y_pos + 3, f'{improvement:.0f}%',
                    ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color=CONDITION_COLORS.get(condition, 'gray'))

    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('% improvement from first block performance', fontsize=16)
    ax.set_title('Selection Error Improvement', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "selection_error_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalized_depth_error(data, plots_dir):
    first_session_data = data[data['session_block'].str.startswith('S1_B1', na=False)].copy()
    condition_baseline = first_session_data.groupby('condition')['mean_depth_error'].mean().reset_index()
    condition_baseline.columns = ['condition', 'baseline_depth']
    data_norm = data.merge(condition_baseline, on='condition', how='left')
    data_norm['normalized_depth'] = np.where(
        data_norm['baseline_depth'] > 0.001,
        -((data_norm['mean_depth_error'] - data_norm['baseline_depth']) / data_norm['baseline_depth']) * 100,
        np.nan
    )
    data_norm = data_norm.dropna(subset=['normalized_depth'])
    if len(data_norm) == 0:
        return

    summary = data_norm.groupby(['condition', 'time_point', 'session_block', 'session']).agg({
        'normalized_depth': ['mean', 'std', 'count']
    }).reset_index()
    summary.columns = ['condition', 'time_point', 'session_block', 'session', 'mean_norm', 'sd_norm', 'n_participants']
    summary['se'] = summary['sd_norm'] / np.sqrt(summary['n_participants'])
    summary['upper'] = summary['mean_norm'] + summary['se']
    summary['lower'] = summary['mean_norm'] - summary['se']

    unique_timepoints = sorted(data_norm['time_point'].unique())
    timepoint_map = {tp: idx + 1 for idx, tp in enumerate(unique_timepoints)}
    summary['seq_time'] = summary['time_point'].map(timepoint_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        band_alpha = 0.25 if condition == 'Interpolated Graded' else 0.2
        ax.fill_between(cond_data['seq_time'], cond_data['lower'], cond_data['upper'],
                        alpha=band_alpha, color=CONDITION_COLORS.get(condition, 'gray'),
                        label='_nolegend_', zorder=1)
        if condition == 'Interpolated Graded':
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    linewidth=2.5, color=CONDITION_COLORS.get(condition, 'gray'),
                    zorder=4, label='_nolegend_')
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='white', markeredgecolor=CONDITION_COLORS.get(condition, 'gray'),
                    markeredgewidth=2.2, zorder=5, label=condition)
        else:
            ax.plot(cond_data['seq_time'].values, cond_data['mean_norm'].values,
                    marker='o', linewidth=2.5, markersize=9,
                    color=CONDITION_COLORS.get(condition, 'gray'),
                    label=condition, zorder=4)

    _add_session_markers(ax, unique_timepoints)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    block_6_timepoint = max(unique_timepoints)
    block_6_seq = timepoint_map[block_6_timepoint]
    for condition in sorted(summary['condition'].unique()):
        cond_data = summary[summary['condition'] == condition].sort_values('seq_time')
        block_6_data = cond_data[cond_data['seq_time'] == block_6_seq]
        if len(block_6_data) > 0:
            improvement = block_6_data.iloc[0]['mean_norm']
            y_pos = block_6_data.iloc[0]['mean_norm']
            ax.text(block_6_seq, y_pos + 3, f'{improvement:.0f}%',
                    ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color=CONDITION_COLORS.get(condition, 'gray'))

    ax.set_xticks(range(1, len(unique_timepoints) + 1))
    ax.set_xticklabels(range(1, len(unique_timepoints) + 1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Block', fontsize=16)
    ax.set_ylabel('% improvement from first block performance', fontsize=16)
    ax.set_title('Depth Error Improvement', fontsize=18, fontweight='bold')
    ax.legend(handles=_legend_elements(summary['condition'].unique()), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "depth_error_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_mixed_model_table_png(results_df, output_dir, title, filename):
    if results_df is None or len(results_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(19, 6.5))
    ax.axis('off')

    table_data = []
    for _, row in results_df.iterrows():
        def format_cell(beta, p_val):
            if pd.isna(beta) or pd.isna(p_val):
                return "---"
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            return f"$\\beta$={beta:.3f}, p={p_str}"

        n_participants = row.get('n_participants', np.nan)
        n_obs = row.get('n_obs', np.nan)
        if pd.isna(n_participants) or pd.isna(n_obs):
            n_display = ""
        else:
            n_display = f"{int(n_participants)} ({int(n_obs)})"

        table_data.append([
            row.get('Metric', ''),
            format_cell(row.get('Discrete Graded', np.nan), row.get('Discrete p', np.nan)),
            format_cell(row.get('Interpolated Graded', np.nan), row.get('Interpolated p', np.nan)),
            format_cell(row.get('Condition × Timepoint', np.nan), row.get('Interaction p', np.nan)),
            n_display
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Slope (Discrete)', 'Slope (Interpolated)', 'Condition × Timepoint', 'n (participants, obs)'],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)

    col_widths = [0.25, 0.215, 0.215, 0.22, 0.135]
    for i, width in enumerate(col_widths):
        for j in range(len(table_data) + 1):
            cell = table[(j, i)]
            cell.set_width(width)

    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#D6E3F0')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

    for i in range(len(table_data)):
        row_color = 'white' if i % 2 == 0 else '#D6E3F0'
        for j in range(5):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)

    plt.title(title, fontsize=14, fontweight='bold', pad=18)
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

