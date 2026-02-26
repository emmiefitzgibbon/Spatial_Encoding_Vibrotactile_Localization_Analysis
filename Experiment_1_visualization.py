#!/usr/bin/env python3
"""Visualization functions for Experiment 1 results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

CONDITIONS = ["Discrete Constant", "Discrete Graded", "Interpolated Constant", "Interpolated Graded"]
CONDITION_COLORS = {
    "Discrete Constant": "#4A5C7A", "Interpolated Constant": "#28A745",
    "Discrete Graded": "#4A5C7A", "Interpolated Graded": "#28A745"
}
CONDITION_FILLED = {
    "Discrete Constant": False, "Interpolated Constant": False,
    "Discrete Graded": True, "Interpolated Graded": True
}

def create_plot(data, y_column, ylabel, title, filename, plots_dir, anova_results=None):
    plot_data = data.copy().dropna(subset=[y_column])
    summary_stats = plot_data.groupby('condition').agg({y_column: ['mean', 'std', 'count']}).reset_index()
    summary_stats.columns = ['condition', 'mean', 'std', 'n']
    summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['n'])
    summary_stats['ci_95'] = summary_stats['se'] * 1.96
    
    condition_x_map = {
        "Discrete Constant": 0.0, "Interpolated Constant": 0.15,
        "Discrete Graded": 0.4, "Interpolated Graded": 0.55
    }
    bar_colors = {c: CONDITION_COLORS.get(c, '#000000') for c in CONDITIONS}
    edge_colors = {c: CONDITION_COLORS.get(c, '#000000') for c in CONDITIONS}
    
    fig, ax = plt.subplots(figsize=(8, 7))
    jitter_map = {}
    participant_positions = {}
    np.random.seed(42)
    
    y_values = plot_data[y_column].dropna()
    y_range = y_values.max() - y_values.min() if len(y_values) > 0 else 1
    y_jitter_scale = y_range * 0.01
    
    for condition in summary_stats['condition']:
        if condition not in condition_x_map:
            continue
        x_pos = condition_x_map[condition]
        cond_data = plot_data[plot_data['condition'] == condition]
        x_jitter = np.random.normal(0, 0.003, len(cond_data))
        y_jitter = np.random.normal(0, y_jitter_scale, len(cond_data))
        
        for idx, (_, row_data) in enumerate(cond_data.iterrows()):
            pid = row_data['participantID']
            x_jit = x_pos + x_jitter[idx]
            y_val = row_data[y_column] + y_jitter[idx]
            jitter_map[(pid, condition)] = (x_jitter[idx], y_jitter[idx])
            if pid not in participant_positions:
                participant_positions[pid] = {}
            participant_positions[pid][condition] = (x_jit, y_val)
    
    for pid, positions in participant_positions.items():
        if 'Discrete Constant' in positions and 'Interpolated Constant' in positions:
            x1, y1 = positions['Discrete Constant']
            x2, y2 = positions['Interpolated Constant']
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.2, linewidth=0.8, zorder=1)
        if 'Discrete Graded' in positions and 'Interpolated Graded' in positions:
            x1, y1 = positions['Discrete Graded']
            x2, y2 = positions['Interpolated Graded']
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.2, linewidth=0.8, zorder=1)
    
    for condition in summary_stats['condition']:
        if condition not in condition_x_map:
            continue
        x_pos = condition_x_map[condition]
        cond_data = plot_data[plot_data['condition'] == condition]
        color = bar_colors.get(condition, '#000000')
        is_interpolated = 'Interpolated' in condition
        edge_color = edge_colors.get(condition, color)
        
        for idx, (_, row_data) in enumerate(cond_data.iterrows()):
            pid = row_data['participantID']
            jitter_tuple = jitter_map.get((pid, condition), (0, 0))
            x_jitter_val, y_jitter_val = jitter_tuple if isinstance(jitter_tuple, tuple) else (jitter_tuple, 0)
            y_val = row_data[y_column] + y_jitter_val
            
            if is_interpolated:
                ax.scatter(x_pos + x_jitter_val, y_val, s=40, zorder=3,
                          facecolors='white', edgecolors=edge_color, linewidths=1.0, alpha=0.8)
            else:
                import matplotlib.colors as mcolors
                face_rgba = mcolors.to_rgba(color, alpha=0.6)
                edge_rgba = mcolors.to_rgba(edge_color, alpha=1.0)
                ax.scatter(x_pos + x_jitter_val, y_val, s=40, zorder=3,
                          facecolors=face_rgba, edgecolors=edge_rgba, linewidths=0.5)
    
    for condition in summary_stats['condition']:
        if condition not in condition_x_map:
            continue
        x_pos = condition_x_map[condition]
        row = summary_stats[summary_stats['condition'] == condition].iloc[0]
        color = bar_colors.get(condition, '#000000')
        edge_color = edge_colors.get(condition, color)
        
        ax.plot([x_pos - 0.075, x_pos + 0.075], [row['mean'], row['mean']],
               color=color, linewidth=2.5, zorder=4)
        
        box_width = 0.15
        box_left = x_pos - 0.075
        ci_bottom = row['mean'] - row['ci_95']
        ci_top = row['mean'] + row['ci_95']
        ci_height = ci_top - ci_bottom
        is_filled = CONDITION_FILLED.get(condition, True)
        
        if is_filled:
            import matplotlib.colors as mcolors
            face_rgba = mcolors.to_rgba(color, alpha=0.2)
            edge_rgba = mcolors.to_rgba(color, alpha=1.0)
            rect = Rectangle((box_left, ci_bottom), box_width, ci_height,
                          facecolor=face_rgba, edgecolor=edge_rgba, linewidth=1.0, zorder=2)
            ax.add_patch(rect)
        else:
            rect = Rectangle((box_left, ci_bottom), box_width, ci_height,
                          facecolor='none', edgecolor=edge_color, linewidth=1.0, zorder=2)
            ax.add_patch(rect)
    
    ax.set_xticks([0.075, 0.475])
    ax.set_xticklabels(['Constant', 'Graded'], fontsize=14, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='medium')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.axhline(y=0, color='black', linewidth=1.0, zorder=2)
    
    discrete_color = CONDITION_COLORS.get("Discrete Graded", '#4A5C7A')
    interpolated_color = CONDITION_COLORS.get("Interpolated Graded", '#28A745')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Discrete',
               markerfacecolor=discrete_color, markeredgecolor=discrete_color,
               markersize=8, markeredgewidth=0.5, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Interpolated',
               markerfacecolor='white', markeredgecolor=interpolated_color,
               markersize=8, markeredgewidth=1.0, alpha=0.8)
    ]
    legend_bbox_anchor = (1.0, 1.08) if title == 'Composite DA Score by Condition' else (1.0, 1.02)
    legend = ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=14, 
                      bbox_to_anchor=legend_bbox_anchor)
    
    title_pad = 40 if title == 'Composite DA Score by Condition' else 20
    if '\n' in title:
        main_title, subtitle = title.split('\n', 1)
        ax.set_title(main_title, fontsize=18, fontweight='bold', pad=title_pad + 25)
        fig = ax.figure
        subtitle_y = 1.0 + ((title_pad + 25) / 72) / fig.get_figheight() - 0.01
        ax.text(0.5, subtitle_y, subtitle, transform=ax.transAxes, ha='center', va='top',
               fontsize=16, fontweight='normal')
    else:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=title_pad)
    
    if anova_results and 'anova_table' in anova_results:
        anova_table = anova_results['anova_table']
        interpolation_row = anova_table[anova_table['Source'].str.contains('interpolation', case=False, na=False)]
        grading_row = anova_table[anova_table['Source'].str.contains('grading', case=False, na=False)]
        
        interpolation_p = interpolation_row['p-unc'].values[0] if len(interpolation_row) > 0 else None
        grading_p = grading_row['p-unc'].values[0] if len(grading_row) > 0 else None
        
        if interpolation_p is not None and interpolation_p < 0.1:
            legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
            bracket_x = (legend_bbox.x0 + legend_bbox.x1) / 2 - (legend_bbox.x1 - legend_bbox.x0) * 0.35
            bar_height = (legend_bbox.y1 - legend_bbox.y0) * 0.9
            adjusted_center = (legend_bbox.y0 + legend_bbox.y1) / 2 + (legend_bbox.y1 - legend_bbox.y0) * 0.02
            bar_top = adjusted_center + bar_height / 2
            bar_bottom = adjusted_center - bar_height * 0.35
            
            if interpolation_p < 0.05:
                ax.plot([bracket_x, bracket_x], [bar_bottom, bar_top], 'k', linestyle='-', linewidth=1.5, 
                       transform=ax.transAxes, zorder=10, clip_on=False)
                ax.plot([bracket_x, bracket_x + 0.01], [bar_bottom, bar_bottom], 'k', linestyle='-', 
                       linewidth=1.5, transform=ax.transAxes, zorder=10, clip_on=False)
                ax.plot([bracket_x, bracket_x + 0.01], [bar_top, bar_top], 'k', linestyle='-', 
                       linewidth=1.5, transform=ax.transAxes, zorder=10, clip_on=False)
                p_display = f"p = {interpolation_p:.3e} ***" if interpolation_p < 0.001 else \
                           (f"p = {interpolation_p:.3f} **" if interpolation_p < 0.01 else f"p = {interpolation_p:.3f} *")
                ax.text(bracket_x - 0.015, adjusted_center + (legend_bbox.y1 - legend_bbox.y0) * 0.05, p_display,
                       ha='right', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes, zorder=10)
        
        if grading_p is not None and grading_p < 0.1:
            current_ylim = ax.get_ylim()
            y_range = current_ylim[1] - current_ylim[0]
            space_needed = y_range * 0.15
            ax.set_ylim(bottom=-space_needed, top=current_ylim[1])
            ax.spines['bottom'].set_position(('data', 0))
            yticks = [t for t in ax.get_yticks() if t > 0]
            ax.set_yticks(yticks)
            ax.spines['left'].set_bounds(0, ax.get_ylim()[1])
            
            x_constant, x_graded = 0.075, 0.475
            bracket_y = -space_needed * 0.75
            
            if grading_p < 0.05:
                ax.plot([x_constant, x_graded], [bracket_y, bracket_y], 'k', linestyle='-', linewidth=1.5, zorder=10)
                ax.plot([x_constant, x_constant], [bracket_y, bracket_y + y_range * 0.02], 'k', linestyle='-', linewidth=1.5, zorder=10)
                ax.plot([x_graded, x_graded], [bracket_y, bracket_y + y_range * 0.02], 'k', linestyle='-', linewidth=1.5, zorder=10)
                p_display = f"p = {grading_p:.3e} ***" if grading_p < 0.001 else \
                           (f"p = {grading_p:.3f} **" if grading_p < 0.01 else f"p = {grading_p:.3f} *")
                ax.text((x_constant + x_graded) / 2, bracket_y - y_range * 0.008, p_display,
                       ha='center', va='top', fontsize=11, fontweight='bold', zorder=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_results_table_png(results_df, output_dir):
    def get_sort_key(row):
        task = row['Task']
        metric = row['Metric']
        if task == 'Locate Task':
            main_order = 0
        elif task == 'Point Task' and 'Selection Error' in metric:
            main_order = 1
        elif task == 'Distal Attribution':
            main_order = 2
        elif task == 'Point Task' and 'Angular Deviation' in metric:
            main_order = 3
        elif task == 'Point Task' and 'Depth Error' in metric:
            main_order = 4
        else:
            main_order = 99
        effect_order = {'Interaction': 0, 'grading': 1, 'interpolation': 2}.get(row['Effect'], 3)
        return (main_order, effect_order)
    
    results_df['sort_key'] = results_df.apply(get_sort_key, axis=1)
    results_df = results_df.sort_values('sort_key')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        metric_display = row['Metric']
        if not row.get('Primary', True):
            metric_display = f"{metric_display} (secondary)"
        p_str = "< 0.001" if pd.notna(row['p']) and row['p'] < 0.001 else (f"{row['p']:.4f}" if pd.notna(row['p']) else "")
        f_str = f"{row['F']:.2f}" if pd.notna(row['F']) else ""
        eta2_str = f"{row['eta_squared_partial']:.3f}" if pd.notna(row['eta_squared_partial']) else ""
        table_data.append([metric_display, row['Effect'], f_str, p_str, eta2_str])
    
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Effect', 'F', 'p', 'η²p'],
                    cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    col_widths = [0.4, 0.15, 0.1, 0.15, 0.1]
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
        row = results_df.iloc[i]
        is_primary = row.get('Primary', True)
        group_num = i // 3
        row_color = 'white' if group_num % 2 == 0 else '#D6E3F0'
        
        for j in range(5):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_color)
            if not is_primary:
                cell.set_text_props(style='italic')
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
    
    plt.title('Experiment 1: Statistical Results (2×2 Mixed ANOVA)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'experiment1_statistical_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_assumptions_table_png(assumption_results, output_dir):
    def get_sort_key(result):
        task = result.get('Task', '')
        metric = result.get('Metric', '')
        if task == 'Locate Task':
            return (0, 0)
        elif task == 'Point Task' and 'Selection Error' in metric:
            return (1, 0)
        elif task == 'Distal Attribution':
            return (2, 0)
        elif task == 'Point Task' and 'Angular Deviation' in metric:
            return (3, 0)
        elif task == 'Point Task' and 'Depth Error' in metric:
            return (4, 0)
        else:
            return (99, 0)
    
    assumption_results = sorted(assumption_results, key=get_sort_key)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    
    table_data = []
    for result in assumption_results:
        metric_display = result.get('Metric', '')
        if not result.get('Primary', True):
            metric_display = f"{metric_display} (secondary)"
        
        norm_sw = result.get('normality', {}).get('shapiro_wilk', {})
        sw_str = f"W={norm_sw.get('statistic', 0):.3f}, p={norm_sw.get('p_value', 0):.4f}" if norm_sw else "N/A"
        sw_status = "✓" if (norm_sw and norm_sw.get('normal', False)) else ("✗" if norm_sw else "")
        
        norm_dp = result.get('normality', {}).get('dagostino_pearson', {})
        dp_str = f"χ²={norm_dp.get('statistic', 0):.3f}, p={norm_dp.get('p_value', 0):.4f}" if norm_dp else "N/A"
        dp_status = "✓" if (norm_dp and norm_dp.get('normal', False)) else ("✗" if norm_dp else "")
        
        homog = result.get('homogeneity', {}).get('levene', {})
        lev_str = f"W={homog.get('statistic', 0):.3f}, p={homog.get('p_value', 0):.4f}" if homog else "N/A"
        lev_status = "✓" if (homog and homog.get('homogeneous', False)) else ("✗" if homog else "")
        
        norm_passed = (not norm_sw or norm_sw.get('normal', True)) and (not norm_dp or norm_dp.get('normal', True))
        homog_passed = not homog or homog.get('homogeneous', True)
        all_passed = norm_passed and homog_passed
        
        if all_passed:
            status_str = "Passed"
        else:
            if not norm_passed and not homog_passed:
                status_str = "Violations\n(Mann-Whitney U,\nWilcoxon)"
            elif not norm_passed:
                status_str = "Violations\n(Wilcoxon)"
            elif not homog_passed:
                status_str = "Violations\n(Mann-Whitney U)"
            else:
                status_str = "Violations"
        
        table_data.append([metric_display, f"{sw_str} {sw_status}", 
                         f"{dp_str} {dp_status}", f"{lev_str} {lev_status}", status_str])
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Metric', 'Normality (SW)', 'Normality (DP)', 'Homogeneity (L)', 'Status'],
                    cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    col_widths = [0.3, 0.18, 0.18, 0.18, 0.16]
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
        for j in range(5):
            cell = table[(i + 1, j)]
            row_color = 'white' if i % 2 == 0 else '#D6E3F0'
            cell.set_facecolor(row_color)
            if not assumption_results[i].get('Primary', True):
                cell.set_text_props(style='italic')
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            if j == 4:
                if "Passed" in table_data[i][j]:
                    cell.set_facecolor('#C8E6C9')
                else:
                    cell.set_facecolor('#FFCDD2')
    
    plt.title('Experiment 1: ANOVA Assumption Checks', fontsize=12, fontweight='bold', pad=10)
    plt.savefig(output_dir / 'experiment1_assumption_checks.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

