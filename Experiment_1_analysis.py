#!/usr/bin/env python3
"""Experiment 1 Statistical Analysis Script"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

import pingouin as pg
from scipy.stats import shapiro, normaltest, levene

import importlib.util

_HELPER_PATH = Path(__file__).parent / "visualization_helpers" / "experiment1_visualization_helpers.py"
_spec = importlib.util.spec_from_file_location("experiment1_visualization_helpers", _HELPER_PATH)
exp1_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1_viz)

CONDITIONS = ["Discrete Constant", "Discrete Graded", "Interpolated Constant", "Interpolated Graded"]

class Experiment1Analyzer:
    def __init__(self, data_dir: str, output_dir: str = "outputs"):
        self.data_dir = Path(data_dir).resolve()
        base_output_dir = Path(output_dir).resolve()
        self.output_dir = base_output_dir / "experiment1"
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "experiment1_plots"
        self.primary_plots_dir = self.plots_dir / "primary"
        self.secondary_plots_dir = self.plots_dir / "secondary"
        self.primary_plots_dir.mkdir(parents=True, exist_ok=True)
        self.secondary_plots_dir.mkdir(parents=True, exist_ok=True)
        self.locate_data = pd.DataFrame()
        self.pointing_data = pd.DataFrame()
        self.questionnaire_data = pd.DataFrame()
        self.locate_summary = pd.DataFrame()
        self.pointing_summary = pd.DataFrame()
    
    def load_data(self):
        locate_data_list = []
        pointing_data_list = []
        
        for condition in CONDITIONS:
            interpolation = "Interpolated" if "Interpolated" in condition else "Discrete"
            grading = "Graded" if "Graded" in condition else "Constant"
            
            locate_path = self.data_dir / condition / "Locate"
            if locate_path.exists():
                for file in locate_path.glob("*.csv"):
                    try:
                        data = pd.read_csv(file)
                        data['condition'] = condition
                        data['interpolation'] = interpolation
                        data['grading'] = grading
                        data['filename'] = file.name
                        pid_match = re.search(r'P\d{3}', file.name)
                        if pid_match:
                            data['participantID'] = pid_match.group()
                        for col in ['testDuration', 'hitCount', 'timeSinceLastHit_sec']:
                            if col in data.columns:
                                data[col] = pd.to_numeric(data[col], errors='coerce')
                        locate_data_list.append(data)
                    except Exception:
                        pass
            
            point_path = self.data_dir / condition / "Point"
            if point_path.exists():
                for file in point_path.glob("*.csv"):
                    try:
                        data = pd.read_csv(file)
                        data['condition'] = condition
                        data['interpolation'] = interpolation
                        data['grading'] = grading
                        data['filename'] = file.name
                        pid_match = re.search(r'P\d{3}', file.name)
                        if pid_match:
                            data['participantID'] = pid_match.group()
                        for col in ['selectionError', 'selectionErrorEdge', 'angularDeviation', 
                                  'angularDeviationEdge', 'testDuration']:
                            if col in data.columns:
                                data[col] = pd.to_numeric(data[col], errors='coerce')
                        pointing_data_list.append(data)
                    except Exception:
                        pass
        
        if locate_data_list:
            self.locate_data = pd.concat(locate_data_list, ignore_index=True)
        if pointing_data_list:
            self.pointing_data = pd.concat(pointing_data_list, ignore_index=True)
        
        questionnaire_path = self.data_dir / "Post-Condition Questionnaire  (Responses) - Form Responses 1.csv"
        if questionnaire_path.exists():
            try:
                self.questionnaire_data = pd.read_csv(questionnaire_path)
            except Exception:
                pass
    
    def filter_longest_test_block(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0 or 'filename' not in data.columns:
            return pd.DataFrame()
        
        filtered_list = []
        for filename, group in data.groupby('filename'):
            if 'timestamp' in group.columns:
                group['timestamp'] = pd.to_datetime(group['timestamp'], errors='coerce')
                group = group.sort_values('timestamp')
            
            if 'trialType' in group.columns:
                group['is_test'] = group['trialType'].str.lower() == 'test'
            elif 'hitCount' in group.columns:
                group['is_test'] = pd.to_numeric(group['hitCount'], errors='coerce') > 0
            else:
                group['is_test'] = True
            
            group['run_id'] = (group['is_test'] != group['is_test'].shift()).cumsum()
            test_runs = group[group['is_test']].copy()
            
            if len(test_runs) == 0:
                continue
            
            run_sizes = test_runs.groupby('run_id').size()
            largest_run_id = run_sizes.idxmax()
            largest_block = test_runs[test_runs['run_id'] == largest_run_id].copy()
            
            if 'interpolation' in data.columns:
                largest_block['interpolation'] = group['interpolation'].iloc[0]
            if 'grading' in data.columns:
                largest_block['grading'] = group['grading'].iloc[0]
            if 'modality' in data.columns:
                largest_block['modality'] = group.get('modality', 'Haptic').iloc[0]
            
            filtered_list.append(largest_block)
        
        return pd.concat(filtered_list, ignore_index=True) if filtered_list else pd.DataFrame()
    
    def filter_data(self):
        self.locate_test = self.filter_longest_test_block(self.locate_data) if len(self.locate_data) > 0 else pd.DataFrame()
        self.pointing_test = self.filter_longest_test_block(self.pointing_data) if len(self.pointing_data) > 0 else pd.DataFrame()
    
    def analyze_locate_task(self):
        locate_summary_list = []
        
        if len(self.locate_test) > 0:
            if 'modality' not in self.locate_test.columns:
                self.locate_test['modality'] = 'Haptic'
            
            for (pid, filename, condition, interpolation, grading, modality), group in self.locate_test.groupby(
                ['participantID', 'filename', 'condition', 'interpolation', 'grading', 'modality']
            ):
                group = group.sort_values('timestamp')
                group['timestamp'] = pd.to_datetime(group['timestamp'], errors='coerce')
                group = group.dropna(subset=['timestamp'])
                
                if len(group) == 0:
                    locate_summary_list.append({
                        'participantID': pid, 'filename': filename, 'condition': condition,
                        'interpolation': interpolation, 'grading': grading, 'modality': modality,
                        'total_hits': 0, 'window_duration_min': 4.0, 'hit_rate': 0.0
                    })
                    continue
                
                first_hit_time = group['timestamp'].iloc[0]
                group['time_from_first_hit_sec'] = (group['timestamp'] - first_hit_time).dt.total_seconds()
                hits_in_window = group[group['time_from_first_hit_sec'] <= 240]
                
                total_hits = len(hits_in_window) if len(hits_in_window) > 0 else 0
                hit_rate = total_hits / 4.0
                
                locate_summary_list.append({
                    'participantID': pid, 'filename': filename, 'condition': condition,
                    'interpolation': interpolation, 'grading': grading, 'modality': modality,
                    'total_hits': total_hits, 'window_duration_min': 4.0, 'hit_rate': hit_rate
                })
        
        files_already_in_summary = {item['filename'] for item in locate_summary_list} if locate_summary_list else set()
        
        if len(self.locate_data) > 0:
            if 'modality' not in self.locate_data.columns:
                self.locate_data['modality'] = 'Haptic'
            all_loaded_files = self.locate_data.groupby(['filename', 'condition', 'interpolation', 'grading', 'modality', 'participantID']).first().reset_index()
            
            for _, row in all_loaded_files.iterrows():
                if row['filename'] not in files_already_in_summary:
                    locate_summary_list.append({
                        'participantID': row['participantID'], 'filename': row['filename'],
                        'condition': row['condition'], 'interpolation': row['interpolation'],
                        'grading': row['grading'], 'modality': row.get('modality', 'Haptic'),
                        'total_hits': 0, 'window_duration_min': 4.0, 'hit_rate': 0.0
                    })
        
        self.locate_summary = pd.DataFrame(locate_summary_list)
    
    def analyze_pointing_task(self):
        if len(self.pointing_test) == 0:
            return
        
        self.pointing_test = self.pointing_test[
            pd.to_numeric(self.pointing_test.get('testDuration', 0), errors='coerce') >= 1.0
        ]
        
        pointing_summary_list = []
        if 'modality' not in self.pointing_test.columns:
            self.pointing_test['modality'] = 'Haptic'
        
        for (pid, filename, condition, interpolation, grading, modality), group in self.pointing_test.groupby(
            ['participantID', 'filename', 'condition', 'interpolation', 'grading', 'modality']
        ):
            for col in ['actualTargetX', 'actualTargetY', 'actualTargetZ', 'selectedTargetX', 
                       'selectedTargetY', 'selectedTargetZ', 'rightControllerX', 'rightControllerY', 
                       'rightControllerZ', 'selectionErrorEdge', 'angularDeviationEdge']:
                if col in group.columns:
                    group[col] = pd.to_numeric(group[col], errors='coerce')
            
            mean_selection_error = group['selectionErrorEdge'].mean() if 'selectionErrorEdge' in group.columns else (
                pd.to_numeric(group['selectionError'], errors='coerce').mean() if 'selectionError' in group.columns else np.nan
            )
            
            mean_angular_dev = group['angularDeviationEdge'].mean() if 'angularDeviationEdge' in group.columns else (
                pd.to_numeric(group['angularDeviation'], errors='coerce').mean() if 'angularDeviation' in group.columns else np.nan
            )
            
            if all(col in group.columns for col in ['actualTargetX', 'actualTargetY', 'actualTargetZ',
                                                    'selectedTargetX', 'selectedTargetY', 'selectedTargetZ',
                                                    'rightControllerX', 'rightControllerY', 'rightControllerZ']):
                dist_controller_to_actual = np.sqrt(
                    (group['rightControllerX'] - group['actualTargetX'])**2 +
                    (group['rightControllerY'] - group['actualTargetY'])**2 +
                    (group['rightControllerZ'] - group['actualTargetZ'])**2
                )
                dist_controller_to_selected = np.sqrt(
                    (group['rightControllerX'] - group['selectedTargetX'])**2 +
                    (group['rightControllerY'] - group['selectedTargetY'])**2 +
                    (group['rightControllerZ'] - group['selectedTargetZ'])**2
                )
                mean_depth_error = np.abs(dist_controller_to_actual - dist_controller_to_selected).mean()
            else:
                mean_depth_error = np.nan
            
            pointing_summary_list.append({
                'participantID': pid, 'filename': filename, 'condition': condition,
                'interpolation': interpolation, 'grading': grading, 'modality': modality,
                'mean_selection_error_edge': mean_selection_error,
                'mean_angular_deviation_edge': mean_angular_dev,
                'mean_depth_error_edge': mean_depth_error
            })
        
        self.pointing_summary = pd.DataFrame(pointing_summary_list)
        if 'mean_selection_error_edge' in self.pointing_summary.columns:
            self.pointing_summary['selection_error'] = self.pointing_summary['mean_selection_error_edge']
        if 'mean_angular_deviation_edge' in self.pointing_summary.columns:
            self.pointing_summary['angular_deviation'] = self.pointing_summary['mean_angular_deviation_edge']
        if 'mean_depth_error_edge' in self.pointing_summary.columns:
            self.pointing_summary['depth_error'] = self.pointing_summary['mean_depth_error_edge'] * 100
    
    def process_questionnaire_data(self):
        if len(self.questionnaire_data) == 0:
            return pd.DataFrame()
        
        column_mapping = {
            'PID (filled out by experimenter)': 'participantID',
            'Configuration (filled out by experimenter)': 'condition',
            'Did vibrations feel like they were coming from the target or from your hands?': 'q1_vibration_location',
            'I experienced: ': 'q2_experience_quality',
            'I experienced:': 'q2_experience_quality',
            'Did you feel like you could perceive the target\'s position directly, or did you have to deliberately interpret it based on the vibrations': 'q3_direct_vs_interpret'
        }
        
        q_data = self.questionnaire_data.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in q_data.columns:
                q_data.rename(columns={old_col: new_col}, inplace=True)
        
        for condition in CONDITIONS:
            mask = q_data['condition'].str.contains(condition.split()[0], case=False, na=False) & \
                   q_data['condition'].str.contains(condition.split()[1], case=False, na=False)
            q_data.loc[mask, 'condition'] = condition
            q_data.loc[mask, 'interpolation'] = "Interpolated" if "Interpolated" in condition else "Discrete"
            q_data.loc[mask, 'grading'] = "Graded" if "Graded" in condition else "Constant"
        
        da_cols = ['q1_vibration_location', 'q2_experience_quality', 'q3_direct_vs_interpret']
        for col in da_cols:
            if col in q_data.columns:
                q_data[col] = pd.to_numeric(q_data[col], errors='coerce')
        
        if all(col in q_data.columns for col in da_cols):
            q_data['composite_da_score'] = q_data[da_cols].mean(axis=1)
        
        for col in da_cols + ['composite_da_score']:
            if col in q_data.columns:
                q_data = q_data[(q_data[col] >= 1) & (q_data[col] <= 7)]
        
        return q_data
    
    def check_anova_assumptions(self, data, dv, between_factor, within_factor, subject='participantID'):
        results = {'normality': {}, 'homogeneity': {}, 'warnings': []}
        
        try:
            anova_data = data[[subject, between_factor, within_factor, dv]].dropna()
            if len(anova_data) < 3:
                return results
            
            data_check = anova_data.groupby(subject).agg({
                between_factor: 'first',
                within_factor: lambda x: x.nunique()
            }).reset_index()
            data_check.columns = [subject, 'grading_group', 'n_interpolations']
            complete_subjects = data_check[data_check['n_interpolations'] == 2][subject].unique()
            data_complete = anova_data[anova_data[subject].isin(complete_subjects)].copy()
            
            if len(complete_subjects) < 3:
                return results
            
            data_complete['group_mean'] = data_complete.groupby([between_factor, within_factor])[dv].transform('mean')
            residuals = data_complete[dv] - data_complete['group_mean']
            residuals = residuals.dropna()
            
            if len(residuals) >= 3 and len(residuals) <= 5000:
                stat_sw, p_sw = shapiro(residuals)
                results['normality']['shapiro_wilk'] = {
                    'statistic': stat_sw, 'p_value': p_sw, 'normal': p_sw > 0.05
                }
            
            if len(residuals) >= 8:
                stat_dp, p_dp = normaltest(residuals)
                results['normality']['dagostino_pearson'] = {
                    'statistic': stat_dp, 'p_value': p_dp, 'normal': p_dp > 0.05
                }
            
            groups = []
            for grading in data_complete[between_factor].unique():
                group_data = data_complete[data_complete[between_factor] == grading][dv].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                stat_levene, p_levene = levene(*groups)
                results['homogeneity']['levene'] = {
                    'statistic': stat_levene, 'p_value': p_levene, 'homogeneous': p_levene > 0.05
                }
        except Exception:
            pass
        
        return results
    
    def run_mixed_anova(self, data, dv, between_factor, within_factor, subject='participantID'):
        anova_data = data[[subject, between_factor, within_factor, dv]].dropna()
        assumptions = self.check_anova_assumptions(data, dv, between_factor, within_factor, subject)
        aov = pg.mixed_anova(data=anova_data, dv=dv, between=between_factor, within=within_factor, subject=subject)
        return {'anova_table': aov, 'data': anova_data, 'assumptions': assumptions}
    
    def export_statistical_results(self, locate_anova, pointing_anova, pointing_angular_anova, pointing_depth_anova, da_anova):
        results_list = []
        assumption_results = []
        
        for anova, task_name, metric, is_primary in [
            (locate_anova, 'Locate Task', 'Target Acquisition Rate (targets/min)', True),
            (pointing_anova, 'Point Task', 'Selection Error (cm)', True),
            (da_anova, 'Distal Attribution', 'Composite DA Score', True),
            (pointing_angular_anova, 'Point Task', 'Angular Deviation (degrees)', False),
            (pointing_depth_anova, 'Point Task', 'Depth Error (cm)', False)
        ]:
            if anova and 'anova_table' in anova:
                for _, row in anova['anova_table'].iterrows():
                    results_list.append({
                        'Task': task_name, 'Metric': metric, 'Effect': row['Source'],
                        'F': row.get('F', ''), 'p': row.get('p-unc', ''),
                        'eta_squared_partial': row.get('np2', ''), 'epsilon': row.get('eps', ''),
                        'Primary': is_primary
                    })
                if 'assumptions' in anova:
                    assumption_results.append({
                        'Task': task_name, 'Metric': metric, 'Primary': is_primary, **anova['assumptions']
                    })
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.output_dir / 'experiment1_statistical_results.csv', index=False)
            if hasattr(exp1_viz, "create_results_table_png"):
                exp1_viz.create_results_table_png(results_df, self.output_dir)
            elif hasattr(exp1_viz, "create_statistical_table_png"):
                exp1_viz.create_statistical_table_png(results_df, self.output_dir)
            if assumption_results:
                exp1_viz.create_assumptions_table_png(assumption_results, self.output_dir)
            return results_df
        return pd.DataFrame()

    def create_experiment1_primary_outcomes_table(self, da_data: pd.DataFrame):
        table_data = []
        condition_order = [
            "Discrete Constant", "Interpolated Constant",
            "Discrete Graded", "Interpolated Graded"
        ]

        if len(self.locate_summary) > 0:
            for condition in condition_order:
                cond_data = self.locate_summary[self.locate_summary['condition'] == condition]
                if len(cond_data) > 0:
                    table_data.append({
                        'metric': 'Target Acquisition Rate (targets/min)',
                        'condition': condition,
                        'mean': cond_data['hit_rate'].mean(),
                        'std': cond_data['hit_rate'].std(),
                        'n': len(cond_data)
                    })

        if len(self.pointing_summary) > 0 and 'selection_error' in self.pointing_summary.columns:
            for condition in condition_order:
                cond_data = self.pointing_summary[self.pointing_summary['condition'] == condition]
                if len(cond_data) > 0:
                    table_data.append({
                        'metric': 'Point Selection Error (cm)',
                        'condition': condition,
                        'mean': cond_data['selection_error'].mean() * 100,
                        'std': cond_data['selection_error'].std() * 100,
                        'n': len(cond_data)
                    })

        if len(da_data) > 0 and 'composite_da_score' in da_data.columns:
            for condition in condition_order:
                cond_data = da_data[da_data['condition'] == condition]
                if len(cond_data) > 0:
                    table_data.append({
                        'metric': 'Composite DA (1--7)',
                        'condition': condition,
                        'mean': cond_data['composite_da_score'].mean(),
                        'std': cond_data['composite_da_score'].std(),
                        'n': len(cond_data)
                    })

        if len(table_data) == 0:
            return

        table_df = pd.DataFrame(table_data)
        csv_output_file = self.output_dir / 'experiment1_primary_outcomes_table.csv'
        table_df.to_csv(csv_output_file, index=False)

        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Experiment 1: Primary Outcomes by Condition}")
        latex_lines.append("\\label{tab:experiment1_primary_outcomes}")
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Outcome & \\multicolumn{2}{c}{Constant} & \\multicolumn{2}{c}{Graded} \\\\")
        latex_lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
        latex_lines.append(" & Discrete & Interpolated & Discrete & Interpolated \\\\")
        latex_lines.append("\\midrule")

        def format_mean_sd(mean, std, metric):
            if np.isnan(mean) or np.isnan(std):
                return "---"
            if metric == 'Point Selection Error (cm)':
                return f"{mean:.1f} $\\pm$ {std:.1f}"
            if metric == 'Target Acquisition Rate (targets/min)':
                return f"{mean:.2f} $\\pm$ {std:.2f}"
            return f"{mean:.2f} $\\pm$ {std:.2f}"

        metrics_in_order = [
            'Target Acquisition Rate (targets/min)',
            'Point Selection Error (cm)',
            'Composite DA (1--7)'
        ]
        condition_map = {
            "Discrete Constant": ("Constant", "Discrete"),
            "Interpolated Constant": ("Constant", "Interpolated"),
            "Discrete Graded": ("Graded", "Discrete"),
            "Interpolated Graded": ("Graded", "Interpolated")
        }

        for metric in metrics_in_order:
            row_values = {
                "Discrete Constant": "---",
                "Interpolated Constant": "---",
                "Discrete Graded": "---",
                "Interpolated Graded": "---"
            }
            metric_data = table_df[table_df['metric'] == metric]
            for _, row in metric_data.iterrows():
                row_values[row['condition']] = format_mean_sd(row['mean'], row['std'], metric)

            if metric.startswith("Target Acquisition"):
                outcome_label = "Locate (targets/min)"
            elif metric.startswith("Point Selection"):
                outcome_label = "Point (selection error, cm)"
            else:
                outcome_label = "Composite DA (1--7)"

            latex_lines.append(
                f"{outcome_label} & {row_values['Discrete Constant']} & {row_values['Interpolated Constant']} & "
                f"{row_values['Discrete Graded']} & {row_values['Interpolated Graded']} \\\\"
            )

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        output_file = self.output_dir / 'experiment1_primary_outcomes_table.tex'
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex_lines))
    
    def run_analysis(self):
        self.load_data()
        self.filter_data()
        self.analyze_locate_task()
        self.analyze_pointing_task()
        da_data = self.process_questionnaire_data()
        
        locate_anova = self.run_mixed_anova(self.locate_summary, 'hit_rate', 'grading', 'interpolation') if len(self.locate_summary) > 0 else None
        pointing_anova = self.run_mixed_anova(self.pointing_summary, 'selection_error', 'grading', 'interpolation') if len(self.pointing_summary) > 0 else None
        pointing_angular_anova = None
        pointing_depth_anova = None
        if len(self.pointing_summary) > 0:
            if 'angular_deviation' in self.pointing_summary.columns:
                pointing_angular_anova = self.run_mixed_anova(self.pointing_summary, 'angular_deviation', 'grading', 'interpolation')
            if 'depth_error' in self.pointing_summary.columns:
                pointing_depth_anova = self.run_mixed_anova(self.pointing_summary, 'depth_error', 'grading', 'interpolation')
        da_anova = None
        if len(da_data) > 0 and 'composite_da_score' in da_data.columns:
            da_data['participantID'] = da_data.get('participantID', da_data.get('PID (filled out by experimenter)', ''))
            da_anova = self.run_mixed_anova(da_data, 'composite_da_score', 'grading', 'interpolation')
        
        self.export_statistical_results(locate_anova, pointing_anova, pointing_angular_anova, pointing_depth_anova, da_anova)
        self.create_experiment1_primary_outcomes_table(da_data)
        
        if len(self.locate_summary) > 0:
            exp1_viz.create_plot(self.locate_summary, 'hit_rate', 'Target Acquisition Rate',
                       'Locate Task Performance\nTarget Acquisition Rate', 'target_acquisition_rate',
                       self.primary_plots_dir, locate_anova)
        
        if len(self.pointing_summary) > 0:
            pointing_plot_data = self.pointing_summary.copy()
            pointing_plot_data['selection_error_cm'] = pointing_plot_data['selection_error'] * 100
            exp1_viz.create_plot(pointing_plot_data, 'selection_error_cm', 'Selection Error (cm)',
                       'Point Task Performance', 'selection_error',
                       self.primary_plots_dir, pointing_anova)

            if 'angular_deviation' in pointing_plot_data.columns:
                exp1_viz.create_plot(pointing_plot_data, 'angular_deviation', 'Angular Deviation (degrees)',
                           'Point Task Performance\nAngular Deviation', 'angular_deviation',
                           self.secondary_plots_dir, pointing_angular_anova)

            if 'depth_error' in pointing_plot_data.columns:
                exp1_viz.create_plot(pointing_plot_data, 'depth_error', 'Depth Error (cm)',
                           'Point Task Performance\nDepth Error', 'depth_error',
                           self.secondary_plots_dir, pointing_depth_anova)
        
        if len(da_data) > 0 and 'composite_da_score' in da_data.columns:
            exp1_viz.create_plot(da_data, 'composite_da_score', 'Composite DA Score',
                       'Composite DA Score by Condition', 'composite_da',
                       self.primary_plots_dir, da_anova)

        print(f"Results saved to: {self.output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1 Analysis')
    parser.add_argument('--data-dir', type=str, default='Experiment 1 Data', help='Path to Experiment 1 data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory for results')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    
    if not data_dir.exists():
        print(f"ERROR: Data directory does not exist: {data_dir}")
        return
    
    analyzer = Experiment1Analyzer(str(data_dir), str(output_dir))
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
