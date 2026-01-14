#!/usr/bin/env python3
"""Experiment 2 Statistical Analysis Script"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import math

import importlib.util

_HELPER_PATH = Path(__file__).parent / "visualization_helpers" / "experiment2_visualization_helpers.py"
_spec = importlib.util.spec_from_file_location("experiment2_visualization_helpers", _HELPER_PATH)
exp2_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp2_viz)


class Experiment2Analyzer:
    def __init__(self, data_dir: str, output_dir: str = "outputs"):
        self.data_dir = Path(os.path.expanduser(data_dir)).resolve()
        base_output_dir = Path(output_dir).resolve()
        self.output_dir = base_output_dir / "experiment2"
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "experiment2_plots"
        self.primary_plots_dir = self.plots_dir / "primary"
        self.secondary_plots_dir = self.plots_dir / "secondary"
        self.primary_plots_dir.mkdir(parents=True, exist_ok=True)
        self.secondary_plots_dir.mkdir(parents=True, exist_ok=True)

        self.locate_data = pd.DataFrame()
        self.pointing_data = pd.DataFrame()
        self.questionnaire_data = pd.DataFrame()
        self.locate_learning = pd.DataFrame()
        self.pointing_learning = pd.DataFrame()
        self.DA_questions = pd.DataFrame()

    def _remap_time_point(self, data: pd.DataFrame) -> pd.Series:
        if len(data) == 0 or 'time_point' not in data.columns:
            return data.get('time_point', pd.Series(dtype='float64'))
        time_point_map = {1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6}

        def remap_value(x):
            if pd.isna(x):
                return x
            try:
                x_int = int(x)
                return time_point_map.get(x_int, x_int)
            except (ValueError, TypeError):
                return x

        return data['time_point'].apply(remap_value)

    def load_data(self):
        locate_data_list = []
        pointing_data_list = []

        condition_folders = [d for d in self.data_dir.iterdir() if d.is_dir()]
        for condition_path in condition_folders:
            condition_name = condition_path.name
            is_visual_baseline = (condition_name == "Visual Baseline")

            if is_visual_baseline:
                locate_path = condition_path / "Locate"
                if locate_path.exists():
                    for file in locate_path.glob("*.csv"):
                        filename = file.name
                        pid_match = re.search(r'[LP][A-Z]*\d{3}', filename)
                        if not pid_match:
                            continue
                        participant_id = pid_match.group()
                        data = pd.read_csv(file)
                        data['participantID'] = participant_id
                        data['condition'] = condition_name
                        session_block_match = re.search(r'S(\d+)_B(\d+)', filename)
                        if session_block_match:
                            session = int(session_block_match.group(1))
                            block = int(session_block_match.group(2))
                            data['session'] = session
                            data['block'] = block
                            data['session_block'] = f"S{session}_B{block}"
                            data['time_point'] = (session - 1) * 3 + block
                        else:
                            data['session'] = 1
                            data['block'] = 1
                            data['session_block'] = 'S1_B1'
                            data['time_point'] = 1
                        data['filename'] = filename
                        locate_data_list.append(data)

                point_path = condition_path / "Point"
                if point_path.exists():
                    for file in point_path.glob("*.csv"):
                        filename = file.name
                        pid_match = re.search(r'[LP][A-Z]*\d{3}', filename)
                        if not pid_match:
                            continue
                        participant_id = pid_match.group()
                        data = pd.read_csv(file)
                        data['participantID'] = participant_id
                        data['condition'] = condition_name
                        session_block_match = re.search(r'S(\d+)_B(\d+)', filename)
                        if session_block_match:
                            session = int(session_block_match.group(1))
                            block = int(session_block_match.group(2))
                            data['session'] = session
                            data['block'] = block
                            data['session_block'] = f"S{session}_B{block}"
                            data['time_point'] = (session - 1) * 3 + block
                        else:
                            data['session'] = 1
                            data['block'] = 1
                            data['session_block'] = 'S1_B1'
                            data['time_point'] = 1
                        data['filename'] = filename
                        pointing_data_list.append(data)

            block_folders = [d for d in condition_path.iterdir()
                             if d.is_dir() and d.name not in ['Locate', 'Point']]
            for block_folder in block_folders:
                locate_path = block_folder / "Locate"
                if locate_path.exists():
                    for file in locate_path.glob("*.csv"):
                        filename = file.name
                        session_block_match = re.search(r'S(\d+)_B(\d+)', filename)
                        if not session_block_match:
                            continue
                        session = int(session_block_match.group(1))
                        block = int(session_block_match.group(2))
                        session_block = f"S{session}_B{block}"
                        pid_match = re.search(r'L[A-Z]*\d{3}', filename)
                        if not pid_match:
                            continue
                        participant_id = pid_match.group()
                        data = pd.read_csv(file)
                        data['participantID'] = participant_id
                        data['condition'] = condition_name
                        data['session'] = session
                        data['block'] = block
                        data['session_block'] = session_block
                        data['time_point'] = (session - 1) * 3 + block
                        data['filename'] = filename
                        locate_data_list.append(data)

                point_path = block_folder / "Point"
                if point_path.exists():
                    for file in point_path.glob("*.csv"):
                        filename = file.name
                        session_block_match = re.search(r'S(\d+)_B(\d+)', filename)
                        if not session_block_match:
                            continue
                        session = int(session_block_match.group(1))
                        block = int(session_block_match.group(2))
                        session_block = f"S{session}_B{block}"
                        pid_match = re.search(r'L[A-Z]*\d{3}', filename)
                        if not pid_match:
                            continue
                        participant_id = pid_match.group()
                        data = pd.read_csv(file)
                        data['participantID'] = participant_id
                        data['condition'] = condition_name
                        data['session'] = session
                        data['block'] = block
                        data['session_block'] = session_block
                        data['time_point'] = (session - 1) * 3 + block
                        data['filename'] = filename
                        pointing_data_list.append(data)

        if locate_data_list:
            self.locate_data = pd.concat(locate_data_list, ignore_index=True)
            self.locate_data['time_point'] = self._remap_time_point(self.locate_data)
        if pointing_data_list:
            self.pointing_data = pd.concat(pointing_data_list, ignore_index=True)
            self.pointing_data['time_point'] = self._remap_time_point(self.pointing_data)

        questionnaire_path = self.data_dir / "Longitudinal Post-Condition Questionnaire  (Responses) - Form Responses 1.csv"
        if questionnaire_path.exists():
            self.questionnaire_data = pd.read_csv(questionnaire_path)
            column_mapping = {
                'Timestamp': 'timestamp',
                'PID (filled out by experimenter)': 'participantID',
                'Configuration (filled out by experimenter)': 'condition',
                'Session (filled out by experimenter)': 'session',
                'Block (filled out by experimenter)': 'block',
                'Did vibrations feel like they were coming from the target or from your hands?': 'q1_target_vs_hands',
                'I experienced:': 'q2_experienced',
                'I experienced: ': 'q2_experienced',
                "Did you feel like you could perceive the target's position directly, or did you have to deliberately interpret it based on the vibrations": 'q3_direct_vs_interpret',
                'How much were you attending to your hands as the basis for your judgments?': 'q4_attending_hands',
                'When you moved the controllers, did the haptic sensations change in a predictable way?': 'q5_predictable_change',
                'Did you perceive each target as existing in a stable, stationary location?': 'q6_stable_location',
                'How intuitive did the haptic feedback feel?': 'q7_intuitive',
                'Did the target feel like it existed independently of your actions?': 'q8_independent_existence',
                'How accurate/reliable did the haptic system feel?': 'q9_accurate_reliable',
                'Any additional comments or feedback?': 'comments'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in self.questionnaire_data.columns:
                    self.questionnaire_data.rename(columns={old_col: new_col}, inplace=True)
            self.questionnaire_data['session_block'] = (
                'S' + self.questionnaire_data['session'].astype(str) +
                '_B' + self.questionnaire_data['block'].astype(str)
            )
            self.questionnaire_data['time_point'] = (
                (self.questionnaire_data['session'] - 1) * 3 +
                self.questionnaire_data['block']
            )
            self.questionnaire_data['time_point'] = self._remap_time_point(self.questionnaire_data)

    def filter_longest_test_block(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0 or 'filename' not in data.columns:
            return pd.DataFrame()
        filtered_list = []
        for filename, group in data.groupby('filename'):
            group = group.sort_values('timestamp')
            if 'trialType' in group.columns:
                group['is_test'] = group['trialType'].str.lower() == 'test'
            else:
                group['is_test'] = True
            group['run_id'] = (group['is_test'] != group['is_test'].shift()).cumsum()
            test_runs = group[group['is_test']].copy()
            if len(test_runs) == 0:
                continue
            run_sizes = test_runs.groupby('run_id').size()
            longest_run_id = run_sizes.idxmax()
            longest_run = test_runs[test_runs['run_id'] == longest_run_id].copy()
            filtered_list.append(longest_run)
        if filtered_list:
            return pd.concat(filtered_list, ignore_index=True)
        return pd.DataFrame()

    def analyze_locate_learning(self):
        if len(self.locate_data) == 0:
            return
        locate_test = self.filter_longest_test_block(self.locate_data)
        if len(locate_test) == 0:
            return
        if 'timestamp' in locate_test.columns:
            locate_test['timestamp'] = pd.to_datetime(locate_test['timestamp'], errors='coerce')
            locate_test = locate_test.dropna(subset=['timestamp'])

        locate_learning_list = []
        for (pid, condition, session, block, session_block, time_point), group in locate_test.groupby(
            ['participantID', 'condition', 'session', 'block', 'session_block', 'time_point']
        ):
            group = group.sort_values('timestamp')
            if len(group) == 0:
                hit_rate = 0.0
            else:
                time_window_minutes = 2 if condition == 'Visual Baseline' else 4
                first_trial_time = group['timestamp'].iloc[0]
                window_end = first_trial_time + pd.Timedelta(minutes=time_window_minutes)
                trials_in_window = group[group['timestamp'] <= window_end]
                total_trials = len(trials_in_window)
                hit_rate = total_trials / float(time_window_minutes)
            locate_learning_list.append({
                'participantID': pid,
                'condition': condition,
                'session': session,
                'block': block,
                'session_block': session_block,
                'time_point': time_point,
                'hit_rate': hit_rate
            })
        self.locate_learning = pd.DataFrame(locate_learning_list)

    def analyze_pointing_learning(self):
        if len(self.pointing_data) == 0:
            return
        pointing_test = self.filter_longest_test_block(self.pointing_data)
        if len(pointing_test) == 0:
            return
        pointing_test['testDuration'] = pd.to_numeric(pointing_test['testDuration'], errors='coerce')
        pointing_filtered = pointing_test[pointing_test['testDuration'] >= 1.0].copy()

        numeric_cols = ['selectionErrorEdge', 'angularDeviation', 'angularDeviationEdge',
                        'actualTargetX', 'actualTargetY', 'actualTargetZ',
                        'selectedTargetX', 'selectedTargetY', 'selectedTargetZ',
                        'rightControllerX', 'rightControllerY', 'rightControllerZ']
        for col in numeric_cols:
            if col in pointing_filtered.columns:
                pointing_filtered[col] = pd.to_numeric(pointing_filtered[col], errors='coerce')

        target_radius = 0.15
        if all(col in pointing_filtered.columns for col in ['rightControllerX', 'rightControllerY', 'rightControllerZ',
                                                            'actualTargetX', 'actualTargetY', 'actualTargetZ',
                                                            'selectedTargetX', 'selectedTargetY', 'selectedTargetZ']):
            pointing_filtered['dist_controller_to_actual'] = np.sqrt(
                (pointing_filtered['rightControllerX'] - pointing_filtered['actualTargetX'])**2 +
                (pointing_filtered['rightControllerY'] - pointing_filtered['actualTargetY'])**2 +
                (pointing_filtered['rightControllerZ'] - pointing_filtered['actualTargetZ'])**2
            )
            pointing_filtered['dist_controller_to_selected'] = np.sqrt(
                (pointing_filtered['rightControllerX'] - pointing_filtered['selectedTargetX'])**2 +
                (pointing_filtered['rightControllerY'] - pointing_filtered['selectedTargetY'])**2 +
                (pointing_filtered['rightControllerZ'] - pointing_filtered['selectedTargetZ'])**2
            )
            pointing_filtered['depth_error'] = np.abs(
                pointing_filtered['dist_controller_to_actual'] -
                pointing_filtered['dist_controller_to_selected']
            )
            pointing_filtered['depth_error'] = np.maximum(0, pointing_filtered['depth_error'] - target_radius)
        else:
            pointing_filtered['depth_error'] = np.nan

        pointing_learning_list = []
        for (pid, condition, session, block, session_block, time_point), group in pointing_filtered.groupby(
            ['participantID', 'condition', 'session', 'block', 'session_block', 'time_point']
        ):
            pointing_learning_list.append({
                'participantID': pid,
                'condition': condition,
                'session': session,
                'block': block,
                'session_block': session_block,
                'time_point': time_point,
                'mean_selection_error': group['selectionErrorEdge'].mean() if 'selectionErrorEdge' in group.columns else np.nan,
                'mean_angular_deviation': group['angularDeviationEdge'].mean() if 'angularDeviationEdge' in group.columns else np.nan,
                'mean_depth_error': group['depth_error'].mean() if 'depth_error' in group.columns else np.nan
            })
        self.pointing_learning = pd.DataFrame(pointing_learning_list)

    def process_da_data(self):
        if len(self.questionnaire_data) == 0:
            return
        da_cols = ['q1_target_vs_hands', 'q2_experienced', 'q3_direct_vs_interpret']
        available_da_cols = [col for col in da_cols if col in self.questionnaire_data.columns]
        if len(available_da_cols) == 0:
            return
        self.DA_questions = self.questionnaire_data[
            ['participantID', 'condition', 'session', 'block', 'session_block', 'time_point'] + available_da_cols
        ].copy()
        for col in available_da_cols:
            self.DA_questions[col] = pd.to_numeric(self.DA_questions[col], errors='coerce')
        self.DA_questions['composite_da_score'] = self.DA_questions[available_da_cols].mean(axis=1)

    def filter_complete_session_participants(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return pd.DataFrame()
        complete_participants = []
        for (pid, condition), group in data.groupby(['participantID', 'condition']):
            unique_timepoints = sorted([int(tp) for tp in group['time_point'].dropna().unique() if pd.notna(tp)])
            if len(unique_timepoints) >= 6:
                first_session = [tp for tp in unique_timepoints if tp <= 3]
                second_session = [tp for tp in unique_timepoints if tp > 3]
                if len(first_session) >= 3 and len(second_session) >= 3:
                    complete_participants.append((pid, condition))
        if len(complete_participants) == 0:
            return pd.DataFrame()
        complete_mask = pd.Series([
            (row['participantID'], row['condition']) in complete_participants
            for _, row in data.iterrows()
        ], index=data.index)
        return data[complete_mask].copy()

    def filter_both_sessions_participants(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) == 0:
            return pd.DataFrame()
        both_sessions_participants = []
        for (pid, condition), group in data.groupby(['participantID', 'condition']):
            unique_timepoints = sorted([int(tp) for tp in group['time_point'].dropna().unique() if pd.notna(tp)])
            if len(unique_timepoints) >= 5:
                first_session = [tp for tp in unique_timepoints if tp <= 3]
                second_session = [tp for tp in unique_timepoints if tp > 3]
                if len(first_session) >= 2 and len(second_session) >= 2:
                    both_sessions_participants.append((pid, condition))
        if len(both_sessions_participants) == 0:
            return pd.DataFrame()
        both_sessions_mask = pd.Series([
            (row['participantID'], row['condition']) in both_sessions_participants
            for _, row in data.iterrows()
        ], index=data.index)
        return data[both_sessions_mask].copy()

    def create_experiment2_primary_outcomes_latex_table(self):
        haptic_conditions = ["Discrete Graded", "Interpolated Graded"]
        table_data = []

        if len(self.locate_learning) > 0:
            locate_haptic = self.locate_learning[self.locate_learning['condition'].isin(haptic_conditions)].copy()
            locate_complete = self.filter_both_sessions_participants(locate_haptic)
            if len(locate_complete) > 0:
                for block in range(1, 7):
                    block_data = locate_complete[locate_complete['time_point'] == block]
                    for condition in haptic_conditions:
                        cond_data = block_data[block_data['condition'] == condition]
                        if len(cond_data) > 0:
                            table_data.append({
                                'metric': 'Locate Performance',
                                'condition': condition,
                                'block': block,
                                'mean': cond_data['hit_rate'].mean(),
                                'std': cond_data['hit_rate'].std()
                            })

        if len(self.pointing_learning) > 0:
            pointing_haptic = self.pointing_learning[self.pointing_learning['condition'].isin(haptic_conditions)].copy()
            pointing_complete = self.filter_both_sessions_participants(pointing_haptic)
            if len(pointing_complete) > 0:
                for block in range(1, 7):
                    block_data = pointing_complete[pointing_complete['time_point'] == block]
                    for condition in haptic_conditions:
                        cond_data = block_data[block_data['condition'] == condition]
                        if len(cond_data) > 0:
                            table_data.append({
                                'metric': 'Point Selection Error',
                                'condition': condition,
                                'block': block,
                                'mean': cond_data['mean_selection_error'].mean() * 100,
                                'std': cond_data['mean_selection_error'].std() * 100
                            })

        if len(self.DA_questions) > 0:
            da_haptic = self.DA_questions[self.DA_questions['condition'].isin(haptic_conditions)].copy()
            da_complete = self.filter_both_sessions_participants(da_haptic)
            if len(da_complete) > 0 and 'composite_da_score' in da_complete.columns:
                for block in range(1, 7):
                    block_data = da_complete[da_complete['time_point'] == block]
                    for condition in haptic_conditions:
                        cond_data = block_data[block_data['condition'] == condition]
                        if len(cond_data) > 0:
                            table_data.append({
                                'metric': 'DA',
                                'condition': condition,
                                'block': block,
                                'mean': cond_data['composite_da_score'].mean(),
                                'std': cond_data['composite_da_score'].std()
                            })

        if len(table_data) == 0:
            return

        table_df = pd.DataFrame(table_data)
        csv_output_file = self.output_dir / 'experiment2_primary_outcomes_table.csv'
        table_df.to_csv(csv_output_file, index=False)
        latex_lines = []
        latex_lines.append("% Note: This table requires \\usepackage{multirow} in your LaTeX preamble")
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Experiment 2: Metrics by Block}")
        latex_lines.append("\\label{tab:experiment2_metrics_by_block}")
        latex_lines.append("\\begin{tabular}{llcccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Metric & Condition & Block 1 & Block 2 & Block 3 & Block 4 & Block 5 & Block 6 \\\\")
        latex_lines.append("\\midrule")

        def format_mean_sd(mean, std, metric):
            if np.isnan(mean) or np.isnan(std):
                return "---"
            if metric == 'Point Selection Error':
                return f"{mean:.1f} $\\pm$ {std:.1f}"
            if metric == 'Locate Performance':
                return f"{mean:.2f} $\\pm$ {std:.2f}"
            return f"{mean:.2f} $\\pm$ {std:.2f}"

        metric_display_names = {
            'Locate Performance': 'Target Acquisition Rate (targets/min)',
            'Point Selection Error': 'Point Selection Error (cm)',
            'DA': 'Composite DA (1--7)'
        }

        for metric in ['Locate Performance', 'Point Selection Error', 'DA']:
            metric_data = table_df[table_df['metric'] == metric]
            if len(metric_data) == 0:
                continue
            metric_display = metric_display_names.get(metric, metric)
            condition_rows = []
            for condition in haptic_conditions:
                cond_data = metric_data[metric_data['condition'] == condition]
                if len(cond_data) == 0:
                    continue
                condition_label = "Discrete" if "Discrete" in condition else "Interpolated"
                block_values = []
                for block in range(1, 7):
                    block_row = cond_data[cond_data['block'] == block]
                    if len(block_row) > 0:
                        row = block_row.iloc[0]
                        block_values.append(format_mean_sd(row['mean'], row['std'], metric))
                    else:
                        block_values.append("---")
                condition_rows.append((condition_label, block_values))
            for row_idx, (condition_label, block_values) in enumerate(condition_rows):
                if row_idx == 0:
                    latex_lines.append(
                        f"\\multirow{{2}}{{*}}{{{metric_display}}} & {condition_label} & {' & '.join(block_values)} \\\\"
                    )
                else:
                    latex_lines.append(
                        f" & {condition_label} & {' & '.join(block_values)} \\\\"
                    )
            if metric != 'DA':
                latex_lines.append("\\addlinespace")

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        output_file = self.output_dir / 'experiment2_primary_outcomes_table.tex'
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex_lines))

    def _fit_mixed_model_terms(self, data: pd.DataFrame, metric_col: str) -> dict:
        if not HAS_STATSMODELS:
            return {'error': 'statsmodels not available'}
        if len(data) == 0 or metric_col not in data.columns:
            return {'error': 'no data'}

        model_data = data[['participantID', 'condition', 'time_point', metric_col]].dropna().copy()
        model_data = model_data[model_data['condition'].isin(['Discrete Graded', 'Interpolated Graded'])]
        if len(model_data) == 0:
            return {'error': 'no valid data'}

        model_data['condition'] = model_data['condition'].astype(str)
        model_data['cond_interpolated'] = (model_data['condition'] == 'Interpolated Graded').astype(int)
        timepoint_mean = model_data['time_point'].mean()
        model_data['timepoint_centered'] = model_data['time_point'] - timepoint_mean
        model_data['interaction'] = model_data['cond_interpolated'] * model_data['timepoint_centered']

        X = model_data[['cond_interpolated', 'timepoint_centered', 'interaction']].values
        X = sm.add_constant(X)
        y = model_data[metric_col].values
        groups = model_data['participantID'].values

        try:
            model = MixedLM(y, X, groups, exog_re=None)
            result = model.fit(reml=False)
        except Exception as e:
            return {'error': str(e)}

        coefs = np.array(result.params)
        pvals = np.array(result.pvalues)
        cov = result.cov_params()

        discrete_slope = coefs[2]
        discrete_p = pvals[2]
        interpolated_slope = coefs[2] + coefs[3]
        var_sum = cov[2, 2] + cov[3, 3] + 2 * cov[2, 3]
        if var_sum > 0:
            se_sum = math.sqrt(var_sum)
            z = interpolated_slope / se_sum
            p_interp = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        else:
            p_interp = np.nan

        return {
            'discrete_slope': discrete_slope,
            'discrete_p': discrete_p,
            'interpolated_slope': interpolated_slope,
            'interpolated_p': p_interp,
            'interaction_beta': coefs[3],
            'interaction_p': pvals[3],
            'n_obs': len(model_data),
            'n_participants': model_data['participantID'].nunique()
        }

    def create_experiment2_mixed_model_tables(self):
        if not HAS_STATSMODELS:
            return

        locate_complete = self.filter_complete_session_participants(self.locate_learning)
        pointing_complete = self.filter_complete_session_participants(self.pointing_learning)
        da_complete = self.filter_complete_session_participants(self.DA_questions)

        primary_specs = [
            ('Locate (targets/min)', locate_complete, 'hit_rate'),
            ('Point (selection error, cm)', pointing_complete, 'mean_selection_error'),
            ('Composite DA (1–7)', da_complete, 'composite_da_score')
        ]
        secondary_specs = [
            ('Angular Deviation (degrees)', pointing_complete, 'mean_angular_deviation'),
            ('Depth Error (cm)', pointing_complete, 'mean_depth_error')
        ]

        def build_rows(specs):
            rows = []
            for metric_label, data, metric_col in specs:
                if len(data) == 0 or metric_col not in data.columns:
                    continue
                model_data = data.copy()
                if metric_label in ['Point (selection error, cm)', 'Depth Error (cm)']:
                    model_data[metric_col] = model_data[metric_col] * 100
                results = self._fit_mixed_model_terms(model_data, metric_col)
                if 'error' in results:
                    continue
                rows.append({
                    'Metric': metric_label,
                    'Discrete Graded': results['discrete_slope'],
                    'Discrete p': results['discrete_p'],
                    'Interpolated Graded': results['interpolated_slope'],
                    'Interpolated p': results['interpolated_p'],
                    'Condition × Timepoint': results['interaction_beta'],
                    'Interaction p': results['interaction_p'],
                    'n_participants': results['n_participants'],
                    'n_obs': results['n_obs']
                })
            return rows

        primary_rows = build_rows(primary_specs)
        secondary_rows = build_rows(secondary_specs)

        def export_tables(rows, basename, caption):
            if not rows:
                return
            df = pd.DataFrame(rows)
            csv_path = self.output_dir / f'{basename}.csv'
            df.to_csv(csv_path, index=False)

            latex_lines = []
            latex_lines.append("\\begin{table}[h]")
            latex_lines.append("\\centering")
            latex_lines.append(f"\\caption{{{caption}}}")
            latex_lines.append("\\begin{tabular}{lccc}")
            latex_lines.append("\\toprule")
            latex_lines.append("Metric & Slope (Discrete) & Slope (Interpolated) & Condition $\\times$ Timepoint \\\\")
            latex_lines.append("\\midrule")

            for _, row in df.iterrows():
                def format_cell(beta, p_val):
                    if np.isnan(beta) or np.isnan(p_val):
                        return "---"
                    p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
                    return f"$\\beta$={beta:.3f}, p={p_str}"

                discrete_cell = format_cell(row['Discrete Graded'], row['Discrete p'])
                interpolated_cell = format_cell(row['Interpolated Graded'], row['Interpolated p'])
                interaction_cell = format_cell(row['Condition × Timepoint'], row['Interaction p'])
                latex_lines.append(
                    f"{row['Metric']} & {discrete_cell} & {interpolated_cell} & {interaction_cell} \\\\"
                )
            latex_lines.append("\\bottomrule")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\end{table}")

            tex_path = self.output_dir / f'{basename}.tex'
            with open(tex_path, 'w') as f:
                f.write('\n'.join(latex_lines))

        export_tables(
            primary_rows,
            'experiment2_mixed_model_primary_metrics',
            'Experiment 2: Mixed-Effects Models for Primary Metrics'
        )
        export_tables(
            secondary_rows,
            'experiment2_mixed_model_secondary_metrics',
            'Experiment 2: Mixed-Effects Models for Secondary Metrics'
        )

    def create_plots(self):
        locate_complete = self.filter_complete_session_participants(self.locate_learning)
        pointing_complete = self.filter_complete_session_participants(self.pointing_learning)
        da_complete = self.filter_complete_session_participants(self.DA_questions)

        if len(locate_complete) > 0:
            exp2_viz.plot_hit_rate(locate_complete, self.primary_plots_dir, show_individuals=False)
            exp2_viz.plot_normalized_hit_rate(locate_complete, self.secondary_plots_dir)
        if len(pointing_complete) > 0:
            exp2_viz.plot_selection_error(pointing_complete, self.primary_plots_dir, show_individuals=False)
            exp2_viz.plot_angular_deviation(pointing_complete, self.secondary_plots_dir, show_individuals=False)
            exp2_viz.plot_depth_error(pointing_complete, self.secondary_plots_dir, show_individuals=False)
            exp2_viz.plot_normalized_selection_error(pointing_complete, self.secondary_plots_dir)
            exp2_viz.plot_normalized_depth_error(pointing_complete, self.secondary_plots_dir)
        if len(da_complete) > 0:
            exp2_viz.plot_combined_da(da_complete, self.primary_plots_dir, show_individuals=False)

    def run_analysis(self):
        self.load_data()
        self.analyze_locate_learning()
        self.analyze_pointing_learning()
        self.process_da_data()
        self.create_plots()
        self.create_experiment2_primary_outcomes_latex_table()
        self.create_experiment2_mixed_model_tables()
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2 analysis")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Experiment 2 Data",
        help="Path to Experiment 2 data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for figures and tables"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    analyzer = Experiment2Analyzer(data_dir=str(data_dir), output_dir=str(output_dir))
    analyzer.run_analysis()

