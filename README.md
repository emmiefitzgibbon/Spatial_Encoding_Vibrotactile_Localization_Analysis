## Running Analysis
`Experiment_1_analysis.py` and `Experiment_2_analysis.py` run the analyses for each experiment.

# Run from the repo root

Mac/Linux:
`python3 -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt && python Experiment_1_analysis.py && python Experiment_2_analysis.py`

Windows (PowerShell):
`python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt; python Experiment_1_analysis.py; python Experiment_2_analysis.py`

## Experiment 1
- Experiment 1 compared distance based intensity grading and directional interpolation in a 2 x 2 mixed between and within subjects design. 
- Data is separated into separate folders for each condition, which each contain subfolders for each task.
- Trajectory folders were excluded from this dataset to reduce file size.

**File Naming Convention:**
- Locate task: `locate_P###_[condition]_MainStudy.csv`
- Pointing task: `pointing_accuracy_P###_[condition]_MainStudy.csv`
- Where `P###` is the participant ID (e.g., P001, P002)

Locate task CSVs:
- A new row was logged with every target acquired.
- Some rows are from setup and practice rounds, denoted by the trialType column, which is either 'setup', 'practice', or 'test'.
Experiment 1 Analysis:
- Rows are filtered for trialType == 'test'
- Only the longest consecutive block of 'test' rows is analyzed, to make sure any trials erroneously marked as 'test' are excluded.
- Target acquisition rate is calculated as the number of rows present within four minutes of the first test row.

---

## Experiment 2
- Experiment 2 compared Discrete Graded and Interpolated Graded in between subjects design over two sessions
- Participants completed 3 blocks per session (6 blocks total).
- Only participants who completed both sessions are included in this dataset.
- Data is separated into folders by Condition, then block number (1-6), then Locate and Point tasks
- Trajectory folders were excluded from this dataset to reduce file size.
- The questionnaire only includes responses from participants who completed both sessions

**File Naming Convention:**
- Locate task: `locate_LP###_s[1-2]b[1-3]_[condition]_LongitudinalStudy_S[1-2]_B[1-3].csv`
- Pointing task: `pointing_accuracy_LP###_s[1-2]b[1-3]_[condition]_LongitudinalStudy_S[1-2]_B[1-3].csv`
- Where `LP###` is the participant ID (e.g., LP001, LP002) and `S[1-2]_B[1-3]` indicates session and block

