## Experiment 1: Main Study
- Experiment 1 compared distance based intensity grading and directional interpolation in a 2 x 2 mixed between and within subjects design. 
- Data is separated into separate folders for each condition, which each contain subfolders for each task.

### Data Structure (Experiment 1)
```
Experiment 1 Data/
├── Discrete Constant/
│   ├── Locate/          # Locate task data files (CSV)
│   └── Point/            # Pointing task data files (CSV)
├── Discrete Graded/
│   ├── Locate/
│   └── Point/
├── Interpolated Constant/
│   ├── Locate/
│   └── Point/
├── Interpolated Graded/
│   ├── Locate/
│   └── Point/
├── Visual Baseline/
│   └── Locate/
├── Post-Condition Questionnaire (Responses) - Form Responses 1.csv
└── Pre Experiment Questionnaire (Responses) - Form Responses 1.csv
```

**Note:** Trajectory folders were excluded from this dataset to reduce file size.

**File Naming Convention:**
- Locate task: `locate_P###_[condition]_MainStudy.csv`
- Pointing task: `pointing_accuracy_P###_[condition]_MainStudy.csv`
- Where `P###` is the participant ID (e.g., P001, P002)

---

## Experiment 2: Longitudinal Study
- Experiment 2 examined learning and retention across two sessions separated by time.
- Participants completed 3 blocks per session (6 blocks total).
- Only participants who completed both sessions (at least 2 blocks from session 1 and 2 blocks from session 2, with at least 5 total blocks) are included in this dataset.

### Data Structure (Experiment 2)
```
Experiment 2 Data/
├── Discrete Graded/
│   ├── 1/                # Block 1 (Session 1, Block 1)
│   │   ├── Locate/       # Locate task data files
│   │   └── Point/         # Pointing task data files
│   ├── 2/                # Block 2 (Session 1, Block 2)
│   ├── 3/                # Block 3 (Session 1, Block 3)
│   ├── 4/                # Block 4 (Session 2, Block 1)
│   ├── 5/                # Block 5 (Session 2, Block 2)
│   └── 6/                # Block 6 (Session 2, Block 3)
├── Interpolated Graded/
│   └── [Same block structure as Discrete Graded]
├── Visual Baseline/
│   └── Locate/
├── Longitudinal Post-Condition Questionnaire (Responses) - Form Responses 1.csv
└── Pre Experiment Questionnaire (Responses) - Form Responses 1.csv
```

**Note:** 
- Trajectory folders were excluded from this dataset to reduce file size.
- The questionnaire has been filtered to only include responses from participants who completed both sessions (132 rows from 22 participants, down from 160 original rows).

**File Naming Convention:**
- Locate task: `locate_LP###_s[1-2]b[1-3]_[condition]_LongitudinalStudy_S[1-2]_B[1-3].csv`
- Pointing task: `pointing_accuracy_LP###_s[1-2]b[1-3]_[condition]_LongitudinalStudy_S[1-2]_B[1-3].csv`
- Where `LP###` is the participant ID (e.g., LP001, LP002) and `S[1-2]_B[1-3]` indicates session and block

**Block to Session Mapping:**
- Blocks 1-3: Session 1 (Blocks 1, 2, 3)
- Blocks 4-6: Session 2 (Blocks 1, 2, 3)

**Included Participants (22 total):**
- Discrete Graded: LP001, LP002, LP009, LP010, LP012, LP014, LP015, LP020, LP021, LP022, LP023
- Interpolated Graded: LP003, LP004, LP006, LP007, LP008, LP013, LP016, LP017, LP018, LP019, LP028
