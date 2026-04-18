# 🚀 QUICK START GUIDE

## Step 1: Setup Virtual Environment

```bash
cd "d:\workspace\New folder"

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Run Data Loading & Cleaning

```bash
# Option A: Using Jupyter Notebook (Recommended)
cd notebooks
jupyter notebook 01_data_exploration_and_cleaning.ipynb
# Open browser and run all cells (Kernel > Run All)

# Option B: Using Python directly
python -c "
import sys
sys.path.insert(0, '.')
from config.config import JOBS_CSV_PATH, RESUMES_CSV_PATH, DATA_CLEANING_CONFIG
from src.data_loader import load_and_explore_data
from src.data_cleaner import run_full_pipeline

jobs_df, resumes_df, _, _ = load_and_explore_data(str(JOBS_CSV_PATH), str(RESUMES_CSV_PATH))
cleaned_jobs, cleaned_resumes, train_jobs, test_jobs = run_full_pipeline(jobs_df, resumes_df, DATA_CLEANING_CONFIG)
print('✅ Cleaning complete!')
"
```

## Step 4: Verify Cleaned Data

```bash
# Check that files were created
ls data/processed/
# Should show:
# - jobs_cleaned.csv
# - resumes_cleaned.csv
# - train_data.csv
# - test_data.csv
```

## Expected Output

```
DATASETS LOADED SUCCESSFULLY
================================================================================
📊 Jobs Dataset:
   Shape: (1167, 5)
   Columns: ['job_id', 'category', 'job_title', 'job_description', 'job_skill_set']

📊 Resumes Dataset:
   Shape: (315, 11)
   Columns: ['Resume_ID', 'Name', 'Skills', 'Experience (Years)', 'Education', ...]

================================================================================
JOBS DATA CLEANING PIPELINE
================================================================================
Starting records: 1167
  ✓ Remove duplicate job IDs
  ✓ Filter for tech jobs (['INFORMATION-TECHNOLOGY'])
  ✓ Filter for relevant job titles
  ✓ Remove rows with missing critical columns
  ✓ Clean and standardize text fields
  ✓ Extract and validate skills
  ✓ Remove jobs with no extracted skills

✅ Jobs cleaning complete!
Final record count: 240 (removed: 927)

================================================================================
RESUMES DATA CLEANING PIPELINE
================================================================================
...
✅ Resumes cleaning complete!
Final record count: 315 (removed: 0)

================================================================================
📋 COMPREHENSIVE DATA CLEANING & PREPARATION SUMMARY
================================================================================
Dataset                 Record Count    Percentage
Jobs (Raw)              1167            100%
Jobs (Cleaned)          240             20.6%
Jobs (Train)            192             16.4%
Jobs (Test)             48              4.1%
Resumes (Raw)           315             100%
Resumes (Cleaned)       315             100%

✅ Data Preparation Complete!
   • Datasets saved to: data/processed
   • Ready for feature engineering and model training
   • Next step: Generate embeddings and build recommendation model
```

---

## File Structure After Cleaning

```
data/
├── raw/
│   ├── preprocessed_jobs.csv (original: 1,167 jobs)
│   └── preprocessed_resumes.csv (original: 315 resumes)
└── processed/
    ├── jobs_cleaned.csv (cleaned: 240 tech jobs)
    ├── resumes_cleaned.csv (cleaned: 315 resumes)
    ├── train_data.csv (192 jobs for training)
    └── test_data.csv (48 jobs for testing)
```

---

## What Gets Done

✅ **Data Cleaning**
- Removes 927 non-tech jobs (keeps only software engineer, data scientist, etc.)
- Removes duplicates
- Extracts skills from job descriptions
- Validates numeric fields
- Handles missing values

✅ **Feature Extraction**
- Parses skills from job descriptions and resumes
- Standardizes education levels
- Validates experience years and salary

✅ **Data Split**
- 80% (192 jobs) for training
- 20% (48 jobs) for testing

---

## Troubleshooting

### Error: "ModuleNotFoundError"
```bash
# Make sure you're in the project root directory
cd "d:\workspace\New folder"
```

### Error: "File not found"
```bash
# Verify raw data exists
ls data/raw/
# Should show preprocessed_jobs.csv and preprocessed_resumes.csv
```

### Slow Execution
- The cleaning process processes ~1,200 jobs, should take < 1 minute
- If slower, check if your disk is busy

---

## Next Steps

Once cleaning is complete:

1. **Feature Engineering** → Generate embeddings with Sentence-Transformers
2. **Model Training** → Train Random Forest / XGBoost on features
3. **Model Evaluation** → Test performance metrics
4. **Build Web App** → Create Streamlit interface
5. **Deploy** → Push to cloud (Streamlit Cloud / Hugging Face)

See `PROJECT_SETUP_SUMMARY.md` for detailed information on each phase.

---

**Questions?** Check the detailed documentation in `PROJECT_SETUP_SUMMARY.md`
