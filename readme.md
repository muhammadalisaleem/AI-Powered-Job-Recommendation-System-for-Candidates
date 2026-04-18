# AI-Powered Job Recommendation System for Candidates

## 🎯 Project Objective

Develop an AI system that empowers job seekers to upload their resume and instantly receive **personalized job recommendations** with match scores and improvement suggestions for software engineering, data science, and technical roles.

**Problem it solves**: Candidates waste time applying to irrelevant positions.  
**Solution**: Upload resume → Get top-ranked personalized recommendations with explanations.

---

## 📋 Project Phases & Implementation Plan

### Phase 1: Problem Definition ✓ (Finalized)
- **Challenge**: Candidates apply to mismatched positions inefficiently
- **Solution**: AI-powered matching system with transparent scoring
- **Stakeholders**: Job seekers, recruiters (future), hiring managers

---

### Phase 2: Data Preparation & Cleaning

#### Dataset Strategy
- **Source**: Preprocessed Job Postings Dataset + Resume Dataset (already available)
- **Format**: CSV files in `/datasets/` folder

#### Data Cleaning Tasks
- [ ] Remove duplicate job postings
- [ ] Filter for relevant roles only: Software Engineering, Data Science, ML, Web Development
- [ ] Handle missing values in: salary, location, skills required
- [ ] Keep only essential columns:
  - Job Title, Company, Description, Skills Required, Experience Level, Location, Job Type
- [ ] Detect and remove outliers (e.g., experience_years > 30)

#### Data Split
```
Training Set: 80%  → Model training & validation
Testing Set: 20%   → Final performance evaluation
```

#### Preprocessing Steps
1. **Normalization**: Standardize numerical features (years of experience, salary ranges)
2. **Categorical Encoding**:
   - One-Hot/Label Encoding: Job type, location categories
   - TF-IDF: Job descriptions and skill fields
   - Sentence-Transformer Embeddings: Semantic text understanding
3. **Feature Engineering**:
   - Extract technical skills from text
   - Calculate years of experience
   - Compute skill overlap percentage
   - Generate education level scores

---

### Phase 3: Model Selection & Implementation

#### Selected Algorithms
| Algorithm | Purpose | Rationale |
|-----------|---------|-----------|
| **Random Forest** | Primary ranking model | Robust, interpretable |
| **XGBoost/LightGBM** | Advanced boosting | Higher accuracy, faster training |
| **Sentence-Transformers** | Semantic matching | Understands job-resume context |
| **Hybrid** | Final ranking | Combines embeddings + ML scores |

#### Model Architecture
```
Resume Upload
    ↓
Feature Extraction & Embeddings
    ↓
ML Model Scoring (Random Forest/XGBoost)
    ↓
Cosine Similarity Scoring (Embeddings)
    ↓
Weighted Combination
    ↓
Top 10-15 Jobs Ranked
```

#### Training Process
1. Train on 80% dataset
2. Apply 5-fold cross-validation
3. Tune hyperparameters with GridSearchCV/Optuna:
   - Number of trees, learning rate, max depth, etc.
4. Validate on 20% test set

---

### Phase 4: Model Evaluation

#### Performance Metrics

**For Match Score Prediction (Regression)**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) → Target: **> 0.75**

**For Job Ranking Quality (Ranking)**
- Precision@5 → Target: **> 80%**
- Precision@10
- Recall@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

#### Model Comparison
- Compare: Random Forest vs XGBoost vs Cosine Similarity
- Select best performer based on metrics
- Document findings in results table

---

### Phase 5: Web App Development & Deployment

#### Streamlit Application Features
| Feature | Description |
|---------|-------------|
| **Resume Upload** | PDF file upload (text extraction) |
| **Job Recommendations** | Display top 10 jobs with match % |
| **Explanations** | "Why this job matches you" insights |
| **Improvement Tips** | Missing skills & experience gaps |
| **Filters** | Remote, Salary range, Experience level |

#### Deployment Options
- **Streamlit Cloud** (recommended, free tier available)
- **Hugging Face Spaces** (alternative free option)

---

### Phase 6: Documentation & Reporting

#### Deliverables
- [ ] Complete project report including:
  - Data cleaning procedures
  - Preprocessing methodology
  - Model comparison results table
  - Performance metrics summary
- [ ] Clean GitHub repository with structure:
  ```
  project/
  ├── data/
  ├── notebooks/
  ├── src/
  │   ├── preprocessing.py
  │   ├── models.py
  │   └── evaluation.py
  ├── app.py (Streamlit)
  ├── requirements.txt
  └── README.md
  ```

---

## 🎓 Success Criteria

- ✅ R² > 0.75 on test set
- ✅ Precision@5 > 80%
- ✅ RMSE minimized
- ✅ Functional web app deployed
- ✅ Clear documentation
- ✅ Reproducible code

---

## 📂 Project Structure

```
datasets/
├── preprocessed_jobs.csv      (Job postings)
└── preprocessed_resumes.csv   (Resume data)

notebooks/
└── exploration.ipynb          (EDA & analysis)

src/
├── data_preparation.py        (Cleaning & preprocessing)
├── feature_engineering.py     (Feature extraction)
├── model_training.py          (Model building)
├── evaluation.py              (Metrics & comparison)
└── utils.py                   (Helper functions)

app.py                          (Streamlit web interface)
requirements.txt               (Dependencies)
results/
├── model_performance.csv
└── final_report.md
```

---

## 🚀 Next Steps

1. **Start Phase 2**: Explore datasets and begin data cleaning
2. **Set up environment**: Install required libraries (pandas, scikit-learn, transformers, etc.)
3. **EDA**: Analyze data distributions and missing values
4. **Build preprocessing pipeline**
5. **Proceed to feature engineering**

---

## 📚 Technologies & Libraries

- **Data Processing**: pandas, numpy, scikit-learn
- **NLP & Embeddings**: transformers (sentence-transformers), TF-IDF
- **Models**: scikit-learn, XGBoost, LightGBM
- **Web App**: Streamlit
- **Evaluation**: scikit-learn metrics, custom ranking metrics
- **Utilities**: matplotlib, seaborn (visualization)
