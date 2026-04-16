# ML Pipeline Monitor

A production-ready MLOps observability platform that gives data science teams real-time visibility into model training pipelines, experiment outcomes, and data distribution health — all through a professional Streamlit interface backed by a local SQLite experiment store.

---

## Overview

Managing ML experiments at scale requires more than a notebook.  This platform provides:

- **End-to-end pipeline orchestration** with per-stage progress tracking and structured logging
- **Experiment tracking** with metric persistence, filtering, and visual comparison
- **Model registry** with lifecycle stage management (development → staging → production)
- **Data drift detection** using the Kolmogorov-Smirnov test and Population Stability Index (PSI)
- **System resource monitoring** to correlate pipeline latency with host pressure

---

## Screenshots

| Page | Description |
|---|---|
| Overview | At-a-glance dashboard with KPIs, recent runs, and system health |
| Pipeline Runner | Configure dataset, algorithm, and hyperparameters; monitor live progress |
| Experiment Tracking | Compare runs across metrics with interactive Plotly charts |
| Model Registry | Manage model lifecycle stages; benchmark performance |
| Data Drift | KS test + PSI analysis with distribution overlay visualisations |

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI framework | [Streamlit](https://streamlit.io) |
| ML framework | [scikit-learn](https://scikit-learn.org) |
| Gradient boosting | [XGBoost](https://xgboost.readthedocs.io) |
| Statistical tests | [SciPy](https://scipy.org) |
| Visualisation | [Plotly](https://plotly.com/python/) |
| Data manipulation | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Persistence | SQLite (via Python stdlib `sqlite3`) |
| System monitoring | [psutil](https://psutil.readthedocs.io) |
| Config | [PyYAML](https://pyyaml.org) |
| Serialisation | [joblib](https://joblib.readthedocs.io) |

---

## Supported Algorithms

**Classification**
- Random Forest
- XGBoost
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

**Regression**
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Ridge Regression
- Support Vector Regressor (SVR)
- Decision Tree Regressor

---

## Supported Datasets

- Breast Cancer Wisconsin (classification)
- Wine Recognition (classification)
- Iris Species (classification)
- Handwritten Digits (classification)
- Synthetic Classification (2,000 samples, 20 features)
- Synthetic Regression (2,000 samples, 20 features)

All datasets are sourced from scikit-learn's built-in dataset library — no external downloads required.

---

## Project Structure

```
ml-pipeline-monitor/
├── app.py                          # Home dashboard (Streamlit entry point)
├── requirements.txt
├── setup.py
├── config.yaml                     # Application configuration
├── src/
│   ├── pipeline.py                 # Core ML pipeline with stage orchestration
│   ├── data_loader.py              # Dataset loading and splitting utilities
│   ├── database.py                 # SQLite persistence layer
│   ├── drift_detector.py           # KS test + PSI drift detection
│   └── system_monitor.py          # Host resource metrics (psutil)
├── pages/
│   ├── 1_Pipeline_Runner.py        # Interactive pipeline execution
│   ├── 2_Experiment_Tracking.py   # Experiment comparison and filtering
│   ├── 3_Model_Registry.py        # Model lifecycle management
│   └── 4_Data_Drift.py            # Distribution drift analysis
└── tests/
    └── test_pipeline.py            # pytest unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/manpatell/ml-pipeline-monitor.git
cd ml-pipeline-monitor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Running tests

```bash
pytest tests/ -v
```

---

## Configuration

Pipeline defaults and UI settings are controlled via `config.yaml`:

```yaml
pipeline:
  random_seed: 42
  test_size: 0.20
  cv_folds: 5

monitoring:
  drift_significance_level: 0.05
  psi_moderate_threshold: 0.10
  psi_significant_threshold: 0.25
```

The database path can be overridden with the `PIPELINE_DB` environment variable:

```bash
PIPELINE_DB=/data/experiments.db streamlit run app.py
```

---

## Pipeline Stages

Each training run proceeds through seven sequential stages:

1. **Data Validation** — schema checks, missing value counts, class balance
2. **Preprocessing** — StandardScaler fit on training split, applied to test
3. **Feature Analysis** — correlation matrix, high-correlation pair count, variance check
4. **Cross-Validation** — stratified K-fold CV with F1 (classification) or R² (regression)
5. **Training** — fit estimator on full training set with timing
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC / RMSE, MAE, R² on holdout
7. **Feature Importance** — tree-based importances or coefficient magnitudes

---

## Drift Detection

The drift analysis page compares a reference window against a current window using:

- **Kolmogorov-Smirnov test**: non-parametric two-sample test for distribution equality
- **Population Stability Index (PSI)**: measures how much the distribution of a variable has shifted

| PSI | Interpretation |
|---|---|
| < 0.10 | Stable — no action required |
| 0.10 – 0.25 | Moderate shift — monitor closely |
| > 0.25 | Significant shift — consider retraining |

---

## License

MIT License.  See [LICENSE](LICENSE) for details.
