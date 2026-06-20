# TODO — Real-world datasets for ML Pipeline Monitor

- [ ] Gather current implementation details for dataset loading/splitting, dataset preview UI, feature store caching, pipeline stage assumptions, and experiment/model registry persistence.
- [ ] Design dataset download/processing strategy for:
  - [ ] Customer Churn Prediction
  - [ ] Credit Card Fraud Detection
  - [ ] House Price Prediction
- [ ] Implement `datasets/raw/`, `datasets/processed/`, automatic download-if-missing, and `metadata.json` generation/statistics.
- [ ] Update `src/data_loader.py` for new datasets while keeping backward compatibility with existing toy datasets.
- [ ] Update feature store and pipeline integration so cached splits and task selection work for real datasets.
- [ ] Update experiment tracking and model registry to store dataset/task correctly for new datasets.
- [ ] Enhance Dataset Preview page with correlation heatmap (and keep existing preview elements).
- [ ] Add/update tests to cover dataset loading, preview stats, caching behavior, and backward compatibility.
- [ ] Run `pytest -q` and fix any failures.
- [ ] Update `requirements.txt`, `config.yaml`, and `README.md` only if needed.
- [ ] Commit with message `feat: add real world datasets`.
- [ ] Push to GitHub.

