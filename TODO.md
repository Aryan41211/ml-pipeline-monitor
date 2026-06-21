# TODO - Automated Retraining Pipeline (Triggered by Data Drift)

- [ ] Step 1: Add configuration keys for automated retraining in config.yaml + src/config_loader.py
- [ ] Step 2: Extend DB schema (src/database.py) with retraining tables + accessor functions
- [ ] Step 3: Implement services/automated_retraining_service.py (threshold, cooldown, retrain, compare, promote)
- [ ] Step 4: Integrate automated retraining trigger into services/drift_service.py
- [ ] Step 5: Create new Streamlit page pages/7_Automated_Retraining.py (history, comparison, promotion, logs)
- [ ] Step 6: Add sidebar nav link in src/ui_theme.py
- [ ] Step 7: Add/Update tests for cooldown + promotion decision + DB persistence
- [ ] Step 8: Run pytest, fix failures, update README/config/requirements if needed
- [ ] Step 9: Git operations: commit `feat: add automated retraining pipeline` and push to GitHub
