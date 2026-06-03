# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkopeRules is a Python library for interpretable anomaly detection. It fits an ensemble of decision trees, extracts logical rules from tree paths, evaluates precision/recall per rule on out-of-bag samples, and deduplicates/clusters similar rules — returning human-readable conditions like `"age > 35 and income <= 50000"`.

## Development Setup

No build system. Install dependencies manually:

```bash
pip install numpy pandas scikit-learn
```

Basic usage:

```python
from skope_rules import SkopeRules
clf = SkopeRules(feature_names=['age', 'income'])
clf.fit(X, y)
predictions = clf.predict(X_test)
```

There are no tests, linting configs, or CI pipelines in this repository.

## Architecture

Two modules, flat structure:

### [rule.py](rule.py)
- `Rule` class: wraps a rule string (e.g. `"X > 5 and Y < 10"`), implements factorization to eliminate redundant conditions (e.g. `"X > 3 and X > 5"` → `"X > 5"`), and `__eq__`/`__hash__` for deduplication.
- `replace_feature_name()`: regex utility for substituting feature names in rule strings.
- `evaluate()`: converts a rule string to a binary activation mask over a DataFrame.

> Note: `evaluate()` at line 75 appears to be accidentally unindented — it belongs inside the `Rule` class.

### [skope_rules.py](skope_rules.py)
- `SkopeRules` (sklearn `BaseEstimator`): the main estimator.
- **Fit pipeline:** trains `BaggingClassifier` ensemble → `_tree_to_rules()` extracts decision paths recursively → `_eval_rule_perf()` scores each rule on OOB samples → deduplication via `Rule` class + `deduplicate()` (semantic tree clustering) → optional graph-based `_cluster_rules()`.
- **Predict pipeline:** `decision_function()` returns weighted anomaly scores (weight = rule precision); `predict()` thresholds; `rules_vote()` / `score_top_rules()` / `predict_top_rules()` offer alternative scoring.
- **In-progress clustering additions** (lines ~360–420, ~700–750): `_build_activation_sets()`, `_cluster_rules()`, `_jaccard()`, `_adjusted_similarity()`, `_asymmetric_similarity()`, `_ensemble_score()` — partially integrated, not wired into `fit()` yet.

## Key Conventions

- scikit-learn estimator API: `fit()` sets `self.rules_` as the primary output attribute.
- Rules are stored as `(rule_str, precision, recall, nb_support)` tuples in `self.rules_`.
- Binary classification assumed: 0 = normal, 1 = anomaly/target class.
- OOB evaluation is used throughout to avoid leakage — `bootstrap=True` is required on the BaggingClassifier.
- Some newer comments are in Vietnamese; treat them as in-progress developer notes.

## Known Issues

- `DecisionTreeClassifier` is instantiated with `n_estimators=100` and `warn_start=True` (line ~279) — neither is a valid parameter for `DecisionTreeClassifier`; this is a bug likely from a copy-paste from `BaggingClassifier`.
- The new clustering features (`_cluster_rules`, `_ensemble_score`) are implemented but not yet called from `fit()`.
