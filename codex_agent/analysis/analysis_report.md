# Codex Agent Execution Analysis Report

## Summary Statistics

- **Total Instances Analyzed**: 179
- **System Failures (Excluded)**: 12 (timeout, rate limit, git errors)
- **Agent Successes**: 73 (40.8%)
- **Agent Failures**: 106 (59.2%)

### Agent Failure Breakdown

- **Complete Miss**: 42 instances (39.6% of failures)
- **Severe Precision Failure**: 20 instances (18.9% of failures)
- **Poor Entity Localization**: 19 instances (17.9% of failures)
- **Moderate Performance**: 11 instances (10.4% of failures)
- **File Only No Entities**: 7 instances (6.6% of failures)
- **Severe Recall Failure**: 7 instances (6.6% of failures)

## Average Metrics (Agent Performance Only)

- **File-level Recall**: 0.637
- **File-level Precision**: 0.285
- **Entity-level Recall**: 0.381
- **Entity-level Precision**: 0.230

## Execution Patterns

- **Avg Reasoning Steps**: 0.8
- **Avg Commands Executed**: 3.2

### Most Common Tools Used

- `rg`: 281 times
- `sed`: 190 times
- `ls`: 61 times
- `find`: 15 times
- `nl`: 10 times
- `python`: 8 times
- `cat`: 7 times
- `./ci/code_checks.sh`: 1 times

## Sample Failure Cases (Agent Performance Issues)

### Complete Miss (42 instances)

#### Instance: `scikit-learn__scikit-learn-25525`

**Problem**: Extend SequentialFeatureSelector example to demonstrate how to use negative tol

### Describe the bug

I utilized the **SequentialFeatureSelector** for feature selection in my code, with the directi...

**Ground Truth Files**: 

**Predicted Files**: sklearn/feature_selection/_sequential.py, sklearn/feature_selection/tests/test_sequential.py, examples/feature_selection/plot_select_from_model_diabetes.py, doc/modules/feature_selection.rst, doc/modules/classes.rst

**Performance**:
- File Recall: 0.000
- File Precision: 0.000
- Entity Recall: 0.000
- Correct: 0 files, 0 entities
- Missed: 0 files, 0 entities

---

#### Instance: `huggingface__transformers-30`

**Problem**: [Feature request] Add example of finetuning the pretrained models on custom corpus...

**Ground Truth Files**: src/transformers/modeling_utils.py, src/transformers/trainer.py, src/transformers/training_args.py

**Predicted Files**: docs/source/en/training.md, docs/source/en/task_summary.md, examples/pytorch/language-modeling/README.md, examples/pytorch/language-modeling/run_clm.py, examples/pytorch/language-modeling/run_mlm.py

**Performance**:
- File Recall: 0.000
- File Precision: 0.000
- Entity Recall: 0.000
- Correct: 0 files, 0 entities
- Missed: 3 files, 10 entities

**Missed Files**: src/transformers/modeling_utils.py, src/transformers/trainer.py, src/transformers/training_args.py

---

### Severe Precision Failure (20 instances)

#### Instance: `pallets__flask-2813`

**Problem**: Allow flexible routing with SERVER_NAME config

### Expected Behavior

Deployed a flask application which is reachable over multiple domains and ports:
- external via load balancer: `client - Host:...

**Ground Truth Files**: src/flask/app.py, tests/test_blueprints.py

**Predicted Files**: src/flask/app.py, src/flask/ctx.py, src/flask/sansio/app.py, docs/config.rst, tests/test_reqctx.py

**Performance**:
- File Recall: 0.500
- File Precision: 0.200
- Entity Recall: 0.333
- Correct: 1 files, 1 entities
- Missed: 1 files, 2 entities

**Missed Files**: tests/test_blueprints.py

---

#### Instance: `pandas-dev__pandas-17200`

**Problem**: read_json(lines=True) broken for s3 urls in Python 3 (v0.20.3)

#### Code Sample, a copy-pastable example if possible

Using Python
```python
import pandas as pd
inputdf = pd.read_json(path_or_bu...

**Ground Truth Files**: pandas/io/json/json.py, pandas/tests/io/json/test_pandas.py

**Predicted Files**: pandas/io/json/json.py, pandas/io/common.py, pandas/io/s3.py, pandas/tests/io/json/test_compression.py, pandas/tests/io/json/test_readlines.py

**Performance**:
- File Recall: 0.500
- File Precision: 0.200
- Entity Recall: 0.000
- Correct: 1 files, 0 entities
- Missed: 1 files, 4 entities

**Missed Files**: pandas/tests/io/json/test_pandas.py

---

### Poor Entity Localization (19 instances)

#### Instance: `pallets__flask-593`

**Problem**: Nestable blueprints

I'd like to be able to register "sub-blueprints" using `Blueprint.register_blueprint(*args, **kwargs)`. This would register the nested blueprints with an app when the "parent" is ...

**Ground Truth Files**: src/flask/app.py, src/flask/blueprints.py, tests/test_blueprints.py

**Predicted Files**: src/flask/blueprints.py, src/flask/app.py, tests/test_blueprints.py, docs/blueprints.rst

**Performance**:
- File Recall: 1.000
- File Precision: 0.750
- Entity Recall: 0.231
- Correct: 3 files, 3 entities
- Missed: 0 files, 10 entities

---

#### Instance: `pallets__flask-3555`

**Problem**: Remove simplejson

In modern Python it's unlikely to be significantly better than the built-in `json`. The module used by `JSONMixin` is overridable, so users can plug it in again if they want.

See...

**Ground Truth Files**: src/flask/json/__init__.py, src/flask/json/tag.py, tests/test_helpers.py

**Predicted Files**: src/flask/json/__init__.py, docs/api.rst, docs/installation.rst

**Performance**:
- File Recall: 0.333
- File Precision: 0.333
- Entity Recall: 0.294
- Correct: 1 files, 5 entities
- Missed: 2 files, 12 entities

**Missed Files**: src/flask/json/tag.py, tests/test_helpers.py

---

## Sample Success Cases

### Instance: `pallets__flask-2264`

**Problem**: Handle app factory in FLASK_APP

`FLASK_APP=myproject.app:create_app('dev')`
[
Gunicorn does this with `eval`](https://github.com/benoitc/gunicorn/blob/fbd151e9841e2c87a18512d71475bcff863a5171/gunic...

**Metrics**:
- File Recall: 1.000
- Entity Recall: 0.500

**Correctly Located Files**: flask/cli.py, tests/test_cli.py

**Execution**: 0 reasoning steps, 0 commands

---

### Instance: `pandas-dev__pandas-11080`

**Problem**: PERF: checking is_monotonic_increasing/decreasing before sorting on an index

We don't keep the sortedness state in an index per-se, but it is rather cheap to check
- `is_monotonic_increasing` or `is_...

**Metrics**:
- File Recall: 1.000
- Entity Recall: 1.000

**Correctly Located Files**: pandas/core/frame.py

**Execution**: 0 reasoning steps, 0 commands

---

### Instance: `psf__requests-3698`

**Problem**: AttributeError: 'NoneType' object has no attribute 'read'

Hello :)

After a recent upgrade for our [coala](https://github.com/coala/coala) project to `requests` 2.12.1 we encounter an exception in ...

**Metrics**:
- File Recall: 1.000
- Entity Recall: 0.500

**Correctly Located Files**: requests/models.py, tests/test_requests.py

**Execution**: 0 reasoning steps, 0 commands

---

