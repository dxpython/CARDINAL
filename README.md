# CARDINAL — Reproducible Research Code

Implementation and reproduction package for

> **CARDINAL: Physiology-Guided Directional Structure and Prototype Learning
> for Small-Sample Echocardiographic Assessment**

---

## 1. Dataset

The patient-level clinical cohort is obtained from the study site and, for
privacy and institutional reasons, is **not distributed in this repository**.
The pipeline is data-agnostic: it reads a single patient-level sheet
(CSV or Excel) whose location you set in `configs/config.yaml` under
`data.clinical_data_path`.

* **Column mapping** — the default `configs/column_mapping.yaml` already maps the
  common clinical-sheet headers (both English and Chinese, e.g. `序号`,
  `患者姓名`, `性别`, `年龄`, `E/e'值`, `左心房容积指数`, `Pd值`, …) to the
  canonical variables. Add a row there if your headers differ.
* **Reference index** — if your sheet already contains a computed `Pd` value, set
  `data.has_pd_column: true`. Otherwise the code computes
  `Pd = 0.45·z(E/e') + 0.35·z(LAVI) + 0.20·z(TR velocity)` automatically.
* **Smoke-test data** — a clearly labelled synthetic cohort
  (`data/synthetic_cohort.csv`, `synthetic_data=1`) is included **only** to verify
  that the pipeline and tests run end-to-end. It is not clinical data.

See `configs/data_schema.yaml` for the full variable dictionary and
`data/README.txt` for data-preparation instructions.

## 2. Environment

```bash
pip install -r requirements.txt            # core dependencies
# optional comparator packages (TabNet, TabPFN, FT-Transformer, TabTransformer):
pip install pytorch-tabnet "tabpfn>=2.0" pytorch-tabular
# or
conda env create -f environment.yml && conda activate cardinal
```

## 3. Reproduce the paper results

From the repository root (`code/`):

```bash
python scripts/run_all.py          # all experiments, then figures and tables
```

Or run the individual steps in order:

```bash
python scripts/run_clinical_main.py      # main test-set results, DeLong/McNemar, repeated runs
python scripts/run_small_sample.py       # 5%/10%/20%/40%/100% training-fraction analysis
python scripts/run_uncertainty.py        # MC-dropout T sensitivity, calibration, risk-coverage
python scripts/run_graph_analysis.py     # prior/learned graph, PC/NOTEARS/DAG-GNN concordance
python scripts/run_ablation.py           # Full CARDINAL vs component ablations
python scripts/run_label_exclusion.py    # Reviewer-5 exclusion (remove E/e', LAVI, TR velocity)
python scripts/run_public_benchmarks.py  # UCI / Heart Failure / MIMIC-IV benchmarks
python scripts/make_all_figures.py       # data-driven result figures
python scripts/make_all_tables.py        # LaTeX table export (main + supplementary)
```

Output is written at runtime to `outputs/{results,figures,tables}/`. A full
mapping from Methods / Table / Figure to the producing script and output file is
in `REPRODUCIBILITY.md`.

## 4. Repository layout

```
code/
  configs/          config.yaml, data_schema.yaml, column_mapping.yaml,
                    prior_graphs/{clinical,uci,hf,mimic}_prior.csv
  data/             
  src/
    config.py, utils.py
    data/           loader, schema, labels, preprocessing, split, dataset
    models/         cardinal.py, graph.py, prototype.py, baselines.py
    metrics/        discrimination, calibration, classification
    statistics/     delong, mcnemar, paired
    uncertainty/    risk-coverage
    graph/          prior, compare, structure, stability
    experiments/    common.py  (leak-controlled train/eval harness)
  scripts/          run_*.py, make_all_figures.py, make_all_tables.py, _common.py
  tests/            pytest suite (leakage, split, exclusion, model, metrics, reproducibility)
  outputs/          created at runtime
```

