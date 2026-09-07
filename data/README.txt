DATA — READ THIS FIRST

Placing real data here
======================
The clinical cohort used in the paper is NOT bundled (it is patient data from
the study site). To reproduce the analysis on your real data:

1. Put a single patient-level file here, e.g.  data/clinical_cohort.csv  (or .xlsx).
2. Columns may be English or Chinese. Map them to canonical names in
   configs/column_mapping.yaml  (the defaults already cover the common
   clinical-sheet headers: 序号, 患者姓名, 性别, 年龄, E/e'值, 左心房容积指数, Pd值, ...).
3. If your sheet already contains a computed  Pd value, set
   has_pd_column: true  (config.yaml) and it is used directly;
   otherwise the pipeline computes  Pd = 0.45*z(E/e') + 0.35*z(LAVI) + 0.20*z(TR velocity)
   for you (z-scores from the cohort reference distribution). EITHER WAY the
   three label-defining inputs (E/e', LAVI, TR velocity) are still tracked so the
   Reviewer-5 label-exclusion analysis can run correctly.
4. Point config.yaml data.clinical_data_path at your file.

What's here now
===============
  synthetic_cohort.csv   — SYNTHETIC smoke-test cohort (SYN-001..SYN-118,
                           synthetic_data=1, simulation_seed=20260730).
                           It is only for checking the pipeline runs.
                           It must NOT be used to regenerate paper results.
  data_schema.csv        — canonical field dictionary (English).
