# Saved result views

`data/` contains saved aggregate and observation CSV files. `tables/` contains plain CSV exports of the eleven displayed numerical tables and the fixed-code comparison notes. These exports preserve recorded values and row order; they do not recalculate metrics.

Repeated model runs use stable labels such as `run_1`, `run_2`, and `run_3`. Run counts, standard deviations, intervals, and positive, zero, and negative observations remain in the corresponding views.

The files use different units of observation. Development configurations, query observations, predicate/run observations, and application aggregates must be interpreted separately. In the application files, `ALL_512` quality and `TIMED_128` cost refer to different cohorts. Data-role and protocol fields retain the recorded distinctions.

See `INDEX.csv` for file topics, row counts, and observation units, and `DATA_DICTIONARY.json` for measurement conventions. Blank cells retain missing metadata. These saved views do not include every input or raw request record needed to rerun all experiments.
