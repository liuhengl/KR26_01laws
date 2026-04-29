# Almost certain query answering over incomplete relational and graph data

Code and data for the KR2026 paper "Almost Certain Query Answering over Incomplete Relational and Graph Data".

## What this does

Given a (naive) query answer, the code builds a formula (condition phi) over makred nulls and checking validity with Z3. It also estimates the probability of correctness via Monte Carlo sampling under multiple distributions.

## Repository layout

```text
.
|-- data/
|   |-- sql/
|   |   `-- all_query_results.csv        # naive evaluation results for SQL queries
|   `-- gql/
|       `-- *_Q5.json                    # naive evaluation results for GQL
|-- tpch_experiments/                    # TPC-H instances with incomplete data
|-- snb_experiments/                     # LDBC SNB datasets with incomplete data
|-- results/
|   |-- sql/                             # SQL plots/CSVs + PDFs
|   `-- gql/                             # GQL caches/logs + PDFs
|-- scripts/
|   |-- sql/                             # Formula phi generation for SQL queries
|   |-- gql/                             # Formula phi generation for the GQL query
|   `-- controllers/
|-- verification_results/                # SQL verification outputs (Q1-Q4 logs + caches)
|-- requirements.txt
`-- README.md
```

## Setup

You need Python 3.9+, the Z3 SMT solver, and the usual scientific Python stack (pandas, numpy, matplotlib, scipy).

```bash
pip install -r requirements.txt
```

## SQL (TPC-H) experiments

### Certainty verification

Check which naive answers from `all_query_results.csv` are certain:

```bash
python scripts/controllers/sql_verify_all.py --query Q1 --mode certainty
python scripts/controllers/sql_verify_all.py --query Q2 --mode certainty
python scripts/controllers/sql_verify_all.py --query Q3 --mode certainty
python scripts/controllers/sql_verify_all.py --query Q4 --mode certainty
```

Results go to `verification_results/Q*/` as combined logs and cache files.

### Probabilistic convergence

Run Monte Carlo convergence analysis over false positive formulas:

```bash
python scripts/controllers/sql_multicore_estimation.py --query Q1 --base_dir verification_results
```

## GQL (LDBC SNB) experiments

### Certainty verification

```bash
python scripts/controllers/gql_verify_all.py --query Q5 --mode certainty
```

Logs and caches go to `results/gql/`.

### Convergence analysis

The GQL estimator lives in `results/gql/GQL_estimate_multicore.py` and runs parallel Monte Carlo estimation using the cached formulas from the verification step.

```bash
python results/gql/GQL_estimate_multicore.py --base_dir results/gql
```

Plots and CSVs end up in `results/gql/`.

## Notes

Certainty results are in per-query log files: `verification_results/Q4/Q4_combined_log.txt` for SQL, `results/gql/Q5_verification_log.txt` for GQL. Convergence plots and tables are under `results/sql/` and `results/gql/`; the PDFs there are the figures from the paper.

Some scripts write cache files (`phi_*.json`) so they don't have to rebuild formulas on reruns.

Naive SQL answers come from PostgreSQL v17 with the `ids4nulls` extension for marked nulls. Naive GQL answers come from Neo4j Cypher 5.20. The queries themselves are in `tpch_experiments/SQL_queries.txt` and `snb_experiments/Q5.cypher`.
