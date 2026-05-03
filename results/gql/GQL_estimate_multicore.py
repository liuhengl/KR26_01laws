import os
import random
import time
import argparse
import json
import re
import ast
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import truncnorm, expon
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

CACHE_FILES = [
    "cache_1_null-sf0.1_Q5.json",
    "cache_2_null-sf0.1_Q5.json",
    "cache_3_null-sf0.1_Q5.json",
    "cache_4_null-sf0.1_Q5.json",
    "cache_5_null-sf0.1_Q5.json"
]

TARGET_CACHE_FILE = "cache_5_null-sf0.1_Q5.json"

class Distribution:
    def sample(self): raise NotImplementedError

class UniformDistribution(Distribution):
    def __init__(self, constants):
        self.constants = constants
    def sample(self): return random.choice(self.constants)

class ZipfianDistribution(Distribution):
    def __init__(self, k, s, constants):
        self.constants = constants
        ranks = np.arange(1, k + 1)
        weights = 1.0 / (ranks ** s)
        self.probabilities = weights / np.sum(weights)
    def sample(self): return random.choices(self.constants, weights=self.probabilities, k=1)[0]

class TruncatedNormalDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, mu, sigma, k, constants):
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        trunc_norm_dist = truncnorm(a, b, loc=mu, scale=sigma)
        slice_probs = np.diff(trunc_norm_dist.cdf(slice_bounds))
        total = slice_probs.sum()
        self.probs = slice_probs / total if total > 0 else np.ones(k)/k
        self.probs = sorted(self.probs, reverse=True)
    def sample(self): return random.choices(self.constants, weights=self.probs, k=1)[0]

class ExponentialDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, scale, k, constants):
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        exp_dist = expon(scale=scale)
        slice_probs = np.diff(exp_dist.cdf(slice_bounds))
        total = slice_probs.sum()
        self.probs = slice_probs / total if total > 0 else np.ones(k)/k
        self.probs = sorted(self.probs, reverse=True)
    def sample(self): return random.choices(self.constants, weights=self.probs, k=1)[0]

def get_intersection_of_ids(base_dir="."):
    common_ids = None

    for fname in CACHE_FILES:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"Error: {fname} missing.")
            return set()

        with open(fpath, 'r') as f:
            keys = set(json.load(f).keys())

        if common_ids is None:
            common_ids = keys
        else:
            common_ids = common_ids.intersection(keys)

    return common_ids

def get_formulas_for_all_instances(target_id, base_dir="."):
    instance_data = []

    for i, fname in enumerate(CACHE_FILES):
        rate_val = (i + 1) * 0.01
        rate_label = f"{rate_val:.2f}"
        fpath = os.path.join(base_dir, fname)

        with open(fpath, 'r') as f:
            data = json.load(f)

        if target_id in data:
            instance_data.append((rate_label, data[target_id]))
        else:
            print(f"Warning: ID {target_id} missing from {fname}")

    return instance_data

def safe_parse_z3_with_nulls(smt_code):
    if not smt_code: return None
    null_var_pattern = re.compile(r'_null\d+')
    try:
        found_vars = set(null_var_pattern.findall(smt_code))
        declarations = "".join([f"(declare-const {var} String)\n" for var in found_vars])
        full_program = declarations + smt_code
        assertions = parse_smt2_string(full_program)
        return assertions[0] if len(assertions) > 0 else BoolVal(True)
    except Exception:
        return None

def collect_vars(expr, seen=None):
    if seen is None: seen = set()
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED and is_string(expr):
        seen.add(expr)
    elif expr.children():
        for child in expr.children(): collect_vars(child, seen)
    return seen

def _collect_string_literals(expr, constants_set):
    if is_string_value(expr): constants_set.add(expr.as_string())
    elif expr.children():
        for child in expr.children(): _collect_string_literals(child, constants_set)

def extract_constants_from_formula(formula):
    constants = set()
    _collect_string_literals(formula, constants)
    return list(constants)

def extend_constant_universe(base_constants, target_count):
    base_set = set(base_constants)
    if not base_set: base_set = {"DUMMY_CONST"}
    base_list = list(base_set)
    if len(base_list) >= target_count: return random.sample(base_list, target_count)
    else:
        extras = [f"dummy_{i}" for i in range(target_count - len(base_list))]
        return base_list + extras

def estimate_mu(formula, null_vars, dist, num_samples):
    sat_count = 0
    var_map = {v.decl().name(): v for v in null_vars}
    var_names = list(var_map.keys())

    for _ in range(num_samples):
        valuation = {name: dist.sample() for name in var_names}
        subs = [(var_map[name], StringVal(str(val))) for name, val in valuation.items()]
        instantiated = substitute(formula, subs)
        s = Solver()
        s.add(instantiated)
        if s.check() == sat: sat_count += 1
    return sat_count / num_samples

def run_trials_for_phi_group(job_params):
    dist_config = job_params['dist_config']
    work_items = job_params['work_items']
    k_config = job_params['k_config']
    NUM_TRIALS = 10

    dist_results = []
    dist_name = dist_config['name'].title()

    for entity_label, smt_code in work_items:
        phi = safe_parse_z3_with_nulls(smt_code)
        if phi is None: continue

        null_vars = list(collect_vars(phi))
        base_constants = extract_constants_from_formula(phi)
        num_nulls = len(null_vars)

        if num_nulls == 0:
            s = Solver()
            s.add(Not(phi))
            res = 1.0 if s.check() == unsat else 0.0
            for x_val in k_config['range']:
                for case in ['worst', 'average']:
                    dist_results.append({
                        'distribution': dist_name,
                        'x_axis_value': x_val,
                        'mu': res,
                        'case': case.title(),
                        'entity_id': entity_label
                    })
            continue

        for x_val in k_config['range']:
            target_k = int(x_val * num_nulls) if k_config['mode'] == 'relative' else x_val
            target_k = max(len(base_constants), target_k, 1)

            for case in ['worst', 'average']:
                trial_mus = []
                for _ in range(NUM_TRIALS):
                    extended_constants = extend_constant_universe(base_constants, target_k)
                    current_constants = list(extended_constants)

                    if case == 'average':
                        random.shuffle(current_constants)

                    k_curr = len(current_constants)

                    if dist_config['name'] == 'uniform':
                        dist = UniformDistribution(current_constants)
                    elif dist_config['name'] == 'zipfian':
                        dist = ZipfianDistribution(k_curr, dist_config['params']['s'], current_constants)
                    elif dist_config['name'] == 'normal':
                        p = dist_config['params']
                        dist = TruncatedNormalDistribution(p['lower_bound'], p['upper_bound'], p['mu'], p['sigma'], k_curr, current_constants)
                    elif dist_config['name'] == 'exponential':
                        p = dist_config['params']
                        dist = ExponentialDistribution(p['lower_bound'], p['upper_bound'], p['scale'], k_curr, current_constants)

                    mu_single = estimate_mu(phi, null_vars, dist, num_samples=200)
                    trial_mus.append(mu_single)

                mu_avg = sum(trial_mus) / len(trial_mus)

                dist_results.append({
                    'distribution': dist_name,
                    'x_axis_value': x_val,
                    'mu': mu_avg,
                    'case': case.title(),
                    'entity_id': entity_label
                })

    return dist_results

def plot_instance_comparison(df, k_config, target_id):
    dist_order = ["Uniform", "Zipfian", "Normal", "Exponential"]
    df = df[df['distribution'].isin(dist_order)]
    df = df.rename(columns={'entity_id': 'Null Rate'})

    xlabel = r'Effective Domain Ratio ($k/n_{\text{eff}}$)' if k_config['mode'] == 'relative' else 'k'

    for case_type in ["Worst", "Average"]:
        df_case = df[df['case'] == case_type]
        if df_case.empty: continue

        g = sns.relplot(
            data=df_case,
            x="x_axis_value",
            y="mu",
            col="distribution",
            col_wrap=4,
            col_order=dist_order,
            hue="Null Rate",
            style="Null Rate",
            kind="line",
            height=4,
            aspect=1.0,
            facet_kws={'sharex': True, 'sharey': True},
            markers=True,
            dashes=False
        )

        g.set_axis_labels(xlabel, r'$\mathbb{P}$ (Satisfiability)')
        g.set_titles(col_template="{col_name}")

        for ax in g.axes.flat:
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

        filename = f"GQL_Q5_Compare_Instances_{case_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_master_facet_grid(df, k_config, title_suffix=""):
    dist_order = ["Uniform", "Zipfian", "Normal", "Exponential"]
    df = df[df['distribution'].isin(dist_order)]

    if df.empty:
        print("Error: DataFrame for plotting is empty.")
        return

    g = sns.relplot(
        data=df, x="x_axis_value", y="mu",
        col="distribution", col_wrap=4, col_order=dist_order,
        style="case", kind="line", height=3.5, aspect=1.0,
        facet_kws={'sharex': True, 'sharey': True},
        markers=True, dashes=True
    )

    xlabel = r'Effective Domain Ratio ($k/n_{eff}$)' if k_config['mode'] == 'relative' else 'k'
    g.set_axis_labels(xlabel, r'$\mathbb{P}_k$ (Certainty)')
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flat:
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    filename = "GQL_Q5_probability_estimation.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = "GQL_Q5_probability_estimation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".", help="Directory containing cache files.")
    parser.add_argument("--compare_instances", action="store_true", help="Pick 1 ID and evaluate it across all 5 instance files.")
    parser.add_argument("--from_csv", type=str, help="Skip computation and plot directly from a CSV file.")
    args = parser.parse_args()

    k_config = {'mode': 'relative', 'range': range(1, 31, 1)}
    distributions = [
            {"name": "uniform"},
            {"name": "zipfian", "params": {"s": 0.5}},
            {"name": "normal", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "mu": 50.0, "sigma": 25.0}},
            {"name": "exponential", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "scale": 20.0}}
    ]

    if args.from_csv:
        if not os.path.exists(args.from_csv):
            print(f"Error: CSV file {args.from_csv} not found.")
            return

        df = pd.read_csv(args.from_csv)

        if args.compare_instances:
            target_label = f"From {os.path.basename(args.from_csv)}"
            plot_instance_comparison(df, k_config, target_label)
        else:
            plot_master_facet_grid(df, k_config)
        return

    if args.compare_instances:
        common_ids = get_intersection_of_ids(args.base_dir)
        if not common_ids: return

        all_cache_data = []
        for fname in CACHE_FILES:
            fpath = os.path.join(args.base_dir, fname)
            with open(fpath, 'r') as f:
                all_cache_data.append(json.load(f))

        valid_candidates = []
        for cid in common_ids:
            is_valid = True
            for data in all_cache_data:
                if cid not in data:
                    is_valid = False; break
                form = data[cid].strip()
                if "(assert true)" in form or form == "true":
                    is_valid = False; break

            if is_valid:
                valid_candidates.append(cid)

        if not valid_candidates:
            print("No suitable non-trivial IDs found.")
            return

        target_id = random.choice(valid_candidates)
        work_items = get_formulas_for_all_instances(target_id, args.base_dir)
    else:
        tpath = os.path.join(args.base_dir, TARGET_CACHE_FILE)
        if not os.path.exists(tpath): return
        
        with open(tpath, 'r') as f: 
            data = json.load(f)
            
        valid_keys = [k for k,v in data.items() if "(assert true)" not in v]
        selected_ids = random.sample(valid_keys, min(20, len(valid_keys)))
        work_items = [(k, data[k]) for k in selected_ids]
        target_id = "Aggregate_Sample"

    job_params_list = []
    for dist in distributions:
        job_params_list.append({
            'dist_config': dist,
            'work_items': work_items,
            'k_config': k_config
        })

    results_data = []
    with ProcessPoolExecutor(max_workers=len(distributions)) as executor:
        for res in tqdm(executor.map(run_trials_for_phi_group, job_params_list), total=len(distributions)):
            results_data.extend(res)

    if results_data:
        df = pd.DataFrame(results_data)
        out_csv = "GQL_Q5_Instance_Comparison.csv" if args.compare_instances else "GQL_Q5_Standard.csv"
        df.to_csv(out_csv, index=False)

        if args.compare_instances:
            plot_instance_comparison(df, k_config, target_id)
        else:
            plot_master_facet_grid(df, k_config)
    else:
        print("No results generated.")

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    main()