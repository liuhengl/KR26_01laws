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

global_null_vars_map = {}

class Distribution:
    def sample(self): raise NotImplementedError

class UniformDistribution(Distribution):
    def __init__(self, constants):
        if not constants: raise ValueError("Constant list cannot be empty.")
        self.constants = constants
    def sample(self): return random.choice(self.constants)

class ZipfianDistribution(Distribution):
    def __init__(self, k, s, constants):
        if k <= 0: raise ValueError("'k' must be positive.")
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
        self.constants = constants
        ranks, weights = np.arange(1, k + 1), 1.0 / (np.arange(1, k + 1) ** s)
        total_weight = np.sum(weights)
        self.probabilities = weights / total_weight if total_weight > 0 else np.ones(k) / k
    def sample(self): return random.choices(self.constants, weights=self.probabilities, k=1)[0]

class TruncatedNormalDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, mu, sigma, k, constants):
        if k <= 0: raise ValueError("'k' must be positive.")
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        trunc_norm_dist = truncnorm(a, b, loc=mu, scale=sigma)
        slice_probabilities = np.diff(trunc_norm_dist.cdf(slice_bounds))
        total_prob = slice_probabilities.sum()
        if total_prob > 0: slice_probabilities /= total_prob
        else: slice_probabilities = np.ones(k) / k
        self.constant_probabilities = sorted(slice_probabilities, reverse=True)
    def sample(self): return random.choices(self.constants, weights=self.constant_probabilities, k=1)[0]

class ExponentialDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, scale, k, constants):
        if k <= 0: raise ValueError("'k' must be positive.")
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        exp_dist = expon(scale=scale)
        slice_probabilities = np.diff(exp_dist.cdf(slice_bounds))
        total_prob = slice_probabilities.sum()
        if total_prob > 0: slice_probabilities /= total_prob
        else: slice_probabilities = np.ones(k) / k
        self.constant_probabilities = sorted(slice_probabilities, reverse=True)
    def sample(self): return random.choices(self.constants, weights=self.constant_probabilities, k=1)[0]

def parse_smt_string_to_z3(smt_code):
    if not smt_code: return None
    null_var_pattern = re.compile(r'\|NULL:[^|]+\|')
    try:
        found_vars = set(null_var_pattern.findall(smt_code))
        declarations = "".join([f"(declare-const {var} String)\n" for var in found_vars])
        full_program = declarations + smt_code
        assertions = parse_smt2_string(full_program)
        return assertions[0] if len(assertions) > 0 else BoolVal(True)
    except Exception as e:
        print(f"Z3 Parsing failed: {e}")
        return None

def lookup_formula_in_cache(cache_path, target_key):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        str_key = str(target_key)
        if str_key in data:
            return data[str_key]
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in {cache_path}")
        return None

def extract_uncertain_formulas(query_name, base_output_dir):
    log_path = os.path.join(base_output_dir, query_name, f"{query_name}_combined_log.txt")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return {}

    instance_pattern = re.compile(r"=====\s*INSTANCE\s*(\d+)")
    cache_path_pattern = re.compile(r"--cache_path\s+([^\s]+)")
    work_map = {}
    current_instance = None
    current_cache_path = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        inst_match = instance_pattern.search(line)
        if inst_match:
            current_instance = int(inst_match.group(1))
            current_cache_path = None
            i += 1; continue

        if "--cache_path" in line:
            path_match = cache_path_pattern.search(line)
            if path_match:
                raw_path = path_match.group(1)
                filename = os.path.basename(raw_path)
                current_cache_path = os.path.join(base_output_dir, query_name, filename)
            i += 1; continue

        if ("--- UNCERTAIN PAIRS" in line or "--- UNCERTAIN KEYS" in line) and (i + 1 < len(lines)):
            data_line = lines[i+1].strip()
            if data_line and data_line != "None":
                try:
                    uncertain_keys = ast.literal_eval(data_line)
                    if uncertain_keys and current_instance is not None and current_cache_path:
                        for key in uncertain_keys:
                            work_map[(current_instance, key)] = current_cache_path
                except Exception: pass
            i += 1; continue
        i += 1
        
    return work_map

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
    if len(base_set) >= target_count: return random.sample(list(base_set), target_count)
    else:
        extras = [f"dummy_{i}" for i in range(target_count - len(base_set))]
        return list(base_set) + extras

def collect_vars(expr, seen=None):
    if seen is None: seen = set()
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED and is_string(expr): seen.add(expr)
    elif expr.children():
        for child in expr.children(): collect_vars(child, seen)
    return seen

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

def safe_parse_z3_with_nulls(smt_code):
    if not smt_code: return None
    null_var_pattern = re.compile(r'\|NULL:[^|]+\|')
    try:
        found_vars = set(null_var_pattern.findall(smt_code))
        declarations = "".join([f"(declare-const {var} String)\n" for var in found_vars])
        full_program = declarations + smt_code
        assertions = parse_smt2_string(full_program)
        return assertions[0] if len(assertions) > 0 else BoolVal(True)
    except Exception as e:
        print(f"Worker Parsing Error: {e}")
        return None

def run_trials_for_phi_group(job_params):
    dist_config = job_params['dist_config']
    work_items = job_params['work_items'] 
    k_config = job_params['k_config']
    
    dist_results = []
    dist_name = dist_config['name'].title()
    loaded_files_cache = {}

    for (inst, key), cache_path in work_items:
        entity_id = f"I{inst}_{key}"
        
        if cache_path not in loaded_files_cache:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    loaded_files_cache[cache_path] = json.load(f)
            else:
                continue
        
        json_data = loaded_files_cache[cache_path]
        str_key = str(key) 
        
        if str_key not in json_data:
            found = False
            for k in json_data.keys():
                if k.replace(" ", "") == str_key.replace(" ", ""):
                    str_key = k
                    found = True
                    break
            if not found:
                continue

        smt_code = json_data[str_key]
        phi = safe_parse_z3_with_nulls(smt_code)
        if phi is None:
            break
            
        null_vars = list(collect_vars(phi))
        base_constants = extract_constants_from_formula(phi)
        num_nulls = len(null_vars)
        
        if num_nulls == 0: continue

        for x_val in k_config['range']:
            target_k = int(x_val * num_nulls) if k_config['mode'] == 'relative' else x_val
            target_k = max(len(base_constants), target_k)
            if target_k == 0: target_k = 1

            extended_constants = extend_constant_universe(base_constants, target_k)
            
            for case in ['worst', 'average']:
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
                
                mu = estimate_mu(phi, null_vars, dist, num_samples=150)
                
                dist_results.append({
                    'distribution': dist_name,
                    'x_axis_mode': k_config['mode'],
                    'x_axis_value': x_val,
                    'mu': mu,
                    'case': case.title(),
                    'entity_id': entity_id,
                    'num_nulls': num_nulls
                })
                
    return dist_results

def plot_master_facet_grid(df, k_config, query_name):
    print("Generating aggregated plot...")
    dist_order = ["Uniform", "Zipfian", "Normal", "Exponential"]
    df = df[df['distribution'].isin(dist_order)]

    g = sns.relplot(
        data=df,
        x="x_axis_value",
        y="mu",
        col="distribution",
        col_wrap=4,
        col_order=dist_order,
        style="case",
        kind="line",
        height=3.5,
        aspect=1.0,
        facet_kws={'sharex': True, 'sharey': True},
        markers=True,
        dashes=True
    )
    
    xlabel = r'Effective Domain Ratio ($k/n_{eff}$)' if k_config['mode'] == 'relative' else 'k = Number of Constants'
    g.set_axis_labels(xlabel, r'$\mathbb{P}_k$ (Certainty)')
    g.set_titles(col_template="{col_name}")
    
    for ax in g.axes.flat:
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    filename = f"{query_name}_aggregated_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./verification_results")
    parser.add_argument("--from_csv", type=str)
    args = parser.parse_args()

    k_config = {'mode': 'relative', 'range': range(1, 31, 1)} 

    if args.from_csv:
        if not os.path.exists(args.from_csv):
            print(f"CSV file not found: {args.from_csv}")
            return
            
        df = pd.read_csv(args.from_csv)
        plot_master_facet_grid(df, k_config, args.query)
        return 

    work_map = extract_uncertain_formulas(args.query, args.base_dir)
    
    if not work_map:
        return

    all_items = list(work_map.items())
    sample_size = 20
    if len(all_items) > sample_size:
        selected_items = random.sample(all_items, sample_size)
    else:
        selected_items = all_items

    distributions = [
        {"name": "uniform"},
        {"name": "zipfian", "params": {"s": 0.5}},
        {"name": "normal", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "mu": 50.0, "sigma": 25.0}},
        {"name": "exponential", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "scale": 20.0}}
    ]

    job_params_list = []
    for dist in distributions:
        job_params_list.append({
            'dist_config': dist,
            'work_items': selected_items,
            'k_config': k_config
        })

    results_data = []
    with ProcessPoolExecutor(max_workers=len(distributions)) as executor:
        for res in tqdm(executor.map(run_trials_for_phi_group, job_params_list), total=len(distributions)):
            results_data.extend(res)

    if results_data:
        df = pd.DataFrame(results_data)
        csv_name = f"{args.query}_uncertain_analysis_full_log_sf01.csv"
        df.to_csv(csv_name, index=False)
        plot_master_facet_grid(df, k_config, args.query)

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    main()