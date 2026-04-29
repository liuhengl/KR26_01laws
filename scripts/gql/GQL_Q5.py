import os
import random
import time
import argparse
import json
import sys
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm, expon

global_null_vars_map = {}

class Distribution:
    def sample(self):
        raise NotImplementedError

class UniformDistribution(Distribution):
    def __init__(self, constants):
        if not constants: raise ValueError("Constant list cannot be empty.")
        self.constants = constants
    def sample(self):
        return random.choice(self.constants)

class ZipfianDistribution(Distribution):
    def __init__(self, k, s, constants):
        if k <= 0: raise ValueError("Number of constants 'k' must be positive.")
        if len(constants) != k: raise ValueError(f"Length ({len(constants)}) != k ({k}).")
        self.constants = constants
        ranks = np.arange(1, k + 1)
        weights = 1.0 / (ranks ** s)
        total_weight = np.sum(weights)
        self.probabilities = weights / total_weight if total_weight > 0 else np.ones(k) / k
    def sample(self):
        return random.choices(self.constants, weights=self.probabilities, k=1)[0]

class TruncatedNormalDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, mu, sigma, k, constants):
        if k <= 0: raise ValueError("Number of slices 'k' must be positive.")
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
        trunc_norm_dist = truncnorm(a, b, loc=mu, scale=sigma)
        cdf_values = trunc_norm_dist.cdf(slice_bounds)
        slice_probabilities = np.diff(cdf_values)
        if slice_probabilities.sum() > 0: slice_probabilities /= slice_probabilities.sum()
        else: slice_probabilities = np.ones(k) / k
        self.constant_probabilities = sorted(slice_probabilities, reverse=True)
    def sample(self):
        return random.choices(self.constants, weights=self.constant_probabilities, k=1)[0]

class ExponentialDistribution(Distribution):
    def __init__(self, lower_bound, upper_bound, scale, k, constants):
        if k <= 0: raise ValueError("Number of slices 'k' must be positive.")
        self.constants = constants
        slice_bounds = np.linspace(lower_bound, upper_bound, k + 1)
        exp_dist = expon(scale=scale)
        cdf_values = exp_dist.cdf(slice_bounds)
        slice_probabilities = np.diff(cdf_values)
        if slice_probabilities.sum() > 0: slice_probabilities /= slice_probabilities.sum()
        else: slice_probabilities = np.ones(k) / k
        self.constant_probabilities = sorted(slice_probabilities, reverse=True)
    def sample(self):
        return random.choices(self.constants, weights=self.constant_probabilities, k=1)[0]

def save_phi_map(phi_map, file_path):
    serializable_map = {}
    for k, v in phi_map.items():
        serializable_map[str(k)] = f"(assert {v.sexpr()})"
    with open(file_path, 'w') as f:
        json.dump(serializable_map, f, indent=2)

def load_phi_map(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    phi_map = {}
    for k_str, smt2 in data.items():
        try:
            assertions = parse_smt2_string(smt2)
            if assertions: phi_map[k_str] = assertions[0].children()[0]
        except Exception as e:
            print(f"Warning: Failed to parse {k_str}: {e}")
    return phi_map

def check_certainty(phi_map, targeted_pairs):
    print("\n" + "="*60)
    print("  CERTAINTY CHECK")
    print("="*60)
    
    results = {"CERTAIN": [], "UNCERTAIN": [], "NOT_FOUND": []}
    
    for path, entity_id in targeted_pairs:
        eid_str = str(entity_id)
        if eid_str not in phi_map:
            results["NOT_FOUND"].append(eid_str)
            continue
            
        phi = phi_map[eid_str]
        
        if is_true(phi):
            results["CERTAIN"].append(eid_str)
            continue
        if is_false(phi):
            results["UNCERTAIN"].append(eid_str)
            continue

        s = Solver()
        s.add(Not(phi))
        
        if s.check() == unsat:
            results["CERTAIN"].append(eid_str)
        else:
            results["UNCERTAIN"].append(eid_str)

    print(f"Certain: {len(results['CERTAIN'])}")
    print(f"Uncertain: {len(results['UNCERTAIN'])}")
    print(f"Not Found: {len(results['NOT_FOUND'])}")
    return results

def estimate_mu(formula, nulls, distribution, num_samples=200):
    sat_count = 0
    for _ in range(num_samples):
        valuation = {n: distribution.sample() for n in nulls}
        subs = [(global_null_vars_map[n], StringVal(str(valuation[n]))) 
                for n in nulls if n in global_null_vars_map]
        
        instantiated = substitute(formula, subs)
        if is_true(simplify(instantiated)):
            sat_count += 1
        else:
            s = Solver()
            s.add(instantiated)
            if s.check() == sat:
                sat_count += 1
    return sat_count / num_samples

def run_mu_plot_multiple(instance_pid_pairs, phi_map, dist_config, k_config, shuffle, output_csv_path=None):
    plt.figure(figsize=(12, 7))
    k_mode, k_values = k_config['mode'], k_config['range']
    all_results = []

    for instance_path, pid in instance_pid_pairs:
        eid_str = str(pid)
        
        if eid_str not in phi_map:
            continue
            
        φ = phi_map[eid_str]
        global_null_vars_map.clear()
        for var in collect_vars(φ): 
            if is_string(var): global_null_vars_map[var.decl().name()] = var
        nulls = list(global_null_vars_map.keys())

        if not nulls:
            continue

        base_constants = extract_constants_from_formula(φ)
        mu_vals = []

        for k_val in k_values:
            num_constants = int(k_val * len(nulls)) if k_mode == 'relative' else k_val
            if num_constants < 1: num_constants = 1
            
            trials = 20
            mu_samples = []
            for _ in range(trials):
                extended_consts = extend_constant_universe(base_constants, num_constants, shuffle)
                k_curr = len(extended_consts)
                
                if dist_config['name'] == 'uniform': 
                    dist = UniformDistribution(extended_consts)
                elif dist_config['name'] == 'zipfian': 
                    dist = ZipfianDistribution(k=k_curr, s=dist_config['params']['s'], constants=extended_consts)
                elif dist_config['name'] == 'normal': 
                    dist = TruncatedNormalDistribution(k=k_curr, constants=extended_consts, **dist_config['params'])
                elif dist_config['name'] == 'exponential': 
                    dist = ExponentialDistribution(k=k_curr, constants=extended_consts, **dist_config['params'])
                
                mu_samples.append(estimate_mu(φ, nulls, dist))
            
            mu_avg = sum(mu_samples) / len(mu_samples)
            mu_vals.append(mu_avg)
            all_results.append({'id': eid_str, 'k': num_constants, 'mu': mu_avg, **dist_config.get('params', {})})

        plt.plot(k_values, mu_vals, label=f"ID: {eid_str}")

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        pd.DataFrame(all_results).to_csv(output_csv_path, index=False)
        
    plt.title(f"Certainty Convergence - {dist_config['name'].title()} - {'Shuffled' if shuffle else 'Ordered'}")
    plt.xlabel("k parameter")
    plt.ylabel("Certainty Estimate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_csv_path:
        base, _ = os.path.splitext(output_csv_path)
        plt.savefig(f"{base}.png")
    plt.close()

def to_z3_representation(entity_data, schema_internal_list):
    z3_repres = {}
    entity_id_str = str(entity_data.get('id', 'unknown_entity_id'))
    for prop_key in schema_internal_list:
        value = entity_data.get(prop_key)
        if isinstance(value, str) and (value.startswith("_null") or value.startswith("NULL:")):
            s_name = value
            if s_name not in global_null_vars_map: global_null_vars_map[s_name] = String(s_name)
            z3_repres[prop_key] = global_null_vars_map[s_name]
        elif pd.isna(value):
            s_name = f"missing_{entity_id_str}_{prop_key}"
            if s_name not in global_null_vars_map: global_null_vars_map[s_name] = String(s_name)
            z3_repres[prop_key] = global_null_vars_map[s_name]
        else:
            z3_repres[prop_key] = StringVal(str(value))
    return z3_repres

def collect_vars(expr, seen=None):
    if seen is None: seen = set()
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        seen.add(expr)
    else:
        for child in expr.children():
            collect_vars(child, seen)
    return seen

def extract_constants_from_formula(formula):
    constants = set()
    def _collect(expr):
        if is_string_value(expr): constants.add(expr.as_string())
        elif expr.children():
            for child in expr.children(): _collect(child)
    _collect(formula)
    constants.add("Firefox")
    constants.add("Chrome")
    return list(constants)

def extend_constant_universe(constants, target_count, shuffle=False):
    if len(constants) >= target_count:
        extended = random.sample(constants, target_count)
    else:
        extras = [f"dummy_{i}" for i in range(target_count - len(constants))]
        extended = constants + extras
    if shuffle: random.shuffle(extended)
    return extended

def load_snb_data(instance_folder):
    all_data = {}
    files_to_load = {
        "persons": ("person_0_0.csv", ['id:ID(Person)', 'browserUsed', 'locationIP']),
        "posts": ("post_0_0.csv", ['id:ID(Post)']),
        "comments": ("comment_0_0.csv", ['id:ID(Comment)', 'locationIP']),
        "post_hasCreator_person": ("post_hasCreator_person_0_0.csv", [':START_ID(Post)', ':END_ID(Person)']),
        "comment_hasCreator_person": ("comment_hasCreator_person_0_0.csv", [':START_ID(Comment)', ':END_ID(Person)']),
        "comment_replyOf_post": ("comment_replyOf_post_0_0.csv", [':START_ID(Comment)', ':END_ID(Post)']),
        "person_likes_post": ("person_likes_post_0_0.csv", [':START_ID(Person)', ':END_ID(Post)']),
    }
    
    for key, (fname, cols) in files_to_load.items():
        path = os.path.join(instance_folder, fname)
        if not os.path.exists(path):
            all_data[key] = pd.DataFrame(columns=cols)
            continue
        all_data[key] = pd.read_csv(path, sep='|', usecols=cols, dtype=str, on_bad_lines='warn')

    persons_map = {row['id:ID(Person)']: {'id': row['id:ID(Person)'], 'browser': row.get('browserUsed'), 'locationIP': row.get('locationIP')} for _, row in all_data["persons"].iterrows()}
    posts_map = {row['id:ID(Post)']: {'id': row['id:ID(Post)']} for _, row in all_data["posts"].iterrows()}
    comments_map = {row['id:ID(Comment)']: {'id': row['id:ID(Comment)'], 'locationIP': row.get('locationIP')} for _, row in all_data["comments"].iterrows()}
    
    post_creators_map = {row[':START_ID(Post)']: row[':END_ID(Person)'] for _, row in all_data["post_hasCreator_person"].iterrows() if row[':START_ID(Post)'] in posts_map and row[':END_ID(Person)'] in persons_map}
    comment_creators_map = {row[':START_ID(Comment)']: row[':END_ID(Person)'] for _, row in all_data["comment_hasCreator_person"].iterrows() if row[':START_ID(Comment)'] in comments_map and row[':END_ID(Person)'] in persons_map}
    
    post_direct_replies_map = {}
    for _, row in all_data["comment_replyOf_post"].iterrows():
        post_direct_replies_map.setdefault(row[':END_ID(Post)'], []).append(row[':START_ID(Comment)'])
        
    person_likes_post_map = {}
    for _, row in all_data["person_likes_post"].iterrows():
        person_likes_post_map.setdefault(row[':END_ID(Post)'], []).append(row[':START_ID(Person)'])
        
    return (persons_map, posts_map, comments_map, post_creators_map, comment_creators_map, post_direct_replies_map, person_likes_post_map)

def generate_phi_for_persons(persons_map, posts_map, post_creators_map, comment_creators_map, post_direct_replies_map, person_likes_post_map, comments_map, target_ids=None):
    schema = ['browser', 'locationIP']
    phi_map = {}
    targets = set(str(t) for t in target_ids) if target_ids else None
    
    for pid, pdata in persons_map.items():
        if targets and pid not in targets:
            continue
            
        p_sym = to_z3_representation(pdata, schema)
        browser_p = p_sym['browser']
        ip_p = p_sym['locationIP']
        
        disjuncts_ψ1 = []
        for post_id, creator_id in post_creators_map.items():
            if creator_id != pid: continue
            for liker_id in person_likes_post_map.get(post_id, []):
                if liker_id not in persons_map: continue
                liker_sym = to_z3_representation(persons_map[liker_id], ['browser'])
                disjuncts_ψ1.append(browser_p != liker_sym['browser'])
        
        ψ1 = Or(disjuncts_ψ1) if disjuncts_ψ1 else BoolVal(False)
        
        conjuncts_ψ2 = []
        for post_id, creator_id in post_creators_map.items():
            if creator_id != pid: continue
            for cmt_id in post_direct_replies_map.get(post_id, []):
                comment_data = comments_map.get(cmt_id)
                if not comment_data: continue
                cmt_sym = to_z3_representation(comment_data, ['locationIP'])
                conjuncts_ψ2.append(cmt_sym['locationIP'] != ip_p)
        
        ψ2 = And(conjuncts_ψ2) if conjuncts_ψ2 else BoolVal(True)
        
        phi_map[pid] = simplify(And(ψ1, ψ2))
        
    return phi_map

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default="phi_cache.json")
    parser.add_argument("--keys_file", type=str, default="targeted_keys.json")
    parser.add_argument("--mode", type=str, choices=["certainty", "analysis"], default="certainty")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    k_configs = [{'mode': 'absolute', 'range': range(1, 121, 5)}]
    dist_configs = [
        {"name": "uniform"},
        {"name": "zipfian", "params": {"s": 0.5}},
        {"name": "normal", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "mu": 50.0, "sigma": 25.0}},
        {"name": "exponential", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "scale": 20.0}}
    ]
    shuffle_options = [True]

    if os.path.exists(args.keys_file):
        with open(args.keys_file, 'r') as f:
            raw_input = json.load(f)
            
        if raw_input and isinstance(raw_input[0], (list, tuple)):
            targeted_pairs = [tuple(p) for p in raw_input]
            target_ids = [str(p[1]) for p in targeted_pairs]
        else:
            target_ids = [str(tid) for tid in raw_input]
            targeted_pairs = [(args.data_path, tid) for tid in target_ids]
    else:
        target_ids = ["1050"]
        targeted_pairs = [(args.data_path, "1050")]
        with open(args.keys_file, 'w') as f:
            json.dump(target_ids, f)

    phi_map = load_phi_map(args.cache_path)
    
    if not phi_map:
        data = load_snb_data(args.data_path)
        if data:
            (persons_map, posts_map, comments_map, post_creators_map, 
             comment_creators_map, post_direct_replies_map, 
             person_likes_post_map) = data
            
            phi_map = generate_phi_for_persons(
                persons_map, posts_map, post_creators_map,
                comment_creators_map, post_direct_replies_map,
                person_likes_post_map, comments_map,
                target_ids=target_ids
            )
            save_phi_map(phi_map, args.cache_path)
        else:
            sys.exit(1)

    if args.mode == 'certainty':
        check_certainty(phi_map, targeted_pairs)

    elif args.mode == 'analysis':
        for k_conf in k_configs:
            for dist_conf in dist_configs:
                for shuffle in shuffle_options:
                    if dist_conf['name'] == 'uniform' and shuffle: continue
                    
                    shuffle_str = "shuffled" if shuffle else "ordered"
                    output_file = f"results/Q5_Analysis_{dist_conf['name']}_{shuffle_str}.csv"
                    
                    run_mu_plot_multiple(
                        targeted_pairs, 
                        phi_map, 
                        dist_conf, 
                        k_conf, 
                        shuffle, 
                        output_csv_path=output_file
                    )