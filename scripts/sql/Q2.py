import os
import random
import time
import argparse
import json
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm, expon
from collections import defaultdict

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
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
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
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
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
        if len(constants) != k: raise ValueError(f"Length of constants list ({len(constants)}) must be equal to k ({k}).")
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
    print(f"Saving phi_map to {file_path}...")
    serializable_map = {}
    for k, v in phi_map.items():
        smt2_program = f"(assert {v.sexpr()})"
        serializable_map[str(k)] = smt2_program
    with open(file_path, 'w') as f:
        json.dump(serializable_map, f, indent=2)

def load_phi_map(file_path):
    if not os.path.exists(file_path):
        return None
    print(f"Loading phi_map from cache: {file_path}...")
    with open(file_path, 'r') as f:
        serializable_map = json.load(f)

    phi_map = {}
    for k_str, smt2_program in serializable_map.items():
        key = eval(k_str)
        try:
            assertions = parse_smt2_string(smt2_program)
            if assertions and len(assertions.children()) > 0:
                original_formula = assertions.children()[0]
                phi_map[key] = original_formula
            else:
                print(f"WARNING: Could not parse a valid formula for key {key} from cache. Skipping.")
        except Exception as e:
            print(f"WARNING: Failed to parse entry for key {key} due to error: {e}. Skipping.")
    return phi_map

def to_z3_representation(entity_data, schema, p_key="id"):
    z3_repres = {}
    entity_id_str = str(entity_data.get(p_key, 'unknown'))
    for prop_key in schema:
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
            if prop_key == 'c_acctbal':
                 z3_repres[prop_key] = RealVal(str(value))
            else:
                 z3_repres[prop_key] = StringVal(str(value))
    return z3_repres

def check_certainty(phi_map, targeted_keys):
    print("\n" + "="*60)
    print("  CERTAINTY CHECK (Validity Test)")
    print("="*60)

    results_map = {
        "CERTAIN": [],
        "UNCERTAIN": [],
        "UNKNOWN": [],
        "NOT_FOUND": []
    }

    for key in targeted_keys:
        if key not in phi_map:
            print(f"WARNING: Key {key}: Formula not found in phi_map.")
            results_map["NOT_FOUND"].append(key)
            continue

        phi = phi_map[key]

        s = Solver()
        s.add(Not(phi))
        check_result = s.check()

        if check_result == unsat:
            results_map["CERTAIN"].append(key)
        elif check_result == sat:
            results_map["UNCERTAIN"].append(key)
        else:
            results_map["UNKNOWN"].append(key)

    for category in ["CERTAIN", "UNCERTAIN", "UNKNOWN"]:
        items = results_map[category]
        print(f"\n--- {category} KEYS ({len(items)}) ---")
        if items:
            print(f"{items}")
        else:
            print("None")

    total_evaluated = len(results_map["CERTAIN"]) + \
                      len(results_map["UNCERTAIN"]) + \
                      len(results_map["UNKNOWN"])

    print("\n" + "-"*60)
    print("  STATISTICS")
    print("-"*60)

    if total_evaluated > 0:
        pct_certain = (len(results_map["CERTAIN"]) / total_evaluated) * 100
        pct_uncertain = (len(results_map["UNCERTAIN"]) / total_evaluated) * 100
        pct_unknown = (len(results_map["UNKNOWN"]) / total_evaluated) * 100

        print(f"Total Evaluated: {total_evaluated}")
        print(f"Certain:         {pct_certain:6.2f}%  ({len(results_map['CERTAIN'])})")
        print(f"Uncertain:       {pct_uncertain:6.2f}%  ({len(results_map['UNCERTAIN'])})")
        print(f"Unknown:         {pct_unknown:6.2f}%  ({len(results_map['UNKNOWN'])})")
    else:
        print("No keys were successfully evaluated.")

    print("="*60 + "\n")
    return results_map

def run_mu_plot_multiple(instance_entity_id_pairs, phi_map, dist_config, k_config, shuffle, output_csv_path=None):
    plt.figure(figsize=(12, 7))
    k_mode, k_values = k_config['mode'], k_config['range']
    all_results_data = []
    
    for instance_path, entity_id in instance_entity_id_pairs:
        print(f"\n=== Running mu estimation for Entity ID={entity_id} using '{dist_config['name']}' distribution ===")
        if entity_id not in phi_map:
            print(f"WARNING: Entity ID {entity_id} not found in the loaded phi_map. Skipping.")
            continue
            
        phi_formula = phi_map[entity_id]
        global_null_vars_map.clear()
        
        for var in collect_vars(phi_formula): 
            global_null_vars_map[var.decl().name()] = var
            
        nulls = list(global_null_vars_map.keys())

        if not nulls:
            s = Solver()
            s.add(phi_formula)
            is_certain = "CERTAINLY TRUE" if s.check() == sat else "CERTAINLY FALSE"
            print(f"INFO: No nulls found. The result is {is_certain}.")
            continue

        print(f"Found {len(nulls)} nulls in the formula for Entity ID {entity_id}.")
        mu_vals = []
        base_constants = extract_constants_from_formula(phi_formula)

        for k_val in k_values:
            num_constants = int(k_val*len(nulls)) if k_mode=='relative' else k_val
            if num_constants < 1: num_constants = 1
            trials = 20
            mu_samples = []
            
            for _ in range(trials):
                extended_consts = extend_constant_universe(base_constants, num_constants, shuffle)
                k = len(extended_consts)
                if dist_config['name']=='uniform': dist=UniformDistribution(extended_consts)
                elif dist_config['name']=='zipfian': dist=ZipfianDistribution(k=k,s=dist_config['params']['s'],constants=extended_consts)
                elif dist_config['name']=='normal': dist=TruncatedNormalDistribution(k=k,constants=extended_consts, **dist_config['params'])
                elif dist_config['name']=='exponential': dist=ExponentialDistribution(k=k,constants=extended_consts, **dist_config['params'])
                mu_samples.append(estimate_mu(phi_formula, nulls, dist))
                
            mu_avg = sum(mu_samples)/len(mu_samples)
            mu_vals.append(mu_avg)
            all_results_data.append({'id':str(entity_id), 'mu_estimate':mu_avg, **dist_config.get('params',{})})
            
        plt.plot(k_values, mu_vals, label=f"ID: {entity_id}")
        
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        pd.DataFrame(all_results_data).to_csv(output_csv_path, index=False)
        
    plt.title(f"Certainty Estimate Convergence - {dist_config['name'].title()} Dist - {'Shuffled' if shuffle else 'Ordered'}")
    plt.xlabel("k = |Constants| / |Nulls|" if k_mode == 'relative' else "k = |Constants|")
    plt.ylabel("Certainty Estimate")
    plt.grid(True)
    plt.legend(title="Entity ID")
    plt.tight_layout()
    
    if output_csv_path:
        base, _ = os.path.splitext(output_csv_path)
        plt.savefig(f"{base}.png")
        plt.savefig(f"{base}.eps", format='eps')
    plt.close()

def estimate_mu(formula, nulls, dist, num_samples=200):
    sat_count = 0
    for _ in range(num_samples):
        valuation = {n: dist.sample() for n in nulls}
        subs = [(global_null_vars_map[n], StringVal(str(valuation[n]))) for n in nulls if n in global_null_vars_map]
        s = Solver()
        s.add(substitute(formula, subs))
        if s.check() == sat: sat_count += 1
    return sat_count / num_samples

def collect_vars(expr, seen=None):
    if seen is None: seen = set()
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED: seen.add(expr)
    else:
        for child in expr.children(): collect_vars(child, seen)
    return seen

def extract_constants_from_formula(formula):
    constants = set()
    _collect_string_literals(formula, constants)
    return list(constants)

def _collect_string_literals(expr, constants_set):
    if is_string_value(expr): constants_set.add(expr.as_string())
    elif expr.children():
        for child in expr.children(): _collect_string_literals(child, constants_set)

def extend_constant_universe(constants, target_count, shuffle=False):
    if len(constants) >= target_count: extended_list = random.sample(constants, target_count)
    else:
        extras = [f"dummy_const_{i}" for i in range(target_count - len(constants))]
        extended_list = constants + extras
    if shuffle: random.shuffle(extended_list)
    return extended_list

def load_and_symbolize_tpch_data_q2(instance_folder):
    print(f"Loading TPC-H data for Customer Query from {instance_folder}...")
    loaded_dfs = {}
    try:
        for table_name in ['customer', 'orders']:
            file_path = os.path.join(instance_folder, f'{table_name}.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{table_name}.csv not found.")

            df = pd.read_csv(file_path, delimiter=',', header=0, dtype=str)
            df.columns = [col.lower() for col in df.columns]
            loaded_dfs[table_name] = df
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

    customer_df = loaded_dfs['customer']
    orders_df = loaded_dfs['orders']

    customers_z3 = {}
    for _, row in customer_df.iterrows():
        key = pd.to_numeric(row['c_custkey'], errors='coerce')
        if pd.notna(key):
            customers_z3[int(key)] = to_z3_representation(
                row.to_dict(), ['c_custkey', 'c_nationkey', 'c_acctbal'], 'c_custkey'
            )

    orders_z3 = []
    for _, row in orders_df.iterrows():
        orders_z3.append(
            to_z3_representation(row.to_dict(), ['o_custkey'], 'o_custkey')
        )

    customer_df.set_index('c_custkey', inplace=True)
    return customer_df, customers_z3, orders_z3

def generate_phi_for_query_2(targeted_custkeys, customer_df, customers_z3, orders_z3):
    phi_map = {}
    target_nations = {'13', '31', '23', '29', '30', '18', '17'}
    print(f"Generating phi for {len(targeted_custkeys)} targeted customer keys...")

    customer_df['c_acctbal_num'] = pd.to_numeric(customer_df['c_acctbal'], errors='coerce')
    customer_df['c_nationkey_str'] = customer_df['c_nationkey'].astype(str)

    subquery_df = customer_df[
        (customer_df['c_nationkey_str'].isin(target_nations)) &
        (customer_df['c_acctbal_num'] > 0.00)
    ]
    avg_balance = subquery_df['c_acctbal_num'].mean()

    all_order_custkeys_z3 = [o_z3['o_custkey'] for o_z3 in orders_z3]

    for c_key in targeted_custkeys:
        if c_key not in customers_z3:
            print(f"WARNING: c_custkey {c_key} not found in customer table. Skipping.")
            continue

        c_z3 = customers_z3[c_key]

        cond_nation = Or([c_z3['c_nationkey'] == StringVal(n) for n in target_nations])

        try:
            c_raw = customer_df.loc[str(c_key)]
            customer_balance = c_raw['c_acctbal_num']
            cond_acctbal = BoolVal(pd.notna(customer_balance) and customer_balance > avg_balance)
        except KeyError:
            print(f"WARNING: c_custkey {c_key} not found in DataFrame index. Skipping.")
            continue

        exists_disjuncts = [ock == c_z3['c_custkey'] for ock in all_order_custkeys_z3]
        cond_no_orders = Not(Or(exists_disjuncts)) if exists_disjuncts else BoolVal(True)

        phi_formula = And(cond_nation, cond_acctbal, cond_no_orders)
        phi_map[c_key] = simplify(phi_formula)

    return phi_map

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/user/Desktop/SQL/tpch-sf0.01-10nulls/")
    parser.add_argument("--cache_path", type=str, default="phi_Q2_cache.json")
    parser.add_argument("--keys_file", type=str, default="targeted_custkeys.json")
    parser.add_argument("--mode", type=str, choices=["certainty", "analysis"], default="analysis")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    DATA_PATH = args.data_path
    PHI_CACHE_PATH = args.cache_path
    KEYS_INPUT_FILE = args.keys_file

    k_configs_to_test = [{'mode': 'relative', 'range': range(1, 31)}]
    dist_configs_to_test = [
        {"name": "normal", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "mu": 50.0, "sigma": 15.0}},
        {"name": "exponential", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "scale": 20.0}}
    ]
    shuffle_options = [True]

    default_targeted_custkeys = [
        24, 75, 87, 147, 225, 234, 342, 357, 360, 432, 441, 543, 672,
        753, 768, 771, 828, 846, 942, 1005, 1023, 1026, 1230, 1305,
        1308, 1311, 1323, 1377, 1395, 1422, 1431, 1437, 1470
    ]

    if os.path.exists(KEYS_INPUT_FILE):
        print(f"Loading targeted customer keys from {KEYS_INPUT_FILE}...")
        with open(KEYS_INPUT_FILE, 'r') as f:
            targeted_custkeys = json.load(f)
    else:
        print(f"File {KEYS_INPUT_FILE} not found. Using default keys and saving to file.")
        targeted_custkeys = default_targeted_custkeys
        with open(KEYS_INPUT_FILE, 'w') as f:
            json.dump(targeted_custkeys, f)

    phi_map = None
    if phi_map is None:
        print("No cache found. Generating new phi_map for Customer Query...")
        symbolized_data = load_and_symbolize_tpch_data_q2(DATA_PATH)
        if symbolized_data:
            customer_df, customers_z3, orders_z3 = symbolized_data
            phi_map = generate_phi_for_query_2(targeted_custkeys, customer_df, customers_z3, orders_z3)
            save_phi_map(phi_map, PHI_CACHE_PATH)
        else:
            print("ERROR: Failed to load data. Exiting.")
            exit()

    if args.mode == 'certainty':
        check_certainty(phi_map, targeted_custkeys)

    elif args.mode == 'analysis':
        if not targeted_custkeys:
            print("No target customer keys to analyze.")
            exit()

        key_to_analyze = targeted_custkeys[0]
        print(f"Proceeding to estimation analysis for c_custkey: {key_to_analyze}")
        tpch_instance_ckey_pairs = [(DATA_PATH, key_to_analyze)]

        for k_config in k_configs_to_test:
            for dist_config in dist_configs_to_test:
                for shuffle in shuffle_options:
                    if dist_config['name'] == 'uniform' and shuffle:
                        continue

                    shuffle_str = "shuffled" if shuffle else "ordered"
                    print(f"\nSTARTING ANALYSIS for c_custkey={key_to_analyze}: dist={dist_config['name']}, k_mode={k_config['mode']}, shuffle={shuffle}")

                    output_filename = f"results/Query2_Analysis_{key_to_analyze}_{dist_config['name']}_{k_config['mode']}_{shuffle_str}.csv"

                    run_mu_plot_multiple(
                        tpch_instance_ckey_pairs,
                        phi_map,
                        dist_config,
                        k_config,
                        shuffle,
                        output_csv_path=output_filename
                    )