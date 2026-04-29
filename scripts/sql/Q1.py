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
    serializable_map = {}
    for k, v in phi_map.items():
        serializable_map[str(k)] = f"(assert {v.sexpr()})"
    with open(file_path, 'w') as f:
        json.dump(serializable_map, f, indent=2)

def load_phi_map(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        serializable_map = json.load(f)

    phi_map = {}
    for k_str, smt2_program in serializable_map.items():
        key_tuple = eval(k_str)
        try:
            assertions = parse_smt2_string(smt2_program)
            if assertions and len(assertions[0].children()) > 0:
                phi_map[key_tuple] = assertions[0].children()[0]
        except Exception as e:
            print(f"WARNING: Failed to parse entry for key {key_tuple}. Skipping.")
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
            z3_repres[prop_key] = StringVal(str(value))
    return z3_repres

def check_certainty(phi_map, targeted_pairs):
    print("\n" + "="*60)
    print("  CERTAINTY CHECK")
    print("="*60)

    results_map = {"CERTAIN": [], "UNCERTAIN": [], "UNKNOWN": [], "NOT_FOUND": []}

    for pair in targeted_pairs:
        pair_key = tuple(pair)

        if pair_key not in phi_map:
            results_map["NOT_FOUND"].append(pair_key)
            continue

        phi = phi_map[pair_key]
        s = Solver()
        s.add(Not(phi))
        check_result = s.check()

        if check_result == unsat:
            results_map["CERTAIN"].append(pair_key)
        elif check_result == sat:
            results_map["UNCERTAIN"].append(pair_key)
        else:
            results_map["UNKNOWN"].append(pair_key)

    for category in ["CERTAIN", "UNCERTAIN", "UNKNOWN"]:
        items = results_map[category]
        print(f"\n--- {category} PAIRS ({len(items)}) ---")
        if items:
            print(f"{items}")

    total_evaluated = len(results_map["CERTAIN"]) + len(results_map["UNCERTAIN"]) + len(results_map["UNKNOWN"])

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
        print("No pairs were successfully evaluated.")
    print("="*60 + "\n")

    return results_map

def run_mu_plot_multiple(instance_entity_id_pairs, phi_map, dist_config, k_config, shuffle, output_csv_path=None):
    plt.figure(figsize=(12, 7))
    k_mode, k_values = k_config['mode'], k_config['range']
    all_results_data = []
    
    for instance_path, entity_id in instance_entity_id_pairs:
        if entity_id not in phi_map:
            continue
            
        phi_formula = phi_map[entity_id]
        global_null_vars_map.clear()
        
        for var in collect_vars(phi_formula): 
            global_null_vars_map[var.decl().name()] = var
            
        nulls = list(global_null_vars_map.keys())

        if not nulls:
            s = Solver()
            s.add(phi_formula)
            continue

        mu_vals = []
        base_constants = extract_constants_from_formula(phi_formula)
        
        for k_val in k_values:
            num_constants = int(k_val*len(nulls)) if k_mode=='relative' else k_val
            if num_constants < 1: num_constants = 1
            trials, mu_samples = 20, []
            
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
    plt.legend(title="Entity Pair")
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

TPCH_SCHEMA = {
    'nation': {'N_NATIONKEY': 'Int64', 'N_NAME': 'str', 'N_REGIONKEY': 'Int64', 'N_COMMENT': 'str'},
    'supplier': {'S_SUPPKEY': 'Int64', 'S_NAME': 'str', 'S_ADDRESS': 'str', 'S_NATIONKEY': 'Int64', 'S_PHONE': 'str', 'S_ACCTBAL': 'float64', 'S_COMMENT': 'str'},
    'orders': {'O_ORDERKEY': 'Int64', 'O_CUSTKEY': 'Int64', 'O_ORDERSTATUS': 'str', 'O_TOTALPRICE': 'float64', 'O_ORDERDATE': 'str', 'O_ORDERPRIORITY': 'str', 'O_CLERK': 'str', 'O_SHIPPRIORITY': 'Int64', 'O_COMMENT': 'str'},
    'lineitem': {'L_ORDERKEY': 'Int64', 'L_PARTKEY': 'Int64', 'L_SUPPKEY': 'Int64', 'L_LINENUMBER': 'Int64', 'L_QUANTITY': 'float64', 'L_EXTENDEDPRICE': 'float64', 'L_DISCOUNT': 'float64', 'L_TAX': 'float64', 'L_RETURNFLAG': 'str', 'L_LINESTATUS': 'str', 'L_SHIPDATE': 'str', 'L_COMMITDATE': 'str', 'L_RECEIPTDATE': 'str', 'L_SHIPINSTRUCT': 'str', 'L_SHIPMODE': 'str', 'L_COMMENT': 'str'}
}

def load_and_symbolize_tpch_data_optimized(instance_folder):
    loaded_dfs = {}
    try:
        for table_name in ['nation', 'supplier', 'orders', 'lineitem']:
            file_path = os.path.join(instance_folder, f'{table_name}.csv')
            if not os.path.exists(file_path):
                file_path_tbl = os.path.join(instance_folder, f'{table_name}.tbl')
                if os.path.exists(file_path_tbl):
                    file_path = file_path_tbl
                else:
                    raise FileNotFoundError(f"Neither {table_name}.csv nor {table_name}.tbl found.")

            df = pd.read_csv(file_path, delimiter=',', header=0, dtype=str)
            df.columns = [col.lower() for col in df.columns]
            loaded_dfs[table_name] = df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    nation_df, supplier_df, orders_df, lineitem_df = [loaded_dfs.get(t) for t in ['nation', 'supplier', 'orders', 'lineitem']]

    nations_z3, suppliers_z3, orders_z3 = {}, {}, {}
    for _, row in nation_df.iterrows():
        key = pd.to_numeric(row['n_nationkey'], errors='coerce')
        if pd.notna(key): nations_z3[int(key)] = to_z3_representation(row.to_dict(), ['n_nationkey', 'n_name'], 'n_nationkey')
    for _, row in supplier_df.iterrows():
        key = pd.to_numeric(row['s_suppkey'], errors='coerce')
        if pd.notna(key): suppliers_z3[int(key)] = to_z3_representation(row.to_dict(), ['s_suppkey', 's_nationkey'], 's_suppkey')
    for _, row in orders_df.iterrows():
        key = pd.to_numeric(row['o_orderkey'], errors='coerce')
        if pd.notna(key): orders_z3[int(key)] = to_z3_representation(row.to_dict(), ['o_orderkey', 'o_orderstatus'], 'o_orderkey')

    lineitem_raw_by_okey = defaultdict(list)
    lineitem_z3_by_okey = defaultdict(list)
    for _, row in lineitem_df.iterrows():
        okey = pd.to_numeric(row['l_orderkey'], errors='coerce')
        if pd.notna(okey):
            z3_rep = to_z3_representation(row.to_dict(), ['l_orderkey', 'l_suppkey'], 'l_orderkey')
            lineitem_raw_by_okey[int(okey)].append(row.to_dict())
            lineitem_z3_by_okey[int(okey)].append(z3_rep)

    return nation_df, supplier_df, nations_z3, suppliers_z3, orders_z3, lineitem_raw_by_okey, lineitem_z3_by_okey

def generate_phi_for_targeted_pairs(targeted_pairs, nation_df, supplier_df, nations_z3, suppliers_z3, orders_z3, lineitem_raw_by_okey, lineitem_z3_by_okey):
    phi_disjuncts = defaultdict(list)
    targeted_pairs_set = set(targeted_pairs)

    japan_nation_row = nation_df[nation_df['n_name'] == 'JAPAN']
    if japan_nation_row.empty:
        return {}
        
    japan_n_key = int(japan_nation_row.iloc[0]['n_nationkey'])
    japan_nation_z3 = nations_z3[japan_n_key]

    for o_key, l1_group_raw in lineitem_raw_by_okey.items():
        relevant_s_keys = {s_key for s_key, o_key_target in targeted_pairs_set if o_key_target == o_key}
        if not relevant_s_keys:
            continue

        l1_group_z3 = lineitem_z3_by_okey[o_key]
        order_z3 = orders_z3.get(o_key)
        if not order_z3: continue

        cond_order_status = (order_z3['o_orderstatus'] == StringVal('F'))

        for i, l1_raw in enumerate(l1_group_raw):
            l1_z3 = l1_group_z3[i]
            cond_l1_late = BoolVal(l1_raw['l_receiptdate'] > l1_raw['l_commitdate'])

            other_raw = [l for j, l in enumerate(l1_group_raw) if i != j]
            other_z3 = [l for j, l in enumerate(l1_group_z3) if i != j]

            l2_disjuncts = [l2_z3['l_suppkey'] != l1_z3['l_suppkey'] for l2_z3 in other_z3]
            cond_exists_l2 = Or(l2_disjuncts) if l2_disjuncts else BoolVal(False)

            l3_disjuncts = [
                And(l3_z3['l_suppkey'] != l1_z3['l_suppkey'], BoolVal(l3_raw['l_receiptdate'] > l3_raw['l_commitdate']))
                for l3_raw, l3_z3 in zip(other_raw, other_z3)
            ]
            cond_not_exists_l3 = Not(Or(l3_disjuncts)) if l3_disjuncts else BoolVal(True)

            phi_base_for_l1 = And(cond_order_status, cond_l1_late, cond_exists_l2, cond_not_exists_l3)

            for s_key in relevant_s_keys:
                s_z3 = suppliers_z3[s_key]
                cond_join_s_l1 = (s_z3['s_suppkey'] == l1_z3['l_suppkey'])
                cond_nation = (s_z3['s_nationkey'] == japan_nation_z3['n_nationkey'])
                phi_disjuncts[(s_key, o_key)].append(And(cond_join_s_l1, cond_nation, phi_base_for_l1))

    phi_map = {key: simplify(Or(disjuncts)) for key, disjuncts in phi_disjuncts.items()}
    return phi_map

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/user/Desktop/SQL/tpch-sf0.01-10nulls/")
    parser.add_argument("--cache_path", type=str, default="phi_Q1_cache.json")
    parser.add_argument("--pairs_file", type=str, default="targeted_pairs.json")
    parser.add_argument("--mode", type=str, choices=["certainty", "analysis"], default="analysis")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    DATA_PATH = args.data_path
    PHI_CACHE_PATH = args.cache_path
    PAIRS_INPUT_FILE = args.pairs_file

    k_configs_to_test = [
        {'mode': 'absolute', 'range': range(1, 121)},
        {'mode': 'relative', 'range': range(1, 121)}
    ]
    dist_configs_to_test = [
        {"name": "uniform"},
        {"name": "zipfian", "params": {"s": 0.5}},
        {"name": "normal", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "mu": 50.0, "sigma": 15.0}},
        {"name": "exponential", "params": {"lower_bound": 0.0, "upper_bound": 100.0, "scale": 20.0}}
    ]
    shuffle_options = [True, False]

    default_targeted_pairs = [
        (54, 1088), (43, 1799), (43, 3075), (54, 3908), (81, 7457), (43, 8352),
        (81, 9537), (81, 9575), (96, 9889), (43, 11078), (54, 14272), (81, 21537),
        (81, 22500), (81, 25479), (81, 25799), (43, 27141), (96, 27556),
        (81, 29312), (96, 29921), (43, 30757), (43, 30821), (43, 31077),
        (43, 34432), (43, 36132), (43, 37188), (43, 37444), (81, 39397),
        (81, 42722), (54, 43585), (43, 44224), (96, 44325), (43, 44711),
        (81, 45158), (81, 47012), (81, 47621), (54, 48547), (81, 51810),
        (54, 52258), (96, 52325), (81, 53988), (96, 54119), (96, 56608),
        (54, 56769), (43, 57349), (81, 57414)
    ]

    if os.path.exists(PAIRS_INPUT_FILE):
        with open(PAIRS_INPUT_FILE, 'r') as f:
            targeted_pairs = [tuple(p) for p in json.load(f)]
    else:
        targeted_pairs = default_targeted_pairs
        with open(PAIRS_INPUT_FILE, 'w') as f:
            json.dump(targeted_pairs, f)

    phi_map = None
    if phi_map is None:
        symbolized_data = load_and_symbolize_tpch_data_optimized(DATA_PATH)
        phi_map = generate_phi_for_targeted_pairs(targeted_pairs, *symbolized_data)
        save_phi_map(phi_map, PHI_CACHE_PATH)

    if args.mode == 'certainty':
        check_certainty(phi_map, targeted_pairs)

    elif args.mode == 'analysis':
        if targeted_pairs:
            pair_to_analyze = targeted_pairs[-1]
            tpch_instance_skok_pairs = [(DATA_PATH, pair_to_analyze)]

            for k_config in k_configs_to_test:
                for dist_config in dist_configs_to_test:
                    for shuffle in shuffle_options:
                        if dist_config['name'] == 'uniform' and shuffle: continue
                        shuffle_str = "shuffled" if shuffle else "ordered"
                        output_filename = f"results/TPCH_Q1_Analysis_{pair_to_analyze[0]}_{pair_to_analyze[1]}_{dist_config['name']}_{k_config['mode']}_{shuffle_str}.csv"
                        
                        run_mu_plot_multiple(
                            tpch_instance_skok_pairs,
                            phi_map,
                            dist_config,
                            k_config,
                            shuffle,
                            output_csv_path=output_filename
                        )