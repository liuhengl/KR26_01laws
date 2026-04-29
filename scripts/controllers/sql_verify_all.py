import csv
import json
import subprocess
import os
import sys
import argparse
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_CSV = os.path.join(REPO_ROOT, "data", "sql", "all_query_results.csv")
DATA_ROOT = os.path.join(REPO_ROOT, "tpch_experiments")
PYTHON_EXECUTABLE = "python"
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "verification_results")

QUERY_CONFIG = {
    "Q1": {
        "script": os.path.join(REPO_ROOT, "scripts", "sql", "Q1.py"),
        "input_arg": "--pairs_file",
        "cache_arg": "--cache_path",
    },
    "Q2": {
        "script": os.path.join(REPO_ROOT, "scripts", "sql", "Q2.py"),
        "input_arg": "--keys_file",
        "cache_arg": "--cache_path",
    },
    "Q3": {
        "script": os.path.join(REPO_ROOT, "scripts", "sql", "Q3.py"),
        "input_arg": "--keys_file",
        "cache_arg": "--cache_path",
    },
    "Q4": {
        "script": os.path.join(REPO_ROOT, "scripts", "sql", "Q4.py"),
        "input_arg": "--keys_file",
        "cache_arg": "--cache_path",
    }
}

def parse_csv_results(query_name, raw_string):
    if not raw_string or raw_string.strip() == "":
        return []

    clean_string = raw_string.replace('"', '').strip()
    records = clean_string.split('|')
    parsed_data = []

    for rec in records:
        if not rec: continue
        try:
            if query_name == "Q1":
                if ',' in rec:
                    k1, k2 = rec.split(',')
                    parsed_data.append([int(k1), int(k2)])
            elif query_name == "Q2":
                if ',' in rec:
                    k1, _ = rec.split(',')
                    parsed_data.append(int(k1))
                else:
                    parsed_data.append(int(rec))
            elif query_name in ["Q3", "Q4"]:
                parsed_data.append(int(rec))
        except ValueError:
            continue
    return parsed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, choices=["Q1", "Q2", "Q3", "Q4"])
    parser.add_argument("--mode", type=str, default="certainty", choices=["certainty", "analysis"])
    args = parser.parse_args()

    target_q = args.query
    config = QUERY_CONFIG[target_q]

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        sys.exit(1)

    output_subdir = os.path.join(BASE_OUTPUT_DIR, target_q)
    os.makedirs(output_subdir, exist_ok=True)
    combined_log_path = os.path.abspath(os.path.join(output_subdir, f"{target_q}_combined_log.txt"))

    with open(combined_log_path, 'w') as f:
        f.write(f"Verification Log for {target_q}\nDate: {time.ctime()}\n\n")

    with open(INPUT_CSV, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0

        for row in reader:
            if row['query_name'] != target_q:
                continue

            instance = row['instance']
            null_rate = row['null_rate']
            raw_results = row['detailed_results']

            data_path = os.path.abspath(f"{DATA_ROOT}/instance_{instance}/incomplete_data/rate_{null_rate}/")
            if not os.path.exists(data_path):
                continue

            parsed_data = parse_csv_results(target_q, raw_results)
            if not parsed_data:
                continue

            input_file_path = os.path.abspath(os.path.join(output_subdir, f"temp_inputs_i{instance}_r{null_rate}.json"))
            cache_file_path = os.path.abspath(os.path.join(output_subdir, f"cache_i{instance}_r{null_rate}.json"))

            with open(input_file_path, 'w') as jf:
                json.dump(parsed_data, jf)

            cmd = [
                PYTHON_EXECUTABLE, config["script"],
                "--data_path", data_path,
                config["input_arg"], input_file_path,
                config["cache_arg"], cache_file_path,
                "--mode", args.mode
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)

                with open(combined_log_path, 'a') as log_file:
                    log_file.write(f"INSTANCE {instance} | RATE {null_rate}\n")
                    log_file.write(f"Cmd: {' '.join(cmd)}\n")
                    log_file.write(result.stdout.strip() + "\n")
                    if result.stderr:
                        log_file.write(f"[STDERR]\n{result.stderr.strip()}\n")
                    log_file.write("\n")

            except Exception as e:
                with open(combined_log_path, 'a') as log_file:
                    log_file.write(f"INSTANCE {instance} | RATE {null_rate}\nError: {str(e)}\n\n")

            count += 1

if __name__ == "__main__":
    main()