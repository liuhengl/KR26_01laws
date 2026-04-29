import json
import subprocess
import os
import sys
import argparse
import time
import re
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEARCH_ROOT = os.path.join(REPO_ROOT, "data/gql")
DATA_ROOT = os.path.join(REPO_ROOT, "snb_experiments")
PYTHON_EXECUTABLE = "python"
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "verification_results_gql")

QUERY_CONFIG = {
    "Q1": {"script": os.path.join(REPO_ROOT, "scripts", "gql", "GQL_Q1.py"), "input_arg": "--pairs_file", "cache_arg": "--cache_path"},
    "Q2": {"script": os.path.join(REPO_ROOT, "scripts", "gql", "GQL_Q2.py"), "input_arg": "--keys_file", "cache_arg": "--cache_path"},
    "Q3": {"script": os.path.join(REPO_ROOT, "scripts", "gql", "GQL_Q5.py"), "input_arg": "--keys_file", "cache_arg": "--cache_path"},
    "Q4": {"script": os.path.join(REPO_ROOT, "scripts", "gql", "GQL_Q4.py"), "input_arg": "--keys_file", "cache_arg": "--cache_path"},
    "Q5": {"script": os.path.join(REPO_ROOT, "scripts", "gql", "GQL_Q5.py"), "input_arg": "--keys_file", "cache_arg": "--cache_path"},
}

def sanitize_and_save_json(source_path, target_path):
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        final_data = data
        if isinstance(data, dict):
            if "results" in data:
                final_data = data["results"]
            elif "answers" in data:
                final_data = data["answers"]

        with open(target_path, 'w') as f:
            json.dump(final_data, f)
        
        return True
    except Exception:
        return False

def parse_filename_info(filename, query_name):
    suffix = f"_{query_name}.json"
    if not filename.endswith(suffix):
        return None, None

    folder_name = filename[:-len(suffix)]
    rate_match = re.match(r'^(\d+)_', folder_name)
    rate_id = rate_match.group(1) if rate_match else "unknown"

    return folder_name, rate_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, choices=["Q1", "Q2", "Q3", "Q4", "Q5"])
    parser.add_argument("--mode", type=str, default="certainty", choices=["certainty", "analysis"])
    args = parser.parse_args()

    target_q = args.query
    
    if target_q in QUERY_CONFIG:
        config = QUERY_CONFIG[target_q]
        if not os.path.exists(config["script"]) and os.path.exists(os.path.join(REPO_ROOT, "scripts", "gql", f"{target_q}.py")):
             config["script"] = os.path.join(REPO_ROOT, "scripts", "gql", f"{target_q}.py")
    else:
        sys.exit(1)

    output_subdir = os.path.join(BASE_OUTPUT_DIR, target_q)
    os.makedirs(output_subdir, exist_ok=True)
    combined_log_path = os.path.abspath(os.path.join(output_subdir, f"{target_q}_verification_log.txt"))
    
    with open(combined_log_path, 'w') as f:
        f.write(f"Verification Log for {target_q}\nDate: {time.ctime()}\n\n")

    search_pattern = os.path.join(SEARCH_ROOT, f"*{target_q}.json")
    json_files = glob.glob(search_pattern)
    json_files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(0)) if re.search(r'(\d+)', os.path.basename(f)) else 0)

    for json_path in json_files:
        filename = os.path.basename(json_path)
        data_folder_name, rate_id = parse_filename_info(filename, target_q)
        
        if not data_folder_name:
            continue

        data_path = os.path.abspath(os.path.join(DATA_ROOT, data_folder_name, "dynamic"))
        
        if not os.path.exists(data_path):
            continue

        temp_input_path = os.path.join(output_subdir, f"temp_input_{data_folder_name}_{target_q}.json")
        cache_file_path = os.path.join(output_subdir, f"cache_{data_folder_name}_{target_q}.json")

        if not sanitize_and_save_json(json_path, temp_input_path):
            continue

        cmd = [
            PYTHON_EXECUTABLE, config["script"],
            "--data_path", data_path,
            config["input_arg"], temp_input_path,
            config["cache_arg"], cache_file_path,
            "--mode", args.mode
        ]

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time

            with open(combined_log_path, 'a') as log_file:
                log_file.write(f"FILE: {filename} | RATE: {rate_id}\n")
                log_file.write(f"Cmd: {' '.join(cmd)}\n")
                log_file.write(result.stdout.strip() + "\n")
                if result.stderr:
                    log_file.write(f"[STDERR]\n{result.stderr.strip()}\n")
                log_file.write(f"[Time: {duration:.2f}s]\n\n")

        except Exception as e:
            with open(combined_log_path, 'a') as log_file:
                 log_file.write(f"Error on {filename}: {e}\n\n")

if __name__ == "__main__":
    main()