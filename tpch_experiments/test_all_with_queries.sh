#!/bin/bash
set -e

# --- Configuration ---
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/tpch_experiments"
OUTPUT_CSV="${REPO_ROOT}/data/sql/all_query_results.csv"

# Postgres Connection Details
PG_USER="postgres"
PG_HOST="localhost"
PG_PORT="5432"
export PGPASSWORD="123456"

# --- Initialize Output CSV ---
# Write the header row
echo "instance,null_rate,query_name,row_count,detailed_results" > "$OUTPUT_CSV"

# --- SQL Queries Definition ---
Q1_SQL="SELECT s_suppkey, o_orderkey FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS ( SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey ) AND NOT EXISTS ( SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate ) AND s_nationkey = n_nationkey AND n_name = 'JAPAN';"

Q2_SQL="SELECT c_custkey, c_nationkey FROM customer WHERE c_nationkey IN (13,31,23,29,30,18,17) AND c_acctbal > ( SELECT avg(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND c_nationkey IN (13,31,23,29,30,18,17) ) AND NOT EXISTS ( SELECT * FROM orders WHERE o_custkey = c_custkey );"

Q3_SQL="SELECT o_orderkey FROM orders WHERE NOT EXISTS ( SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_suppkey <> 77 );"

Q4_SQL="SELECT o_orderkey FROM orders WHERE NOT EXISTS ( SELECT * FROM lineitem, part, supplier, nation WHERE l_orderkey = o_orderkey AND l_partkey = p_partkey AND l_suppkey = s_suppkey AND p_name LIKE '%purple%' AND s_nationkey = n_nationkey AND n_name = 'CANADA' );"

# --- Function: Create SQL Load Template ---
create_sql_template() {
    cat <<EOF
-- TPC-H Load Script for Custom Null Handling
-- Dynamic Path: $1

-- 1. Cleanup
DROP TABLE IF EXISTS lineitem CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS nation CASCADE;
DROP TABLE IF EXISTS region CASCADE;

-- 2. Create Schema (Standard Types)
CREATE TABLE PART ( P_PARTKEY integer, P_NAME varchar(55), P_MFGR char(25), P_BRAND char(10), P_TYPE varchar(25), P_SIZE integer, P_CONTAINER char(10), P_RETAILPRICE numeric, P_COMMENT varchar(23) );
CREATE TABLE SUPPLIER ( S_SUPPKEY integer, S_NAME char(25), S_ADDRESS varchar(40), S_NATIONKEY integer, S_PHONE char(15), S_ACCTBAL numeric, S_COMMENT varchar(101) );
CREATE TABLE PARTSUPP ( PS_PARTKEY integer, PS_SUPPKEY integer, PS_AVAILQTY integer, PS_SUPPLYCOST numeric, PS_COMMENT varchar(199) );
CREATE TABLE CUSTOMER ( C_CUSTKEY integer, C_NAME varchar(25), C_ADDRESS varchar(40), C_NATIONKEY integer, C_PHONE char(15), C_ACCTBAL numeric, C_MKTSEGMENT char(10), C_COMMENT varchar(117) );
CREATE TABLE ORDERS ( O_ORDERKEY integer, O_CUSTKEY integer, O_ORDERSTATUS char(1), O_TOTALPRICE numeric, O_ORDERDATE date, O_ORDERPRIORITY char(15), O_CLERK char(15), O_SHIPPRIORITY integer, O_COMMENT varchar(79) );
CREATE TABLE LINEITEM ( L_ORDERKEY integer, L_PARTKEY integer, L_SUPPKEY integer, L_LINENUMBER integer, L_QUANTITY numeric, L_EXTENDEDPRICE numeric, L_DISCOUNT numeric, L_TAX numeric, L_RETURNFLAG char(1), L_LINESTATUS char(1), L_SHIPDATE date, L_COMMITDATE date, L_RECEIPTDATE date, L_SHIPINSTRUCT char(25), L_SHIPMODE char(10), L_COMMENT varchar(44) );
CREATE TABLE NATION ( N_NATIONKEY integer, N_NAME char(25), N_REGIONKEY integer, N_COMMENT varchar(152) );
CREATE TABLE REGION ( R_REGIONKEY integer, R_NAME char(25), R_COMMENT varchar(152) );

-- 3. Apply Custom Types (ids4nulls)
CREATE EXTENSION IF NOT EXISTS ids4nulls;

ALTER TABLE PART
  ALTER P_PARTKEY       SET DATA TYPE intt,
  ALTER P_NAME          SET DATA TYPE textt,
  ALTER P_MFGR          SET DATA TYPE textt,
  ALTER P_BRAND         SET DATA TYPE textt,
  ALTER P_TYPE          SET DATA TYPE textt,
  ALTER P_SIZE          SET DATA TYPE intt,
  ALTER P_CONTAINER     SET DATA TYPE textt,
  ALTER P_RETAILPRICE   SET DATA TYPE numericc,
  ALTER P_COMMENT       SET DATA TYPE textt;

ALTER TABLE SUPPLIER
  ALTER S_SUPPKEY       SET DATA TYPE intt,
  ALTER S_NAME          SET DATA TYPE textt,
  ALTER S_ADDRESS       SET DATA TYPE textt,
  ALTER S_NATIONKEY     SET DATA TYPE intt,
  ALTER S_PHONE         SET DATA TYPE textt,
  ALTER S_ACCTBAL       SET DATA TYPE numericc,
  ALTER S_COMMENT       SET DATA TYPE textt;

ALTER TABLE PARTSUPP
  ALTER PS_PARTKEY      SET DATA TYPE intt,
  ALTER PS_SUPPKEY      SET DATA TYPE intt,
  ALTER PS_AVAILQTY     SET DATA TYPE intt,
  ALTER PS_SUPPLYCOST   SET DATA TYPE numericc,
  ALTER PS_COMMENT      SET DATA TYPE textt;

ALTER TABLE CUSTOMER
  ALTER C_CUSTKEY       SET DATA TYPE intt,
  ALTER C_NAME          SET DATA TYPE textt,
  ALTER C_ADDRESS       SET DATA TYPE textt,
  ALTER C_NATIONKEY     SET DATA TYPE intt,
  ALTER C_PHONE         SET DATA TYPE textt,
  ALTER C_ACCTBAL       SET DATA TYPE numericc,
  ALTER C_MKTSEGMENT    SET DATA TYPE textt,
  ALTER C_COMMENT       SET DATA TYPE textt;

ALTER TABLE ORDERS
  ALTER O_ORDERKEY      SET DATA TYPE intt,
  ALTER O_CUSTKEY       SET DATA TYPE intt,
  ALTER O_ORDERSTATUS   SET DATA TYPE textt,
  ALTER O_TOTALPRICE    SET DATA TYPE numericc,
  ALTER O_ORDERPRIORITY SET DATA TYPE textt,
  ALTER O_CLERK         SET DATA TYPE textt,
  ALTER O_SHIPPRIORITY  SET DATA TYPE intt,
  ALTER O_COMMENT       SET DATA TYPE textt;

ALTER TABLE LINEITEM
  ALTER L_ORDERKEY      SET DATA TYPE intt,
  ALTER L_PARTKEY       SET DATA TYPE intt,
  ALTER L_SUPPKEY       SET DATA TYPE intt,
  ALTER L_LINENUMBER    SET DATA TYPE intt,
  ALTER L_QUANTITY      SET DATA TYPE numericc,
  ALTER L_EXTENDEDPRICE SET DATA TYPE numericc,
  ALTER L_DISCOUNT      SET DATA TYPE numericc,
  ALTER L_TAX           SET DATA TYPE numericc,
  ALTER L_RETURNFLAG    SET DATA TYPE textt,
  ALTER L_LINESTATUS    SET DATA TYPE textt,
  ALTER L_SHIPINSTRUCT  SET DATA TYPE textt,
  ALTER L_SHIPMODE      SET DATA TYPE textt,
  ALTER L_COMMENT       SET DATA TYPE textt;

ALTER TABLE NATION
  ALTER N_NATIONKEY SET DATA TYPE intt,
  ALTER N_NAME      SET DATA TYPE textt,
  ALTER N_REGIONKEY SET DATA TYPE intt,
  ALTER N_COMMENT   SET DATA TYPE textt;

ALTER TABLE REGION
  ALTER R_REGIONKEY SET DATA TYPE intt,
  ALTER R_NAME      SET DATA TYPE textt,
  ALTER R_COMMENT   SET DATA TYPE textt;

-- 4. Keys
ALTER TABLE region ADD PRIMARY KEY (r_regionkey);
ALTER TABLE nation ADD PRIMARY KEY (n_nationkey);
ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey);
ALTER TABLE part ADD PRIMARY KEY (p_partkey);
ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey);
ALTER TABLE customer ADD PRIMARY KEY (c_custkey);
ALTER TABLE orders ADD PRIMARY KEY (o_orderkey);
ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber);

-- 5. Load Data
\copy region FROM '$1/region.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy nation FROM '$1/nation.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy part FROM '$1/part.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy supplier FROM '$1/supplier.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy partsupp FROM '$1/partsupp.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy customer FROM '$1/customer.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy orders FROM '$1/orders.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
\copy lineitem FROM '$1/lineitem.csv' WITH (FORMAT csv, DELIMITER ',', HEADER true);
EOF
}

# --- Function: Run and Log Query ---
run_and_log() {
    local q_name="$1"
    local q_sql="$2"
    local db="$3"
    local inst="$4"
    local rate="$5"

    # Execute SQL, get raw output
    # Notes: -w avoids password prompt (uses PGPASSWORD env var)
    raw_output=$(psql -h "$PG_HOST" -U "$PG_USER" -d "$db" -w -t -A -F"," -c "$q_sql" 2>/dev/null)

    if [ -z "$raw_output" ]; then
        row_count=0
        formatted_results=""
    else
        row_count=$(echo "$raw_output" | wc -l)
        # Format results: newlines -> pipes, escape double quotes
        formatted_results=$(echo "$raw_output" | tr '\n' '|' | sed 's/|$//' | sed 's/"/""/g') 
    fi

    # Append to Main CSV
    echo "$inst,$rate,$q_name,$row_count,\"$formatted_results\"" >> "$OUTPUT_CSV"
    echo "      -> $q_name: $row_count rows."
}

# --- Main Execution Loop ---

echo "Starting Global Load and Benchmark..."

# Loop through Instances 1 to 5
for inst in {1..5}; do
    # Loop through Rates
    for rate in 0.01 0.02 0.03 0.04 0.05; do
        
        # 1. Setup Environment Variables
        CURRENT_DATA_DIR="${DATA_ROOT}/instance_${inst}/incomplete_data/rate_${rate}"
        RATE_CLEAN=$(echo $rate | sed 's/0.//') 
        DB_NAME="tpch_i${inst}_r${RATE_CLEAN}"
        
        echo "---------------------------------------------------"
        echo "Processing: Instance $inst | Rate $rate"
        echo "Database:   $DB_NAME"
        
        if [ ! -d "$CURRENT_DATA_DIR" ]; then
            echo "Skipping - Directory not found."
            continue
        fi

	# 2. LOAD PHASE: Create DB and Schema
        # The -w flag tells Postgres to not prompt, but fail if no password provided
        echo "  [Phase 1] Loading Data..."
        dropdb -h $PG_HOST -U $PG_USER -w --if-exists "$DB_NAME"
        createdb -h $PG_HOST -U $PG_USER -w "$DB_NAME"

        ABS_PATH=$(realpath "$CURRENT_DATA_DIR")
        TEMP_SQL_FILE="temp_load_${inst}_${RATE_CLEAN}.sql"
        
        create_sql_template "$ABS_PATH" > "$TEMP_SQL_FILE"
        psql -h $PG_HOST -U $PG_USER -d "$DB_NAME" -w -f "$TEMP_SQL_FILE" > /dev/null
        rm "$TEMP_SQL_FILE"

        # 3. TEST PHASE: Run Queries
        echo "  [Phase 2] Running Queries..."
        run_and_log "Q1" "$Q1_SQL" "$DB_NAME" "$inst" "$rate"
        run_and_log "Q2" "$Q2_SQL" "$DB_NAME" "$inst" "$rate"
        run_and_log "Q3" "$Q3_SQL" "$DB_NAME" "$inst" "$rate"
        run_and_log "Q4" "$Q4_SQL" "$DB_NAME" "$inst" "$rate"

    done
done

echo "---------------------------------------------------"
echo "All tasks finished."
echo "Results saved to: $OUTPUT_CSV"
