"""Synthetic data generator for LedgerLock 3-way financial reconciliation.

Generates 70 realistic Indian payment aggregator transactions across:
- settlement_report.csv
- internal_ledger.csv
- bank_statement.csv
- ground_truth.json (hidden validation ground truth)
"""

from __future__ import annotations

import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from faker import Faker

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard Indian TDS sections for payment aggregators
TDS_RULES = {
    "194-O": {"rate": 0.01, "desc": "Payment of certain sums by e-commerce operator to e-commerce participants (1%)"},
    "194H": {"rate": 0.05, "desc": "Commission or brokerage (5%)"},
    "194C": {"rate": 0.02, "desc": "Payment to contractors - company/firm (2%)"},
    "194J": {"rate": 0.10, "desc": "Fees for professional services (10%)"},
}

INDIAN_COUNTERPARTIES = [
    "Swiggy Delivery Ops",
    "Zomato Merchant Services",
    "Flipkart Internet Pvt Ltd",
    "Amazon Seller Services IN",
    "Razorpay Software Pvt Ltd",
    "PhonePe Merchant Services",
    "Paytm Payment Gateway",
    "Blinkit Commerce Pvt Ltd",
    "Zepto Express Logistics",
    "Nykaa Retail Logistics",
    "Myntra Designs Pvt Ltd",
    "Meesho Technologies",
    "BigBasket Supermarket Groceries",
    "Tata CLiQ Digital",
    "Dunzo Digital Services",
]


def generate_dataset(num_records: int = 70) -> Dict[str, Any]:
    """Generate 3-way reconciliation datasets with strictly controlled failure distributions."""
    # Category allocation summing to 70
    counts = {
        "clean_exact": 38,        # ~55%
        "date_offset": 7,         # 10%
        "reference_typo": 7,      # 10%
        "split_settlement": 6,    # ~8% (3 pairs)
        "tds_misapplied": 5,      # ~7%
        "duplicate": 4,           # ~5% (2 pairs)
        "missing_record": 3,      # ~5%
    }

    assert sum(counts.values()) == num_records, f"Counts sum to {sum(counts.values())}, expected {num_records}"

    settlement_records: List[Dict[str, Any]] = []
    ledger_records: List[Dict[str, Any]] = []
    bank_records: List[Dict[str, Any]] = []
    ground_truth: Dict[str, Dict[str, Any]] = {}

    base_date = datetime(2026, 8, 1, 10, 0, 0)
    tx_counter = 1000

    def get_tx_id():
        nonlocal tx_counter
        tx_counter += 1
        return f"TXN_{tx_counter}"

    def get_ref_id(counter: int):
        return f"PAY-IN-{base_date.strftime('%Y%m%d')}-{counter:04d}"

    ref_counter = 1

    # 1. Clean Exact Matches (Tier 1)
    for _ in range(counts["clean_exact"]):
        tx_id = get_tx_id()
        ref_id = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(5000.0, 450000.0), 2)
        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = random.choice(list(TDS_RULES.keys()))
        rate = TDS_RULES[section]["rate"]
        tds_amount = round(amount * rate, 2)

        record = {
            "transaction_id": tx_id,
            "reference_id": ref_id,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        }
        settlement_records.append(dict(record))
        ledger_records.append(dict(record))
        bank_records.append(dict(record))

        ground_truth[tx_id] = {
            "category": "clean_exact",
            "expected_resolution": "matched",
            "expected_tier": "tier1",
            "reference_id": ref_id,
            "amount": amount,
            "details": "Clean exact match across settlement, ledger, and bank.",
        }

    # 2. Date Offset (1-3 days) (Tier 1 tolerance window)
    for _ in range(counts["date_offset"]):
        tx_id = get_tx_id()
        ref_id = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(10000.0, 250000.0), 2)
        base_dt = base_date + timedelta(days=random.randint(0, 12))
        ledger_date = base_dt.strftime("%Y-%m-%d")
        settlement_date = (base_dt + timedelta(days=random.randint(1, 2))).strftime("%Y-%m-%d")
        bank_date = (base_dt + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = random.choice(list(TDS_RULES.keys()))
        rate = TDS_RULES[section]["rate"]
        tds_amount = round(amount * rate, 2)

        common = {
            "transaction_id": tx_id,
            "reference_id": ref_id,
            "amount": amount,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        }
        ledger_records.append({**common, "date": ledger_date})
        settlement_records.append({**common, "date": settlement_date})
        bank_records.append({**common, "date": bank_date})

        ground_truth[tx_id] = {
            "category": "date_offset",
            "expected_resolution": "matched",
            "expected_tier": "tier1",
            "reference_id": ref_id,
            "amount": amount,
            "details": f"Date offset within 1-3 days tolerance (Ledger: {ledger_date}, Settlement: {settlement_date}, Bank: {bank_date}).",
        }

    # 3. Reference ID typo / format mismatch (Tier 2)
    for _ in range(counts["reference_typo"]):
        tx_id = get_tx_id()
        base_ref = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(15000.0, 300000.0), 2)
        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = random.choice(list(TDS_RULES.keys()))
        rate = TDS_RULES[section]["rate"]
        tds_amount = round(amount * rate, 2)

        # Injected variation in reference ID (missing dash, prefix difference, typo)
        typo_type = random.choice(["prefix", "typo_digit", "case_dash"])
        if typo_type == "prefix":
            settlement_ref = base_ref.replace("PAY-IN", "PGW-IN")
            bank_ref = base_ref
        elif typo_type == "typo_digit":
            # swap two characters
            settlement_ref = base_ref[:-2] + base_ref[-1] + base_ref[-2]
            bank_ref = base_ref
        else:
            settlement_ref = base_ref.replace("-", "_").lower()
            bank_ref = base_ref

        ledger_records.append({
            "transaction_id": tx_id,
            "reference_id": base_ref,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        })
        settlement_records.append({
            "transaction_id": tx_id,
            "reference_id": settlement_ref,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        })
        bank_records.append({
            "transaction_id": tx_id,
            "reference_id": bank_ref,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        })

        ground_truth[tx_id] = {
            "category": "reference_typo",
            "expected_resolution": "matched",
            "expected_tier": "tier2",
            "reference_id": base_ref,
            "amount": amount,
            "details": f"Reference ID format mismatch: Ledger ({base_ref}) vs Settlement ({settlement_ref}).",
        }

    # 4. Split Settlement (Tier 2) - 1 Ledger entry <=> 2 Settlement rows
    # 6 records = 3 split master transactions (each master = 1 ledger, 2 settlements, 1 or 2 bank credits)
    for _ in range(counts["split_settlement"] // 2):
        tx_id_1 = get_tx_id()
        tx_id_2 = get_tx_id()
        base_ref = get_ref_id(ref_counter)
        ref_counter += 1

        total_amount = round(random.uniform(50000.0, 200000.0), 2)
        split_1 = round(total_amount * random.uniform(0.4, 0.6), 2)
        split_2 = round(total_amount - split_1, 2)

        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = "194-O"
        rate = TDS_RULES[section]["rate"]

        # 1 Master ledger entry
        master_tx_id = f"{tx_id_1}_M"
        ledger_records.append({
            "transaction_id": master_tx_id,
            "reference_id": base_ref,
            "amount": total_amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": round(total_amount * rate, 2),
            "tds_rate": rate,
        })

        # 2 Settlement rows
        settlement_records.append({
            "transaction_id": tx_id_1,
            "reference_id": f"{base_ref}-PART1",
            "amount": split_1,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": round(split_1 * rate, 2),
            "tds_rate": rate,
        })
        settlement_records.append({
            "transaction_id": tx_id_2,
            "reference_id": f"{base_ref}-PART2",
            "amount": split_2,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": round(split_2 * rate, 2),
            "tds_rate": rate,
        })

        # Bank credit matches total amount or two parts
        bank_records.append({
            "transaction_id": master_tx_id,
            "reference_id": base_ref,
            "amount": total_amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": round(total_amount * rate, 2),
            "tds_rate": rate,
        })

        ground_truth[tx_id_1] = {
            "category": "split_settlement",
            "expected_resolution": "matched",
            "expected_tier": "tier2",
            "reference_id": base_ref,
            "amount": split_1,
            "master_id": master_tx_id,
            "details": f"Split settlement part 1 ({split_1}) for ledger {master_tx_id} ({total_amount}).",
        }
        ground_truth[tx_id_2] = {
            "category": "split_settlement",
            "expected_resolution": "matched",
            "expected_tier": "tier2",
            "reference_id": base_ref,
            "amount": split_2,
            "master_id": master_tx_id,
            "details": f"Split settlement part 2 ({split_2}) for ledger {master_tx_id} ({total_amount}).",
        }

    # 5. TDS Rate Misapplied vs Correct Section (Tier 3)
    for _ in range(counts["tds_misapplied"]):
        tx_id = get_tx_id()
        ref_id = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(20000.0, 180000.0), 2)
        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = random.choice(["194-O", "194H", "194C"])
        correct_rate = TDS_RULES[section]["rate"]

        # Injected wrong rate (e.g. 5% instead of 1%, or 1% instead of 5%)
        wrong_rate = 0.05 if correct_rate == 0.01 else 0.01
        actual_tds_amount = round(amount * wrong_rate, 2)

        record = {
            "transaction_id": tx_id,
            "reference_id": ref_id,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": actual_tds_amount,
            "tds_rate": wrong_rate,
        }
        settlement_records.append(dict(record))
        ledger_records.append(dict(record))
        bank_records.append(dict(record))

        ground_truth[tx_id] = {
            "category": "tds_misapplied",
            "expected_resolution": "matched",
            "expected_tier": "tier3",
            "reference_id": ref_id,
            "amount": amount,
            "correct_rate": correct_rate,
            "applied_rate": wrong_rate,
            "tds_section": section,
            "details": f"TDS rate misapplied: {wrong_rate*100}% applied under {section}, statutory rate is {correct_rate*100}%.",
        }

    # 6. Duplicate Entry (Tier 2) - 4 records (2 unique transactions duplicated)
    for _ in range(counts["duplicate"] // 2):
        tx_id_a = get_tx_id()
        tx_id_b = f"{tx_id_a}_DUP"
        ref_id = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(8000.0, 95000.0), 2)
        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = "194-O"
        rate = TDS_RULES[section]["rate"]
        tds_amount = round(amount * rate, 2)

        record_a = {
            "transaction_id": tx_id_a,
            "reference_id": ref_id,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        }
        record_b = {
            "transaction_id": tx_id_b,
            "reference_id": ref_id,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        }

        settlement_records.extend([record_a, record_b])
        ledger_records.append(record_a)
        bank_records.append(record_a)

        ground_truth[tx_id_a] = {
            "category": "duplicate",
            "expected_resolution": "matched",
            "expected_tier": "tier2",
            "reference_id": ref_id,
            "amount": amount,
            "details": "Original transaction resolved from duplicate settlement stream.",
        }
        ground_truth[tx_id_b] = {
            "category": "duplicate",
            "expected_resolution": "matched",
            "expected_tier": "tier2",
            "reference_id": ref_id,
            "amount": amount,
            "details": "Duplicate settlement entry de-duplicated and linked to primary record.",
        }

    # 7. Genuinely Missing Record in one source (True Exception - MUST NOT MATCH)
    for _ in range(counts["missing_record"]):
        tx_id = get_tx_id()
        ref_id = get_ref_id(ref_counter)
        ref_counter += 1
        amount = round(random.uniform(25000.0, 500000.0), 2)
        tx_date = (base_date + timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d")
        counterparty = random.choice(INDIAN_COUNTERPARTIES)
        section = "194-O"
        rate = TDS_RULES[section]["rate"]
        tds_amount = round(amount * rate, 2)

        # Present in internal ledger and settlement, but MISSING in bank statement
        record = {
            "transaction_id": tx_id,
            "reference_id": ref_id,
            "amount": amount,
            "date": tx_date,
            "counterparty": counterparty,
            "tds_section": section,
            "tds_amount": tds_amount,
            "tds_rate": rate,
        }
        ledger_records.append(record)
        settlement_records.append(record)
        # Omit from bank_records!

        ground_truth[tx_id] = {
            "category": "missing_record",
            "expected_resolution": "exception",
            "expected_tier": "unresolved",
            "reference_id": ref_id,
            "amount": amount,
            "details": "Genuinely missing record in bank statement — true reconciliation exception.",
        }

    return {
        "settlement": pd.DataFrame(settlement_records),
        "ledger": pd.DataFrame(ledger_records),
        "bank": pd.DataFrame(bank_records),
        "ground_truth": ground_truth,
        "counts": counts,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = generate_dataset(70)

    settlement_file = os.path.join(DATA_DIR, "settlement_report.csv")
    ledger_file = os.path.join(DATA_DIR, "internal_ledger.csv")
    bank_file = os.path.join(DATA_DIR, "bank_statement.csv")
    gt_file = os.path.join(DATA_DIR, "ground_truth.json")

    dataset["settlement"].to_csv(settlement_file, index=False)
    dataset["ledger"].to_csv(ledger_file, index=False)
    dataset["bank"].to_csv(bank_file, index=False)

    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(dataset["ground_truth"], f, indent=2)

    print("Successfully generated synthetic 3-way reconciliation dataset:")
    print(f"  - Settlement report: {len(dataset['settlement'])} rows -> {settlement_file}")
    print(f"  - Internal ledger:   {len(dataset['ledger'])} rows -> {ledger_file}")
    print(f"  - Bank statement:    {len(dataset['bank'])} rows -> {bank_file}")
    print(f"  - Ground truth:      {len(dataset['ground_truth'])} records -> {gt_file}")
    print("\nInjected Category Distribution:")
    for cat, count in dataset["counts"].items():
        pct = (count / len(dataset['ground_truth'])) * 100
        print(f"  * {cat:20s}: {count:2d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
