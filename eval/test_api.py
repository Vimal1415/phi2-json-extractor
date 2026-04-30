# eval/test_api.py
# ─────────────────────────────────────────────────────────────
# Automated evaluation script for the JSON extraction API.
#
# Usage:
#   python eval/test_api.py                    # tests local server
#   python eval/test_api.py --url https://...  # tests deployed server
#
# Exit codes:
#   0 = all tests passed, accuracy above threshold
#   1 = accuracy below threshold — CI/CD pipeline blocks deployment
# ─────────────────────────────────────────────────────────────

import requests
import json
import sys
import argparse
import time

# ── Configuration ─────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.85   # 85% — below this, deployment is blocked
DEFAULT_URL        = "http://127.0.0.1:8000"

# ── Test cases ────────────────────────────────────────────────
# Each test has:
# - input:    the sentence sent to the API
# - expected: the exact values we expect back
# - description: human readable label for the test report

TEST_CASES = [
    {
        "description": "Simple direct sentence",
        "input": "Aarav is 28 years old and works as a software engineer in Bangalore.",
        "expected": {
            "person": {"name": "Aarav", "age": 28},
            "work":   {"job": "software engineer", "city": "Bangalore"},
        }
    },
    {
        "description": "Senior from years experience >= 5",
        "input": "Priya has been a data scientist in Mumbai for 5 years. She is 31.",
        "expected": {
            "person":  {"name": "Priya", "age": 31},
            "work":    {"job": "data scientist", "city": "Mumbai", "years_experience": 5},
            "details": {"is_senior": True}
        }
    },
    {
        "description": "Fresher — years_experience should be 0",
        "input": "Meera just landed her first job as a UI/UX designer in Pune. She is 22.",
        "expected": {
            "person":  {"name": "Meera", "age": 22},
            "work":    {"job": "UI/UX designer", "city": "Pune", "years_experience": 0},
            "details": {"is_senior": False}
        }
    },
    {
        "description": "Certification detection",
        "input": "Tara is a certified NLP engineer in Bangalore with 3 years experience. She is 28.",
        "expected": {
            "person":  {"name": "Tara", "age": 28},
            "work":    {"city": "Bangalore"},
            "details": {"has_certification": True}
        }
    },
    {
        "description": "Promotion keyword = is_senior true",
        "input": "Ravi was promoted to lead architect in Hyderabad after 8 years. He is 35.",
        "expected": {
            "person":  {"name": "Ravi", "age": 35},
            "work":    {"city": "Hyderabad", "years_experience": 8},
            "details": {"is_senior": True}
        }
    },
    {
        "description": "Passive voice extraction",
        "input": "A senior data scientist role in Mumbai was taken up by Priya, who is 31 with 5 years experience.",
        "expected": {
            "person":  {"name": "Priya", "age": 31},
            "work":    {"city": "Mumbai"},
            "details": {"is_senior": True}
        }
    },
    {
        "description": "Two people — extract only subject",
        "input": "Anjali, 26, is a junior frontend developer in Delhi with 2 years experience. Her manager Suresh is 45.",
        "expected": {
            "person": {"name": "Anjali", "age": 26},
            "work":   {"city": "Delhi", "years_experience": 2},
            "details": {"is_senior": False}
        }
    },
    {
        "description": "City mentioned indirectly",
        "input": "Vikram, a certified blockchain developer, has been based in Kolkata for 3 years. He turned 27.",
        "expected": {
            "person":  {"name": "Vikram", "age": 27},
            "work":    {"city": "Kolkata", "years_experience": 3},
            "details": {"has_certification": True, "is_senior": False}
        }
    },
    {
        "description": "Senior keyword explicit",
        "input": "Sneha is a senior QA engineer in Chennai with 6 years experience. She is 29.",
        "expected": {
            "person":  {"name": "Sneha", "age": 29},
            "work":    {"city": "Chennai", "years_experience": 6},
            "details": {"is_senior": True}
        }
    },
    {
        "description": "Age boundary — exactly not senior (4 years)",
        "input": "Rohan is a frontend developer in Pune with 4 years experience. He is 25.",
        "expected": {
            "person":  {"name": "Rohan", "age": 25},
            "work":    {"city": "Pune", "years_experience": 4},
            "details": {"is_senior": False}
        }
    },
]


def check_fields(actual: dict, expected: dict, path: str = "") -> tuple[int, int]:
    """
    Recursively check if all expected fields match actual response.
    Returns (matched, total) counts.
    """
    matched = 0
    total   = 0

    for key, expected_val in expected.items():
        current_path = f"{path}.{key}" if path else key

        if key not in actual:
            print(f"    MISSING field: {current_path}")
            total += 1
            continue

        actual_val = actual[key]

        if isinstance(expected_val, dict):
            # Recurse into nested dict
            m, t = check_fields(actual_val, expected_val, current_path)
            matched += m
            total   += t
        else:
            total += 1
            # Normalize string comparison
            if isinstance(expected_val, str):
                match = expected_val.lower() in str(actual_val).lower()
            else:
                match = actual_val == expected_val

            if match:
                matched += 1
            else:
                print(f"    MISMATCH {current_path}: expected={expected_val}, got={actual_val}")

    return matched, total


def run_tests(base_url: str) -> float:
    """
    Run all test cases against the API.
    Returns accuracy score between 0.0 and 1.0.
    """
    print(f"\n{'='*60}")
    print(f"  PHI-2 JSON EXTRACTOR — EVAL SUITE")
    print(f"  Target: {base_url}")
    print(f"  Tests:  {len(TEST_CASES)}")
    print(f"  Threshold: {ACCURACY_THRESHOLD*100:.0f}%")
    print(f"{'='*60}\n")

    # Check server is up first
    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        if health.status_code != 200:
            print(f"FATAL: Health check failed — {health.status_code}")
            sys.exit(1)
        print(f"✓ Server is healthy\n")
    except requests.exceptions.ConnectionError:
        print(f"FATAL: Cannot connect to {base_url}")
        print(f"Make sure the server is running first.")
        sys.exit(1)

    total_fields   = 0
    matched_fields = 0
    passed_tests   = 0
    results        = []

    for i, test in enumerate(TEST_CASES):
        print(f"Test {i+1:2d}/{len(TEST_CASES)}: {test['description']}")
        print(f"  Input: {test['input'][:70]}...")

        start = time.time()

        try:
            response = requests.post(
                f"{base_url}/extract",
                json={"text": test["input"]},
                timeout=600       # 10 min timeout for CPU inference
            )
            latency = time.time() - start

            if response.status_code != 200:
                print(f"  ✗ API returned {response.status_code}")
                print(f"  Error: {response.json().get('detail', {}).get('error', 'unknown')}")
                results.append(False)
                continue

            actual = response.json()
            m, t   = check_fields(actual, test["expected"])

            matched_fields += m
            total_fields   += t
            test_passed     = m == t

            if test_passed:
                passed_tests += 1
                print(f"  ✓ PASS — {m}/{t} fields correct ({latency:.1f}s)")
            else:
                print(f"  ✗ FAIL — {m}/{t} fields correct ({latency:.1f}s)")

            results.append(test_passed)

        except requests.exceptions.Timeout:
            print(f"  ✗ TIMEOUT after 600s")
            results.append(False)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append(False)

        print()

    # ── Final report ─────────────────────────────────────────
    test_accuracy  = passed_tests / len(TEST_CASES)
    field_accuracy = matched_fields / total_fields if total_fields > 0 else 0

    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Tests passed:    {passed_tests}/{len(TEST_CASES)} = {test_accuracy*100:.0f}%")
    print(f"  Field accuracy:  {matched_fields}/{total_fields} = {field_accuracy*100:.0f}%")
    print(f"  Threshold:       {ACCURACY_THRESHOLD*100:.0f}%")
    print()

    if test_accuracy >= ACCURACY_THRESHOLD:
        print(f"  ✓ PASSED — accuracy {test_accuracy*100:.0f}% >= {ACCURACY_THRESHOLD*100:.0f}%")
        print(f"  → Deployment approved")
    else:
        print(f"  ✗ FAILED — accuracy {test_accuracy*100:.0f}% < {ACCURACY_THRESHOLD*100:.0f}%")
        print(f"  → Deployment BLOCKED")
    print(f"{'='*60}\n")

    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval suite for phi2 json extractor API")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the API (default: {DEFAULT_URL})"
    )
    args = parser.parse_args()

    accuracy = run_tests(args.url)

    # Exit code 1 = failure — GitHub Actions treats this as pipeline failure
    # Exit code 0 = success — GitHub Actions proceeds to deployment
    if accuracy < ACCURACY_THRESHOLD:
        sys.exit(1)
    else:
        sys.exit(0)