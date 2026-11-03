"""Test the LAB-2 API using the requests library.

NOTE: The API must be running before you execute this file, for example:
    uvicorn main:app --reload
Then open another terminal and run:
    python test_api.py
"""

import sys

import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120  # the first request may be slow because the model is downloaded/loaded


def show(title: str, response: requests.Response) -> None:
    print(f"\n=== {title} ===")
    print("Status:", response.status_code)
    print("Body  :", response.json())


def main() -> int:
    failures = 0

    def check(condition: bool, message: str) -> None:
        nonlocal failures
        if not condition:
            failures += 1
            print(f"  [FAIL] {message}")
        else:
            print(f"  [OK]   {message}")

    try:
        r = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to {BASE_URL}. Start the API first (uvicorn main:app).")
        return 1
    show("GET /", r)
    check(r.status_code == 200, "GET / returns 200")

    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    show("GET /health", r)
    check(r.status_code == 200 and r.json().get("status") == "ok", "GET /health has status 'ok'")

    # --- Two different inputs (positive / negative) ---
    positive = "I love this movie, it was absolutely fantastic!"
    r = requests.post(f"{BASE_URL}/predict", json={"text": positive}, timeout=TIMEOUT)
    show("POST /predict (positive sentence)", r)
    check(r.status_code == 200, "returns 200")
    check(r.json().get("label") == "POSITIVE", "label is POSITIVE")

    negative = "This was a terrible experience, I want my money back."
    r = requests.post(f"{BASE_URL}/predict", json={"text": negative}, timeout=TIMEOUT)
    show("POST /predict (negative sentence)", r)
    check(r.status_code == 200, "returns 200")
    check(r.json().get("label") == "NEGATIVE", "label is NEGATIVE")

    # --- Error cases ---
    r = requests.post(f"{BASE_URL}/predict", json={"text": "   "}, timeout=TIMEOUT)
    show("POST /predict (empty text)", r)
    check(r.status_code == 422, "empty text -> 422")

    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=TIMEOUT)
    show("POST /predict (missing text)", r)
    check(r.status_code == 422, "missing text -> 422")

    r = requests.post(f"{BASE_URL}/predict", json={"text": 12345}, timeout=TIMEOUT)
    show("POST /predict (wrong data type)", r)
    check(r.status_code == 422, "wrong type -> 422")

    print(f"\nDone: {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
