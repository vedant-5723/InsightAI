"""Smoke-test InsightAI's Ollama and Gemini providers.

Run from the backend folder while Uvicorn is running:
    python test_insightai_providers.py

This script uploads a temporary synthetic dataset to the local API, tests each
provider's /api/chat connectivity, and runs data-analysis questions against the
uploaded dataset. It does not modify project source files.
"""

import getpass
import json
import os
import sys
from typing import Any

import requests

BASE_URL = os.getenv("INSIGHTAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 90

SAMPLE_CSV = """Product,Region,Revenue
Laptop,West,50000
Laptop,East,45000
Phone,West,35000
Phone,East,30000
Tablet,West,20000
"""

QUESTIONS = [
    ("combined_products", "What is the total revenue for Laptop and Phone?", ["160,000", "160000"]),
    ("west_product_ranking", "Which product has maximum revenue in the West?", ["laptop", "50,000", "50000"]),
]


def show_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw_response": response.text[:2000]}


def main() -> int:
    print(f"\nInsightAI provider smoke test: {BASE_URL}\n")

    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=10)
        if not health.ok:
            print(f"Health endpoint returned HTTP {health.status_code}.")
            print("If your project uses a different health route, continue only after confirming the API is running.")
        else:
            print("Health:", json.dumps(show_json(health), ensure_ascii=False))
    except requests.RequestException as exc:
        print(f"Cannot connect to backend at {BASE_URL}: {exc}")
        print("Start the backend first with: uvicorn app.main:app --reload")
        return 1

    gemini_key = getpass.getpass("Gemini API key (input hidden; press Enter to skip Gemini): ").strip()

    try:
        upload = requests.post(
            f"{BASE_URL}/api/datasets",
            files={"file": ("insightai_provider_test.csv", SAMPLE_CSV.encode("utf-8"), "text/csv")},
            timeout=30,
        )
        upload.raise_for_status()
        upload_data = upload.json()
        dataset_id = upload_data["datasetId"]
        print(f"\nDataset uploaded: {upload_data.get('filename')} ({upload_data.get('rowCount')} rows)")
    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"Dataset upload failed: {exc}")
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            print(exc.response.text[:2000])
        return 1

    providers = [("ollama", None, None)]
    if gemini_key:
        providers.append(("gemini", gemini_key, "gemini-3.8-flash"))
    else:
        print("Gemini skipped because no API key was entered.")

    failures = 0
    for provider, api_key, model in providers:
        print(f"\n{'=' * 60}\nPROVIDER: {provider.upper()}\n{'=' * 60}")
        chat_payload = {
            "message": "Reply with exactly: PROVIDER_CONNECTION_OK",
            "provider": provider,
        }
        if api_key:
            chat_payload["apiKey"] = api_key
        if model:
            chat_payload["model"] = model

        try:
            chat_response = requests.post(
                f"{BASE_URL}/api/chat", json=chat_payload, timeout=TIMEOUT_SECONDS
            )
            print(f"Chat connectivity: HTTP {chat_response.status_code}")
            chat_data = show_json(chat_response)
            print(json.dumps(chat_data, ensure_ascii=False, indent=2)[:4000])
            if not chat_response.ok:
                failures += 1
                print("RESULT: PROVIDER CONNECTION FAILED — inspect backend terminal logs and provider credentials/model.")
                continue
        except requests.RequestException as exc:
            failures += 1
            print(f"Chat request failed: {exc}")
            continue

        for label, question, expected_markers in QUESTIONS:
            payload = {
                "datasetId": dataset_id,
                "question": question,
                "provider": provider,
            }
            if api_key:
                payload["apiKey"] = api_key
            if model:
                payload["model"] = model
            try:
                response = requests.post(
                    f"{BASE_URL}/api/agent/analyze", json=payload, timeout=TIMEOUT_SECONDS
                )
                data = show_json(response)
                print(f"\nAnalysis test: {label} | HTTP {response.status_code}")
                print("Question:", question)
                print(json.dumps(data, ensure_ascii=False, indent=2)[:6000])
                if not response.ok:
                    failures += 1
                    print("RESULT: ANALYSIS REQUEST FAILED")
                else:
                    answer_text = json.dumps(data, ensure_ascii=False).lower()
                    if label == "combined_products" and not any(x in answer_text for x in expected_markers):
                        print("CHECK: Could not find the expected 160,000 in the response text. Review the response above; formatting may differ.")
                    elif label == "west_product_ranking" and not all(x in answer_text for x in expected_markers):
                        print("CHECK: Review the response to confirm Laptop is identified with West revenue 50,000.")
                    else:
                        print("CHECK: Expected answer marker(s) found in response.")
            except requests.RequestException as exc:
                failures += 1
                print(f"Analysis request failed: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"Smoke test finished with {failures} failed request(s). See details above.")
        return 1
    print("Smoke test finished: all attempted provider requests returned successfully.")
    print("Note: inspect the analysis answers above; HTTP success alone does not prove semantic correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
