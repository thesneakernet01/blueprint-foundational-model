# SPDX-License-Identifier: Apache-2.0
"""Built-in defaults so the UI is never empty before `export_for_demo.py` runs.

These are replaced at startup by the real artifacts in `demo_artifacts/` once
`summary.json` / `examples.json` exist.
"""

from __future__ import annotations

from typing import Dict, List


def builtin_summary() -> Dict:
    return {
        "placeholder": True,
        "n_raw_features": 13,
        "pca_dim": 64,
        "models": [
            {"key": "raw", "label": "Raw tabular features", "test_auc": None, "test_ap": None},
            {"key": "embed", "label": "Foundation-model embeddings", "test_auc": None, "test_ap": None},
            {"key": "combined", "label": "Combined", "test_auc": None, "test_ap": None},
        ],
        "lift": {"embed_auc_pct": None, "embed_ap_pct": None,
                 "combined_auc_pct": None, "combined_ap_pct": None},
        "note": "Run export_for_demo.py to populate real test-set metrics.",
    }


def builtin_examples() -> List[Dict]:
    return [
        {"label": "High-risk online (odd hour)", "is_fraud": None,
         "txn": {"Amount": "$842.50", "Merchant Name": "DIGITAL-GOODS-LLC",
                 "Merchant City": "ONLINE", "Merchant State": "ONLINE",
                 "Use Chip": "Online Transaction", "MCC": 5942, "Zip": "00000",
                 "Time": "03:14", "Year": 2019, "Month": 11, "Day": 22,
                 "Card": 0, "User": 0}},
        {"label": "Everyday grocery (chip)", "is_fraud": None,
         "txn": {"Amount": "$48.20", "Merchant Name": "KROGER",
                 "Merchant City": "AUSTIN", "Merchant State": "TX",
                 "Use Chip": "Chip Transaction", "MCC": 5411, "Zip": "78758",
                 "Time": "18:02", "Year": 2019, "Month": 11, "Day": 22,
                 "Card": 1, "User": 0}},
        {"label": "Fuel swipe", "is_fraud": None,
         "txn": {"Amount": "$61.00", "Merchant Name": "SHELL",
                 "Merchant City": "DALLAS", "Merchant State": "TX",
                 "Use Chip": "Swipe Transaction", "MCC": 5541, "Zip": "75201",
                 "Time": "08:41", "Year": 2019, "Month": 11, "Day": 22,
                 "Card": 0, "User": 0}},
    ]
