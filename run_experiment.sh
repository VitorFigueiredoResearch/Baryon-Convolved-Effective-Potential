#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] Data ingest"
python -m src.io --config config/h1_config.yaml

echo "[2/7] Cross-validated fit (L, μ)"
python -m src.cv_protocol --config config/h1_config.yaml

echo "[3/7] RC metrics & plots"
python -m src.plotting --task rc --config config/h1_config.yaml

echo "[4/7] BTFR"
python -m src.btfr --config config/h1_config.yaml

echo "[5/7] RAR"
python -m src.rar --config config/h1_config.yaml

echo "[6/7] Lensing prediction"
python -m src.lensing_predict --config config/h1_config.yaml

echo "[7/7] Pass/Fail report"
python -m src.model --report --config config/h1_config.yaml

echo "DONE."
