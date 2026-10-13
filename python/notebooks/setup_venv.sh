#!/usr/bin/env bash
# Create a virtual environment for the notebooks and install their requirements.
#
#   ./setup_venv.sh            # uses python3; set PYTHON=/path/to/python to override
#
# Then run the printed command once to register the Jupyter kernel the
# notebooks use ("modelgami").
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV=".venv"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

echo
echo "Done. To register the Jupyter kernel the notebooks use, run:"
echo
echo "  $(pwd)/$VENV/bin/python -m ipykernel install --user --name modelgami --display-name \"Python (modelgami)\""
echo
