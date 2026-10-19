#!/bin/sh
# Copy the Python package next to the web app so Pyodide can load it.
set -e
cd "$(dirname "$0")/.."
mkdir -p web/py/unbake
for f in __init__.py scene.py easing.py gradients.py morph.py reconstruct.py strokes.py export.py web_api.py; do cp "src/unbake/$f" "web/py/unbake/$f"; done
echo "web/py/unbake updated"
