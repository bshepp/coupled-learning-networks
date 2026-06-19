# Ensures the repo root is importable so tests can import the flat top-level modules
# (solver.py, network.py, ...) regardless of pytest's import mode.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
