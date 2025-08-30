# testPKL_patch.py
import pickle
import numpy
import types

# 🩹 Patch missing numpy._core for legacy pickle compatibility
if not hasattr(numpy, "_core"):
    numpy._core = types.SimpleNamespace()
    numpy._core.multiarray = numpy.core.multiarray

# 🔄 Try loading the file again
from pathlib import Path

pkl_path = Path("/mnt/e/VLA_data/CleanData224/v5/0507_162201/2025-05-07T16-22-01.188691.pkl")

with open(pkl_path, "rb") as f:
    step = pickle.load(f)

print("✅ Successfully loaded:", type(step))