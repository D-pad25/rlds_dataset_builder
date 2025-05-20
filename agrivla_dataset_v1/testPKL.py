import pickle
from pathlib import Path

test_pkl = Path("/mnt/e/VLA_data/CleanData224/v5/0507_162201/step0.pkl")
with open(test_pkl, "rb") as f:
    step = pickle.load(f)

print("✅ Successfully loaded:", type(step))
