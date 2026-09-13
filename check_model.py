import pickle
import sys

THRESHOLD = 0.90

with open("model.pkl", "rb") as f:
    meta = pickle.load(f)["metadata"]

acc = meta["train_accuracy"]
print(f"accuracy={acc:.3f}, threshold={THRESHOLD}")
sys.exit(0 if acc >= THRESHOLD else 1)
