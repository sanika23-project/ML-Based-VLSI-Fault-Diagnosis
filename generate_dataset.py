"""
generate_dataset.py

Builds the training dataset for fault diagnosis.

Each row = one fault class's "signature": the circuit's output response
to all 32 possible input test vectors, flattened into 64 binary features
(this mimics the real ATPG-style approach: apply a full test set, record
the output response, use that response to diagnose the fault).

Since a single fault only gives ONE signature, we add small amounts of
random test-response noise (simulating real-world test/measurement
imperfection and partial test sets) to create multiple noisy samples
per class, so the classifier has something to learn from and generalize.

Output: dataset.csv  (64 feature columns 'b0'..'b63' + 'label')
"""

import random
import pandas as pd
from circuit_sim import all_fault_classes, signature

SAMPLES_PER_CLASS = 200   # noisy variants generated per fault class
NOISE_FLIP_PROB = 0.03    # probability each bit is flipped (simulated test noise)
RANDOM_SEED = 42


def add_noise(bits, flip_prob, rng):
    return [1 - b if rng.random() < flip_prob else b for b in bits]


def build_dataset():
    rng = random.Random(RANDOM_SEED)
    rows = []
    classes = all_fault_classes()

    for label, site, value in classes:
        base_signature = signature(site, value)
        for _ in range(SAMPLES_PER_CLASS):
            noisy = add_noise(base_signature, NOISE_FLIP_PROB, rng)
            row = {f"b{i}": bit for i, bit in enumerate(noisy)}
            row["label"] = label
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("dataset.csv", index=False)
    print(f"Generated dataset.csv with {len(df)} rows, {df['label'].nunique()} classes")
    print(df["label"].value_counts())
