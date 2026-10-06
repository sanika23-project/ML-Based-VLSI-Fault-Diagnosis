from circuit_sim import signature, all_fault_classes
import pandas as pd


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


# Generate clean signatures
signatures = {}

for label, site, value in all_fault_classes():
    signatures[label] = signature(site, value)


print("\n" + "=" * 80)
print("             FAULT SIGNATURE ANALYSIS")
print("=" * 80)

labels = list(signatures.keys())

# Print signature of every fault
for label in labels:
    print(f"\n{label}:")
    print("".join(map(str, signatures[label])))


# Find closest fault pairs
pairs = []

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        distance = hamming_distance(
            signatures[labels[i]],
            signatures[labels[j]]
        )

        pairs.append({
            "Fault 1": labels[i],
            "Fault 2": labels[j],
            "Different Bits": distance
        })


pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sort_values("Different Bits")

print("\n" + "=" * 80)
print("             CLOSEST FAULT SIGNATURES")
print("=" * 80)

print(pairs_df.head(15).to_string(index=False))

pairs_df.to_csv("signature_similarity.csv", index=False)

print("\nResults saved to: signature_similarity.csv")