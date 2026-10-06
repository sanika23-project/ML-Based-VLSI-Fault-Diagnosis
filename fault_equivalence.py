from circuit_sim import signature, all_fault_classes


def build_equivalence_groups():
    """
    Groups faults that produce exactly the same primary-output signature.
    """
    groups = {}

    for label, site, value in all_fault_classes():
        sig = tuple(signature(site, value))

        if sig not in groups:
            groups[sig] = []

        groups[sig].append(label)

    return list(groups.values())


def find_equivalent_faults(predicted_fault):
    """
    Return the complete equivalence group containing the predicted fault.
    """
    groups = build_equivalence_groups()

    for group in groups:
        if predicted_fault in group:
            return group

    return [predicted_fault]


if __name__ == "__main__":
    groups = build_equivalence_groups()

    print("\n" + "=" * 70)
    print("             FAULT EQUIVALENCE GROUPS")
    print("=" * 70)

    for i, group in enumerate(groups, 1):
        if len(group) > 1:
            print(f"Group {i}: {', '.join(group)}")

    print("\nTotal equivalence groups:", len(groups))
    