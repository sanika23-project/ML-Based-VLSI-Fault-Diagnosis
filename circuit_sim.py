"""
circuit_sim.py

Logic-level simulator for the ISCAS-85 'c17' benchmark circuit, with
support for injecting single stuck-at faults (SA0 / SA1) at any
internal gate output.

Netlist (standard c17):
    N10 = NAND(N1, N3)
    N11 = NAND(N3, N6)
    N16 = NAND(N2, N11)
    N19 = NAND(N11, N7)
    N22 = NAND(N10, N16)
    N23 = NAND(N16, N19)

Primary inputs : N1, N2, N3, N6, N7
Primary outputs: N22, N23
Internal lines (fault sites): N10, N11, N16, N19, N22, N23
"""

from itertools import product

PRIMARY_INPUTS = ["N1", "N2", "N3", "N6", "N7"]
FAULT_SITES = ["N10", "N11", "N16", "N19", "N22", "N23"]


def _nand(a, b):
    return 1 - (a & b)


def simulate(inputs, fault_site=None, fault_value=None):
    """
    Simulate c17 for one input pattern.

    inputs: dict mapping primary input name -> 0/1
    fault_site: name of internal line to stick (or None for fault-free)
    fault_value: 0 or 1 (stuck-at value), required if fault_site is set

    Returns dict of all line values including outputs N22, N23.
    """
    lines = dict(inputs)

    def get(name):
        # apply stuck-at fault if this is the faulty line
        if fault_site == name:
            return fault_value
        return lines[name]

    lines["N10"] = _nand(get("N1"), get("N3"))
    lines["N11"] = _nand(get("N3"), get("N6"))
    lines["N16"] = _nand(get("N2"), get("N11"))
    lines["N19"] = _nand(get("N11"), get("N7"))
    lines["N22"] = _nand(get("N10"), get("N16"))
    lines["N23"] = _nand(get("N16"), get("N19"))

    # re-apply fault to final output values too, in case the fault
    # site itself is a primary output line
    if fault_site in ("N22", "N23"):
        lines[fault_site] = fault_value

    return lines


def all_input_patterns():
    """All 2^5 = 32 combinations of the 5 primary inputs, in fixed order."""
    patterns = []
    for bits in product([0, 1], repeat=len(PRIMARY_INPUTS)):
        patterns.append(dict(zip(PRIMARY_INPUTS, bits)))
    return patterns


def signature(fault_site=None, fault_value=None):
    """
    Run all 32 input patterns through the circuit (with the given fault,
    or fault-free if fault_site is None) and return the concatenated
    output response as a flat list of 0/1 bits (length 64: 32 patterns
    x 2 outputs). This is the ML feature vector for one fault class.
    """
    bits = []
    for pattern in all_input_patterns():
        result = simulate(pattern, fault_site, fault_value)
        bits.append(result["N22"])
        bits.append(result["N23"])
    return bits


def all_fault_classes():
    """
    List of (label, fault_site, fault_value) for every class the model
    predicts: fault-free plus SA0/SA1 at each of the 6 fault sites.
    """
    classes = [("fault_free", None, None)]
    for site in FAULT_SITES:
        classes.append((f"{site}_SA0", site, 0))
        classes.append((f"{site}_SA1", site, 1))
    return classes
