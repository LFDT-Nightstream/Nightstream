"""Count the candidate sampler's normalized matrices from Lean-emitted forms.

Range rows are normalized in the Lean experiment. Poseidon inputs and all
round constants come from the same experiment. Reuse the existing exact
field and sparse linear-algebra counter to avoid expanding expression trees.
"""

import argparse
import json
from pathlib import Path

from application_matrix_cost import add, entries, external, field, scale


def measure(data):
    counts = list(data["range_nonzeros"])
    range_counts = list(counts)
    retained = (2, 34 * 86, data["poseidon_retained_start"])
    rows = 17 * 681
    for invocation, inputs in enumerate(data["poseidon_inputs"]):
        state = external([entries(form) for form in inputs])
        next_slot = invocation * 86

        def sbox(value):
            nonlocal next_slot, rows
            output = field(retained, next_slot)
            counts[1] += 1
            counts[4] += len(output)
            counts[5] += len(value)
            next_slot += 1
            rows += 1
            return output

        for constants in data["initial_constants"]:
            state = external([sbox(add(form, {0: constant}))
                              for form, constant in zip(state, constants, strict=True)])
        for constant in data["internal_constants"]:
            state[0] = sbox(add(state[0], {0: constant}))
            total = add(*state)
            state = [add(scale(coefficient, form), total)
                     for coefficient, form in zip(data["internal_diagonal"], state, strict=True)]
        for constants in data["terminal_constants"]:
            state = external([sbox(add(form, {0: constant}))
                              for form, constant in zip(state, constants, strict=True)])
        assert next_slot == (invocation + 1) * 86
    assert rows == data["rows"]
    return {
        "scope": "candidate sampler only; previous Poseidon endpoint as input",
        "selected_in_production": False,
        "logical_rows": rows,
        "committed_coordinates": data["coordinates"],
        "normalized_matrix_nonzeros": sum(counts),
        "normalized_matrix_nonzeros_by_port": counts,
        "range_matrix_nonzeros": sum(range_counts),
        "poseidon_matrix_nonzeros": sum(counts) - sum(range_counts),
        "temporary_helpers": 17 * 1404,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()
               if line.startswith('{')]
    assert len(records) == 1, "expected one complete Lean cost record"
    print(json.dumps(measure(records[0]), indent=2))


if __name__ == "__main__":
    main()
