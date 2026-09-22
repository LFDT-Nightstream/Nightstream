"""Read-only cost and source-read scan of the installed transition block."""

import collections
import hashlib
import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[3] / (
    "formal/nightstream-fprime/artifacts/"
    "nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
)
EXPECTED_SHA256 = "6216d1f62250a58d073ecf0a908bd074d3834957ec5be3361620bbdeb5a97642"
P = 18446744069414584321
SCRATCH_START, SCRATCH_END = 29040309, 29336446
ROW_START, ROW_END = 28872529, 29218024


def main():
    with ARTIFACT.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == EXPECTED_SHA256, "This scan describes one exact installed package"
    package = json.loads(ARTIFACT.read_text())
    ranges, grids = package[2][19][1][2]

    def source_width(column):
        for start, count, retained, offset in ranges:
            if start <= column < start + count:
                assert retained[0] == 2
                return 41
        for start, major_count, stride, minor_count, minor_stride, run, retained, mode, *_ in grids:
            if column >= start:
                major, remainder = divmod(column - start, stride)
                minor, offset = divmod(remainder, minor_stride)
                if major < major_count and minor < minor_count and offset < run:
                    assert retained[0] == 2 and mode == 1
                    return 8 * 41
        raise ValueError(f"Unresolved transition source {column}")

    def nonzeros(combination):
        combined = collections.defaultdict(int)
        for column, coefficient in combination[1]:
            combined[column] = (combined[column] + coefficient) % P
        return int(combination[0] % P != 0) + sum(
            source_width(column) for column, coefficient in combined.items() if coefficient
        )

    def scratch(column):
        return SCRATCH_START <= column < SCRATCH_END

    def expression_reads(expression):
        if expression[0] == 0:
            return int(scratch(expression[1]))
        if expression[0] == 1:
            return 0
        assert expression[0] in (2, 3)
        return expression_reads(expression[1]) + expression_reads(expression[2])

    old = [0, 0, 0, 0]
    own_instructions = own_assertions = external_row_reads = 0
    for row, output, left, right in package[1][11]:
        if ROW_START <= row < ROW_END:
            own_instructions += 1
            old[0] += 1
            for index, combination in enumerate((left, right, [0, [[output, 1]]])):
                old[index + 1] += nonzeros(combination)
        else:
            external_row_reads += sum(scratch(column) for combination in (left, right)
                                      for column, _ in combination[1])
    for row, left, right, output in package[1][12]:
        if ROW_START <= row < ROW_END:
            own_assertions += 1
            old[0] += 1
            for index, combination in enumerate((left, right, output)):
                old[index + 1] += nonzeros(combination)
        else:
            external_row_reads += sum(scratch(column) for combination in (left, right, output)
                                      for column, _ in combination[1])
    recipe_reads = sum(expression_reads(expression)
                       for _, recipes, _ in package[1][10] for expression in recipes)
    hint_reads = sum(expression_reads(hint[1])
                     for _, _, hints in package[1][10] for hint in hints)
    permutation_reads = sum(scratch(column) for invocation in package[1][7]
                            for combination in invocation[3] for column, _ in combination[1])
    compact_reads = sum(scratch(column + index * stride) for invocation in package[1][9]
                        for _, count, column, stride in invocation[4] for index in range(count))
    print(json.dumps({
        "artifact_sha256": digest,
        "scope": "Installed package scan; no universal support or candidate integration proof.",
        "scratch_columns_half_open": [SCRATCH_START, SCRATCH_END],
        "source_rows_half_open": [ROW_START, ROW_END],
        "old_row_count": own_instructions + own_assertions,
        "old_instructions": own_instructions,
        "old_assertions": own_assertions,
        "old_nnz_by_selector_ABC": old,
        "old_nnz": sum(old),
        "outside_ordinary_and_instruction_reads": external_row_reads,
        "all_recipe_and_hint_reads": recipe_reads + hint_reads,
        "permutation_input_reads": permutation_reads,
        "compact_input_reads": compact_reads,
        "candidate_analytical_nnz": 49248 * 84 + 56 * 371 + 49 * 2 + 84 + 44 + 4 * 85,
    }, indent=2))


if __name__ == "__main__":
    main()
