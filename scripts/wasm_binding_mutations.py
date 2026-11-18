#!/usr/bin/env python3
"""Write mutated copies of the Micro binding export into its directory, one for each check that
`checkWasmBinding` must fail: ranks swapped across rows, an unaligned region, a changed bound
column, a changed table map, and a written memory that shares a table."""

import copy
import json
from pathlib import Path
import sys


def overlay(logical, table):
    tables = []
    for t in range(max(table) + 1):
        members = [memory for memory, owner in zip(logical, table) if owner == t]
        tables.append([members[0][0] if members else 0, [port for memory in members for port in memory[1]]])
    return tables


def main():
    directory = Path(sys.argv[1])
    export = json.loads((directory / 'wasm-binding-micro-v1.json').read_text())
    mutations = {}
    x = copy.deepcopy(export); x[9][0][0][0], x[9][1][0][0] = x[9][1][0][0], x[9][0][0][0]
    mutations['ranks-across-rows'] = x
    x = copy.deepcopy(export); x[7][0][0] += 1
    mutations['unaligned-region'] = x
    x = copy.deepcopy(export); x[9][100][0][4] += 1
    mutations['bound-column'] = x
    x = copy.deepcopy(export); x[5][10] = x[5][9]
    mutations['table-map'] = x
    x = copy.deepcopy(export); x[5][1] = x[5][0]; x[6] = overlay(x[4], x[5])
    mutations['shared-written-table'] = x
    for name, value in mutations.items():
        (directory / f'mutated-{name}.json').write_text(json.dumps(value, separators=(',', ':')) + '\n')
        print(directory / f'mutated-{name}.json')


if __name__ == '__main__':
    main()
