These files are expected test values from the recorded Lean exports at source
commit `9787d8e77069246e3e2afc7dcfab755556fd5023`. The Rust application builder
does not read them. Ordinary Rust tests need no Lean installation.

`poseidon2-application-reference.json` is the exact application plan at index
3 of `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`.
It contains the four input state columns, four private message columns, four
output state columns, all 7,700 application rows, and all 7,696 arithmetic
witness recipes. These dimensions belong to the selected application.

`poseidon2-application-execution.json` contains `[prior_state, message, output]`
from `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-parity.json`.
The prior state is the current-state block at words 35 through 38 of the
recorded prior preimage. The saved profile is the Nightstream Goldilocks
profile with `b = 2`, `k_rho = 16`, and `B = 2^16`.

Run this extraction from the repository root after the separate maintainer
workflow has produced the recorded Lean artifacts:

```python
import json
from pathlib import Path

source = Path("formal/nightstream-fprime/artifacts")
target = Path("crates/nightstream/tests/fixtures")
package = json.loads((source / "nightstream-fprime-stage1-poseidon2-hash-chain-v1.json").read_text())
parity = json.loads((source / "nightstream-fprime-stage1-poseidon2-hash-chain-v1-parity.json").read_text())
fixtures = {
    "poseidon2-application-reference.json": package[3],
    "poseidon2-application-execution.json": [parity[1][1][35:39], parity[1][2], parity[2][0]],
}
for name, value in fixtures.items():
    (target / name).write_text(json.dumps(value, separators=(",", ":")) + "\n")
```

The row test compares canonical A/B/C coefficients, physical variable indices,
and row order after port relocation. The witness test executes the saved Lean
recipe program separately and compares every generated field value. It also
compares the recorded output and the native Poseidon2 hash. These checks cover
this concrete application. They are not a proof for all Rust applications.

The base/recursive computation test reads the two saved Lean caller fixtures
listed below. It runs the Rust application on each recorded current state and
message, compares all four output words with the stored Lean output, and checks
the generated witness against every stored A/B/C application row. The computed
base output must equal the next fixture's input. A changed output must violate
a stored row. This test performs no recursive proving and runs no Lean command.

The assembly test also compares the complete raw application plan, including
duplicate sparse terms and witness expression order. A separate test compares
the complete assembled package value with the selected saved reference.

The following links expose existing recorded outputs to package-local tests:

| Package path | Recorded source |
| --- | --- |
| `lean/nightstream-fprime-stage1-base-step-fixture-v1.json` | Same filename under `formal/nightstream-fprime/artifacts`. |
| `lean/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json` | Same filename under `formal/nightstream-fprime/artifacts`. |
| `lean/nightstream-fprime-stage1-base-nifs-result-v1.json` | Same filename under `formal/nightstream-fprime/artifacts`. |
| `stage1_actual_nifs/actual_result.json` | Same filename under `crates/neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs`. |
| `stage1_actual_nifs/proof.native` | `proof.bin` under `crates/neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs`. |
| `stage1_recursive_states/nonzero-running.json` | Same filename under `crates/neo-fold-clean/tests/nifs/fixtures/stage1_recursive_states`. |

Cargo packaging stores the linked file contents. The links add no second copy
of the recorded data to Git. The native proof has a different package filename
because the repository ignores new files named `proof.bin`; its bytes are
unchanged. Its source and generation evidence are recorded in the original
`stage1_actual_nifs/README.md` and the repository's
`docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.md`.

The selected verifier blueprint has one package copy under `artifacts`, shared
by assembly and lifecycle tests. These test-only saved outputs are never read
by `Circuit::compile`, `load`, `prove`, `extend`, or `verify`.
