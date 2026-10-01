These application fixtures are extracted from the current Lean exports below.
The Rust application builder does not read them. Ordinary Rust tests need no
Lean installation. Saved fold fixtures require a separate fresh conformance run.

`poseidon2-application-reference.json` is the exact application plan at index
3 of `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`.
It contains the four input state columns, four private message columns, four
output state columns, all 5,484 application rows, and all 5,480 arithmetic
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

The Lean entries below link to the maintained formal artifacts. Native fold
fixtures and `golden-v1.zip` are regenerated only after fresh CPU execution and
both complete Lean comparisons pass. The state-request fixture supplies the
fixed public initial state and application messages; it contains no proof or
commitment authority.

| Package path | Current source |
| --- | --- |
| `lean/nightstream-fprime-stage1-base-step-fixture-v1.json` | The current Lean base emitter. |
| `lean/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json` | First fresh checked Lean caller. |
| `lean/nightstream-fprime-stage1-base-nifs-result-v1.json` | First fresh checked complete ten-field Lean NIFS result. |
| `stage1_actual_nifs/actual_result.json` | First fresh native fold, compared completely with Lean. |
| `stage1_actual_nifs/proof.native` | The same fold's exact native proof bytes, also encoded from Lean. |
| `stage1_recursive_states/nonzero-running.json` | Public state and message request for the second fold. |
| `golden-v1.zip` | Both checked folds and state interfaces; exactly 19 files. |

The linked files have one copy in Git. The archive excludes private witness
matrices; fresh conformance separately compares complete physical witnesses.
See [the workflow](../../../../scripts/GOLDEN_CONFORMANCE.md) and
[the replacement report](../../../../docs/reviews/pirlc-sampler-replacement/REPORT.md).

The selected verifier blueprint has one package copy under `artifacts`, shared
by assembly and lifecycle tests. These test-only saved outputs are never read
by `Circuit::compile`, `load`, `prove`, `extend`, or `verify`.
