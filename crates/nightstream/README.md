# Nightstream

Rust application circuits, a generic assembler for the Lean-exported recursive
verifier, and the native `prepare`, `prove`, `extend`, and `verify` lifecycle.
Lean is used only by maintainers to produce and check the shared exports.

The package includes the selected verifier blueprint at
`artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`.
Copy this file into the application's assets and pass its bytes to
`Circuit::prepare`. The shared manifest and formula library are included by
the Rust build. No Lean installation or artifact generation is needed.

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit};

let reference = std::fs::read("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")?;
let circuit = Circuit::prepare(&reference, poseidon2_hash_chain_v1()?)?;
```

Use the packaged blueprint as local verifier configuration. `prepare` checks
the selected reference identities before it inserts the Rust application.
Keep the prepared `Circuit` for later proving and verification.
Call `prove` for the first step, `extend` for each later step, and `verify`
with the state expected by the application.

Applications use four Goldilocks state words, private inputs, affine operations,
multiplication, and equality constraints. The assembler keeps every required
verifier component and binds the resulting application and circuit identity.

The current `Circuit` uses a prefix of the selected production commitment key.
Private input words plus generated local words must be at most **7,701**. This
comes from the existing layout
`252,695,531 + 41 × (private_words + local_words)` and the selected key's
253,011,276-coefficient capacity. The exported row and domain checks also apply.
A larger application needs a separately supported key; preparation rejects it.

The selected profile remains `b = 2`, `k_rho = 16`, `B = 2^16`, with one fresh
claim and sixteen carried claims. Terminal verification checks all remaining
openings. A production compression backend is outside this crate goal.

The independent Rust Poseidon2 application supplies the first comparison with
the existing Lean package. Exact comparisons are implementation evidence, not
a universal Lean proof of Rust application semantics or assembly. The project
contract is `NIGHTSTREAM_CRATE_GOAL.md` in the repository root. See the
[artifact instructions](artifacts/README.md) for consumer files, saved tests,
and the separate maintainer workflow.

The Cargo package includes the saved test inputs. Tests use package-local data
and do not run Lean. `neo-fold-clean` is a development dependency for comparison
with the unchanged implementation; it is not a production dependency.

See [VALIDATION.md](VALIDATION.md) for the completed fresh two-fold replay,
full reference comparisons, terminal checks, measured costs, and scope limits.
The selected implementation goal is complete. A single-process active
`extend` run and a universal Rust refinement proof are not claimed.
