# Architecture

`nightstream` owns the public application and lifecycle API. Lean emits the
light and refresh step packages and the finish package, and
`nightstream-fprime` loads them. `nightstream` puts the Rust application in
the step packages. Production compilation and execution do not run Lean.

| Owner | Responsibility |
|---|---|
| `nightstream/src/application` | Application program and witness construction |
| `nightstream/src/circuit.rs` | The public API that compiles, saves and loads the application and makes the prover and verifier |
| `nightstream/src/folding` | PiCCS, PiRLC, PiDEC, the light and refresh folds, and the verifier-driven transcript |
| `nightstream/src/lifecycle/option_b` | Steps, verification, proof bytes and the final proof |
| `nightstream/src/nebula` | Nebula memory applications |
| `nightstream-fprime/src/option_b` | Loading of the option-B packages, their identity and their words |
| `nightstream-fprime/src/package` | Package formats, matrix programs and the PiCCS transcripts of the folds |
| `neo-reductions` | Folding arithmetic and sum-check engines |
| `neo-ajtai` | Fixed-key commitments and low-norm decomposition |
| `neo-spartan` | Layer 1 and the shrink layer of the final proof |
| `formal/nightstream-fprime` | Protocol, circuit, layout, and export proofs |

The application compiler uses one general path. Protocol-binding hashes use
Poseidon2.

See [lifecycle](lifecycle.md), [applications](frontends.md), and the
[protocol specification](../../SPEC.md).
