# Architecture

`nightstream` owns the public application and lifecycle API. It compiles an
application against the single exported recursive verifier in
`nightstream-fprime`. Production compilation and execution do not run Lean.

| Owner | Responsibility |
|---|---|
| `nightstream/src/application` | Application program and witness construction |
| `nightstream/src/assembly` | Assemble that program with the shared verifier |
| `nightstream/src/circuit.rs` | Compile, save, and load the circuit package |
| `nightstream/src/folding` | PiCCS, PiRLC, PiDEC, and verifier-driven transcript |
| `nightstream/src/lifecycle` | Prove, extend, and verify the terminal state |
| `nightstream-fprime/src/package` | Sealed format, exact rows, and witness recipes |
| `neo-reductions` | Folding arithmetic and sum-check engines |
| `neo-ajtai` | Fixed-key commitments and low-norm decomposition |
| `formal/nightstream-fprime` | Protocol, circuit, layout, and export proofs |

The application compiler uses one general path. The wide sampler replaces
the old sampler in the canonical layout; it does not add a second package.
Protocol-binding hashes use Poseidon2.

See [lifecycle](lifecycle.md), [applications](frontends.md), and
[terminal verification](decider.md).
