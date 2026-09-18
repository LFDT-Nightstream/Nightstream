# RUST-REFINE

`property_id`: `RUST-REFINE`

`claim`: The supported Rust verification surfaces have the same acceptance
boundary as the Lean executable model. Native F' base and recursive steps
accept exactly `Step.LocalHolds`; adding the next-consumer or terminal outgoing
link closes `Step.Holds`. Direct terminal verification accepts exactly
`TerminalCE.Holds`. Invalid inputs return a named rejection. The public compact
entrypoint is also conformant to its current contract: it always fails closed
with `Unsupported`.

`assumptions`:

- The Rust-to-Lean type and primitive-operation mapping in this spec is a
  trusted translation boundary. Full-file hashes make any mapped-source change
  reopen this property.
- NIFS, application, digest, Nebula, commitment, projection, norm, and ring
  evaluation are verifier-owned executable parameters at this layer. Their
  cryptographic soundness and exact circuit implementations are separate
  properties.
- Counter machine representation is discharged by `FPR-COUNTER-REFINE`; this
  module uses the M3 mathematical `Nat` state.

`non_goals`:

- Claiming a working compressed decider or `DEC-SOUND`.
- Replacing M4 generated-circuit correspondence.
- Cryptographic probability bounds or the M6 verifier reduction.

`paper_sources`: HyperNova Construction 2; SuperNeo terminal CE relation and
final-fold verifier obligations.

`rust_surfaces`:

- `paper::f_prime::native::verify`
- `paper::construction2::transition::{state_base_case_check, advance_state}`
- `lifecycle::verify::{verify_uncompressed, verify_uncompressed_audit,
  check_running_witnesses_authority}`
- `paper::decider::{validate_witness, prove, verify}`
- `lifecycle::compress::{compress, verify}`

`circuit_or_encoding_artifacts`: Not applicable to the native control-flow
refinement. Existing `ENC-CANON` and M4 artifacts own canonical public encoding
and exact generated rows. The source manifest is an assurance artifact, not a
protocol digest.

`failure_class`: wrong branch variant, malformed entry state, empty prior or
next batch, broken prior link, NIFS rejection, semantic/Nebula advance failure,
wrong next state, wrong `x_out`, any terminal CE authority failure, or an
attempt to use the unsupported compact decider.

`counterexample_or_witness`:

- `tests/FPrimeStep.lean` reaches every Rust-shaped F' error class.
- `tests/TerminalCE.lean` reaches every terminal error class.
- `system_formal_conformance.rs` runs honest production lifecycle/terminal
  processing, mutates each authority family, replays a two-step audit, and
  confirms compact compression returns `Error::Decider(Unsupported)`.

`lean_theorems`:

- `Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal`
- `Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step`
- `Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection`
- `Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE`
- `Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection`
- `Nightstream.Implementation.Rust.Terminal.verify_ok_iff_check`

`axiom_report`: The named F' and terminal refinement/rejection theorems use only
`propext` and `Quot.sound`. The expectations are fail-closed in
`tests/Axioms.lean`.

`proof_hash`:

- F' control flow: `sha256:9808a16918233d44fc3ba2ab918d2106880455abe3d952d8fcf194098bb6a1c5`
- terminal control flow: `sha256:5920e181bf541cef0fa9711798dbe92e5ee27528ec9323efd2bea6f4a18341ca`
- runtime suite: `sha256:76f5a96dd84683297601302ed25e0fef9e18ff7be33f047c00ba1e296f0f34a5`

`conformance_status`: `rust-conformant` for the currently supported
uncompressed/audit F' lifecycle and direct terminal CE verifier. The compact
decider's mapped behavior is fail-closed `Unsupported`; enabling compact
acceptance automatically reopens this property and `DEC-SOUND` through the
source-hash gate.

Completion-gate decisions:

1. Statement parity: pass; preconditions, postconditions, and named failures
   are mapped in the two Rust-shaped Lean programs.
2. State parity: pass for authority-bearing M3 state; u64 representation and
   overflow are delegated to `FPR-COUNTER-REFINE`.
3. Transition parity: pass for base, recursive, outgoing-link, terminal, and
   compact-unsupported branches.
4. Encoding/artifact parity: not applicable to native in-memory verification;
   `ENC-CANON`/M4 own that independent boundary.
5. Runtime regression: pass in `system_formal_conformance`.
6. Lean build, assumption report, proof hash: pass.
7. Drift gate: pass via full-file SHA-256 hashes and theorem anchors in
   `assurance/rust-conformance-manifest.json`. SHA-256 is used only by this
   offline drift tool, never in a protocol transcript or public digest.
8. Concurrency/cancellation: not applicable; all mapped verifier functions are
   deterministic synchronous calls with no task ownership or cancellation.

`retest_commands`:

```bash
cd formal/nightstream-lean
lake build tests.FPrimeStep tests.TerminalCE tests.Axioms check
lake exe check

cd ../..
perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean \
  --release --test system_formal_conformance -- --nocapture
```
