# TERM-CE

`property_id`: `TERM-CE`

`claim`: Terminal acceptance uses verifier-derived children and accepts exactly
when every child has a same-index witness satisfying the verifier-owned CE
relation: public width, Ajtai commitment opening, public projection, low norm,
evaluation-point shape, all ring evaluations, constant terms, and supported
sidecar data. A prover-recorded child list is never authority for itself.

`assumptions`:

- The verifier supplies the relation, norm bound, expected public width, and
  child list.
- The executable primitive operations in `TerminalCE.Semantics` are the mapped
  commitment, projection, norm, evaluation, and sidecar operations. Their Rust
  mapping is pinned by `assurance/rust-conformance-manifest.json` and exercised
  by the production conformance test.
- The Lean kernel and the recorded standard-library foundations are trusted.

`non_goals`:

- Soundness of a compact Spartan or terminal-CE proof.
- Probability bounds for Ajtai binding, SumCheck, Fiat-Shamir, or Poseidon2.
- Circuit correspondence for the terminal verifier.

`paper_sources`: SuperNeo Definition 13 and the terminal CE obligations induced
by the final Construction-2 fold.

`rust_surfaces`:

- `lifecycle::verify::{verify_uncompressed, check_running_witnesses_authority,
  validate_final_witness_authority, verify_uncompressed_audit}`
- `paper::decider::validate_witness`
- `paper::decider::{prove, verify}` only for their explicit fail-closed
  `Unsupported` result

`circuit_or_encoding_artifacts`: Not applicable. This property concerns the
native direct verifier over in-memory claims and witnesses. `ENC-CANON` and the
M4 row-family properties separately own byte/field and generated-row parity.

`failure_class`: disconnected child authority, wrong witness cardinality,
public-width mismatch, commitment mismatch, public-projection mismatch,
low-norm violation, malformed evaluation point, wrong evaluation, wrong
constant term, or unsupported/invalid sidecar.

`counterexample_or_witness`: `tests/TerminalCE.lean` independently reaches all
ten named Lean rejections. `terminal_ce_native_success_and_each_authority_rejection_are_live`
reaches the corresponding production Rust failures, including a recorded
terminal child disconnected from verifier-derived NIFS output.

`lean_theorems`:

- `Nightstream.Protocol.TerminalCE.terminalCE_sound`
- `Nightstream.Protocol.TerminalCE.terminalCE_complete`
- `Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE`
- `Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection`
- `Nightstream.Implementation.Rust.Terminal.verify_ok_iff_check`

`axiom_report`: `terminalCE_sound` and `terminalCE_complete` use `propext`.
The Rust-shaped refinement and rejection theorems use `propext` and
`Quot.sound`. `tests/Axioms.lean` fails closed on any change.

`proof_hash`:

- terminal relation: `sha256:ce46162fe4361039287da85777c5be3e7955bfac01b8af2ee929a60b3195bd44`
- Rust-shaped verifier: `sha256:5920e181bf541cef0fa9711798dbe92e5ee27528ec9323efd2bea6f4a18341ca`

`conformance_status`: `rust-conformant` for the supported direct terminal CE
verifier. The compact decider remains fail-closed and is owned by `DEC-SOUND`.

`retest_commands`:

```bash
cd formal/nightstream-lean
lake build tests.TerminalCE tests.Axioms
lake exe check

cd ../..
perl -e 'alarm shift; exec @ARGV' 300 cargo test -p neo-fold-clean \
  --release --test system_formal_conformance -- --nocapture
```
