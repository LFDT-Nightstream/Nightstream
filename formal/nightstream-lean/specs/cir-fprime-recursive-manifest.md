# CIR-FPR-RECURSIVE-MANIFEST

```text
property_id: CIR-FPR-RECURSIVE-MANIFEST
claim:
  For the exact diagnostic plain/stateless/direct-CCS-bit-carrier steady
  recursive profile, the generated `FPrimeRecursiveManifest.totalRows` rows
  form eight contiguous named top-level families. Its generated
  `FPrimeRecursiveManifest.nifsRowCount`-row NIFS interval is exactly
  partitioned into PiCCS, PiRLC, PiDEC, and point-binding families. These
  generated totals are diagnostic measurements, not semantic authority.
  Every interval hashes the exact sparse A/B/C triplets. The Rust harness
  renders and byte-compares the Lean data module from those same rows, and
  source drift fails the regression.

  Independently, ProjectedChecks is the explicit production-facing semantic
  proof interface. Its kernel theorem composes decoded facts into
  Step.LocalHolds or a named BadRoot; DecodedChecks remains the exact branch.
assumptions:
  - SHA-256 is used only for offline drift identity, never as protocol
    authority or as a premise of a semantic theorem.
  - The selected carrier is a direct-CCS diagnostic padding relation over the
    verifier's current F' public carrier. It is not the selective fixed-point
    relation.
non_goals:
  - Satisfaction of a range implies its ProjectedChecks fields.
  - CIR-SOUND or CIR-COMPLETE.
  - Stateful, Nebula, different batch-size, or full-history profiles.
rust_surfaces:
  - engine/r1cs_circuit/builder.rs::{record_row_family,row_family_ranges}
  - paper/f_prime/r1cs.rs::enforce_f_prime_recursive_step_circuit
  - paper/nifs/circuit/mod.rs::enforce_nifs_v_circuit_with_transcript_inner
  - paper/nifs/circuit/pi_rlc/** for the complete PiRLC lifecycle
  - tests/gadgets/f_prime_recursive_manifest.rs
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program
  - Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block
  - Nightstream.Assurance.FPrimeRecursiveCircuit.decodedChecks_sound
  - Nightstream.Assurance.FPrimeRecursiveCircuit.decodedChecks_local_sound
artifact:
  assurance/fprime-recursive-program-manifest.json
evidence_state: artifact-checked
route: harness-first
remaining_bridges:
  - PiCCS row satisfaction implies the concrete PiCCS verifier result.
  - PiRLC row satisfaction implies the shared-coefficient combination result.
  - PiDEC and point-binding satisfaction imply strict decomposition output.
  - Prelude/transcript/prior-link/accumulator/counter/output satisfaction
    jointly decodes every ProjectedChecks field.
  - Full coefficient equality bridges projected PiRLC to native NIFS; otherwise
    CIR-SOUND returns the named bounded BadRoot event.
  - Valid decoded checks construct a satisfying witness for completeness.
```
