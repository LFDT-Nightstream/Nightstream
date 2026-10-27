import CompressionLeanExe.Digest

/-!
The digest of `poseidonTemplate` (`templateDigest`) that
`crates/neo-spartan/tests/internal/circuit.rs` pins (`TEMPLATE_DIGEST`). A
change to the Lean template fails here; a change to the Rust template fails
there. Recompute: in `formal/compression-lean`, run
`lake env lean CompressionLeanTests/BlockDigest.lean`; it prints a changed
value.
-/

/-- info: [9961535152869634148, 8036698533330372018, 14857419453680543167, 17712442702339023467] -/
#guard_msgs in
#eval CompressionLeanExe.Digest.templateDigest
