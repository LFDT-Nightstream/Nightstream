import CompressionLean.Circuit.Block

/-!
The digest of `poseidonTemplate` that `crates/neo-spartan/tests/internal/circuit.rs`
pins (`TEMPLATE_DIGEST`). A change to the Lean template fails here; a change
to the Rust template fails there. Recompute: in `formal/compression-lean`,
run `lake env lean CompressionLeanTests/BlockDigest.lean`; it prints a changed
value.
-/

namespace CompressionLean.Circuit.Block

open NightstreamFPrime.Spec

/-- The sizes, the input count, the output cells and the entry count, then
each entry's matrix, row, cell (`0` for the constant one, else `cell + 1`)
and coefficient. -/
def templateWords : List F :=
  ([poseidonTemplate.rows, poseidonTemplate.cells, 16, output, output + 16,
      poseidonTemplate.entries.length].map Poseidon2.ofNat) ++
    poseidonTemplate.entries.flatMap fun e =>
      [Poseidon2.ofNat e.matrix.val, Poseidon2.ofNat e.row,
        Poseidon2.ofNat ((e.cell.map (· + 1)).getD 0), e.coef]

/-- The Poseidon2 sponge hash of `templateWords`, as canonical words. -/
def templateDigest : List ℕ := (Poseidon2.hash templateWords).map Fin.val

end CompressionLean.Circuit.Block

/-- info: [9961535152869634148, 8036698533330372018, 14857419453680543167, 17712442702339023467] -/
#guard_msgs in
#eval CompressionLean.Circuit.Block.templateDigest
