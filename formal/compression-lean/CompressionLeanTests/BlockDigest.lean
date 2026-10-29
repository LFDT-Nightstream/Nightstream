import CompressionLean.Circuit.Block

/-! Rust pins this digest (`TEMPLATE_DIGEST`, `crates/neo-spartan/tests/internal/circuit.rs`). -/

namespace CompressionLean.Circuit.Block

open NightstreamFPrime.Spec

def templateWords : List F :=
  ([poseidonTemplate.rows, poseidonTemplate.cells, 16, output, output + 16,
      poseidonTemplate.entries.length].map Poseidon2.ofNat) ++
    poseidonTemplate.entries.flatMap fun e =>
      [Poseidon2.ofNat e.matrix.val, Poseidon2.ofNat e.row,
        Poseidon2.ofNat ((e.cell.map (· + 1)).getD 0), e.coef]

def templateDigest : List ℕ := (Poseidon2.hash templateWords).map Fin.val

end CompressionLean.Circuit.Block

/-- info: [9961535152869634148, 8036698533330372018, 14857419453680543167, 17712442702339023467] -/
#guard_msgs in
#eval CompressionLean.Circuit.Block.templateDigest
