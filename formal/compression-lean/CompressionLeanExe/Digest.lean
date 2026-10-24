import CompressionLean.Circuit.Record
import CompressionLean.Circuit.Block
import CompressionLean.Replay.Transcript
import CompressionLean.Fields

/-!
Owns the digests that Rust pins: the run digest that `lowerCheck` and
`terminalCheck` print (`run_digest`, `crates/neo-spartan/src/shrink/mod.rs`;
`Trace::digest`, `circuit/record.rs`), and the digest of the permutation
template (`TEMPLATE_DIGEST`, `crates/neo-spartan/tests/internal/circuit.rs`),
which `CompressionLeanTests/BlockDigest.lean` pins.
-/

namespace CompressionLeanExe.Digest

open CompressionLean CompressionLean.Shrink NightstreamFPrime.Spec
open CompressionLean.Replay (Word)

/-- The run digest (`Trace::digest`): Poseidon2's leaf sponge
(`Replay.hashLeaf`) of the glue cell, block, statement word and row counts,
then every row's seven sides, each as its term count and, per term, the wire
(`0 cell`, `1 block cell` or `2 index`) and the coefficient. The statement
and cell values are not part of it. -/
def runDigest (run : Run Gl) (rows : Array (GlueRow Gl)) : Array Word :=
  let wireWords : Wire → Array ℕ
    | .glue cell => #[0, cell]
    | .block b cell => #[1, b, cell]
    | .pub index => #[2, index]
  let sideWords (side : List (Wire × Gl)) : Array ℕ :=
    side.foldl (fun out (w, c) => (out ++ wireWords w).push c.val) #[side.length]
  let words := rows.foldl (init := #[run.cells, run.blocks, run.publics.length, rows.size]) fun out row =>
    (List.finRange 7).foldl (fun out s => out ++ sideWords (row.side s)) out
  Replay.hashLeaf (words.map Word.ofNat)

open CompressionLean.Circuit.Block in
/-- The sizes, the input count, the output cells and the entry count of
`poseidonTemplate`, then each entry's matrix, row, cell (`0` for the constant
one, else `cell + 1`) and coefficient. -/
def templateWords : List F :=
  ([poseidonTemplate.rows, poseidonTemplate.cells, 16, output, output + 16,
      poseidonTemplate.entries.length].map Poseidon2.ofNat) ++
    poseidonTemplate.entries.flatMap fun e =>
      [Poseidon2.ofNat e.matrix.val, Poseidon2.ofNat e.row,
        Poseidon2.ofNat ((e.cell.map (· + 1)).getD 0), e.coef]

/-- The Poseidon2 sponge hash of `templateWords`, as canonical words. -/
def templateDigest : List ℕ := (Poseidon2.hash templateWords).map Fin.val

end CompressionLeanExe.Digest
