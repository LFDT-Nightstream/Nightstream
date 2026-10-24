import CompressionLean.Circuit.Record
import CompressionLean.Replay.Transcript
import CompressionLean.Fields

/-!
Owns the run digest that `lowerCheck` and `terminalCheck` print and that Rust
pins (`run_digest`, `crates/neo-spartan/src/shrink/mod.rs`; `Trace::digest`,
`circuit/record.rs`).
-/

namespace CompressionLean.Circuit.Digest

open CompressionLean.Shrink NightstreamFPrime.Spec
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

end CompressionLean.Circuit.Digest
