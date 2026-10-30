import CompressionLean.Ports.Circuit.Record
import CompressionLean.Ports.Circuit.Block
import CompressionLean.Ports.Replay.Transcript
import CompressionLean.Fields

/-! Rust pins these digests (`run_digest`, `Trace::digest`, `TEMPLATE_DIGEST`). -/

namespace CompressionLeanExe.Digest

open CompressionLean CompressionLean.Shrink NightstreamFPrime.Spec
open CompressionLean.Replay (Word)

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
def templateWords : List F :=
  ([poseidonTemplate.rows, poseidonTemplate.cells, 16, output, output + 16,
      poseidonTemplate.entries.length].map Poseidon2.ofNat) ++
    poseidonTemplate.entries.flatMap fun e =>
      [Poseidon2.ofNat e.matrix.val, Poseidon2.ofNat e.row,
        Poseidon2.ofNat ((e.cell.map (· + 1)).getD 0), e.coef]

def templateDigest : List ℕ := (Poseidon2.hash templateWords).map Fin.val

end CompressionLeanExe.Digest
