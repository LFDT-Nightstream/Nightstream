import CompressionLean.Circuit.Record
import CompressionLean.Replay.Transcript
import NightstreamFPrime.Spec.GoldilocksPrime

/-! Rust pins this digest (`run_digest`, `Trace::digest`). -/

namespace CompressionLean.Circuit.Digest

open CompressionLean.Shrink NightstreamFPrime.Spec
open CompressionLean.Replay (Gl)

instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

abbrev G := ZMod goldilocksModulus

def runDigest (run : Run G) (rows : Array (GlueRow G)) : Array Gl :=
  let wireWords : Wire → Array ℕ
    | .glue cell => #[0, cell]
    | .block b cell => #[1, b, cell]
    | .pub index => #[2, index]
  let sideWords (side : List (Wire × G)) : Array ℕ :=
    side.foldl (fun out (w, c) => (out ++ wireWords w).push c.val) #[side.length]
  let words := rows.foldl (init := #[run.cells, run.blocks, run.publics.length, rows.size]) fun out row =>
    (List.finRange 7).foldl (fun out s => out ++ sideWords (row.side s)) out
  Replay.hashLeaf (words.map Gl.ofNat)

end CompressionLean.Circuit.Digest
