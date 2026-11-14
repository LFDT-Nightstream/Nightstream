import Lean
import NightstreamFPrime.Spec.Compression.Rust.Layout
import NightstreamFPrime.Spec.Compression.Rust.Transcript

/-! The compression parity file. For each production claim set, Lean's layout (`layoutNew`), the
exact error sum that bounds the soundness error of the argument (`runError (productionCounts …)`,
`runCost_le_runError`, `error_le_runError`), its count over `q²` (`runCount`) and the bits of the
Rust gate (`runBits`). For generated claim sets, Lean's layout or the error of `layoutNew`, and the
same values with the production `m` and `t`. Rust tests compare them with `Layout::new` and
`error_bits` (`crates/nightstream/tests/bounded_depth_lifecycle/compression_error.rs`,
`crates/ns-compress/tests/unit/layout.rs`). -/

namespace NightstreamFPrime.Export.CompressionParity

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Compression.Rust
open NightstreamFPrime.Spec.Compression.Verifier (Slice fieldOrder)
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)

/-- The ring blocks of the light and the refresh relation of the reference application (Rust
`Relation::blocks`). The Rust test fails when the relations have other counts. -/
def productionBlocks : Fin 2 → ℕ := ![74530, 497376]

private def slice (s : Slice) : Lean.Json := Lean.toJson [s.offset, s.vars]

private def layout (l : Placed) : Lean.Json :=
  Lean.Json.mkObj [
    ("cells", Lean.Json.arr (l.cells.map fun c => Lean.Json.mkObj [
      ("witness", slice c.witness),
      ("blocks", Lean.toJson c.blocks),
      ("table", match c.table with
        | some (bound, counts) => Lean.Json.arr #[Lean.toJson bound, slice counts]
        | none => Lean.Json.null)]).toArray),
    ("key_vars", Lean.toJson l.keyVars),
    ("witness_vars", Lean.toJson l.witnessVars),
    ("witness_len", Lean.toJson l.witnessLen),
    ("span_vars", Lean.toJson l.spanVars),
    ("span_len", Lean.toJson l.spanLen),
    ("inverse_vars", Lean.toJson l.inverseVars)]

private def error (c : RunCounts) : Lean.Json :=
  Lean.Json.mkObj [
    ("numerator", Lean.toJson (toString (runError c).num)),
    ("denominator", Lean.toJson (toString (runError c).den)),
    ("count", Lean.toJson (runCount c)),
    ("bits", Lean.toJson (runBits c))]

private def production (pc : Fin 2) (d0 d1 : ℕ) : Lean.Json :=
  let depths : Fin 2 → ℕ := ![d0, d1]
  let keyBlocks := (keyBlocksFor (max (productionBlocks 0) (productionBlocks 1))).getD 0
  let entry := [("pc", Lean.toJson ((pc : ℕ) + 1)), ("depths", Lean.toJson [d0, d1])]
  match layoutNew (productionClaims productionBlocks pc depths) keyBlocks with
  | .ok l => Lean.Json.mkObj (entry ++ [("layout", layout l),
      ("error", error (productionCounts l.production depths))])
  | .error e => Lean.Json.mkObj (entry ++ [("rejected", Lean.toJson e.name)])

/-- One step of Knuth's MMIX linear congruential generator. -/
private def step (state : ℕ) : ℕ := (6364136223846793005 * state + 1442695040888963407) % 2 ^ 64

/-- A value below `n` from the high bits of the next state, and that state. -/
private def draw (state n : ℕ) : ℕ × ℕ :=
  let next := step state
  (next / 2 ^ 32 % n, next)

/-- A generated claim under a key of `key` blocks: empty, above the key, small, any size or a
power of two, with no table or a table at the edges of the code width. -/
private def claim (key state : ℕ) : Claim × ℕ :=
  let (kind, state) := draw state 20
  let (size, state) := draw state (max key 64)
  let (power, state) := draw state (Nat.log 2 (max key 1) + 1)
  let blocks := if kind = 0 then 0 else if kind = 1 then key + 1 else if kind = 2 then key
    else if kind < 10 then 1 + size % 64 else if kind < 16 then 1 + size % max key 1 else 2 ^ power
  let (norm, state) := draw state 3
  let (edge, state) := draw state 10
  let (bound, state) := draw state 3000
  let table := if norm = 0 then none else some (if edge = 0 then 0 else if edge = 1 then 1
    else if edge = 2 then 32767 else if edge = 3 then 32768 else bound)
  (⟨blocks, table⟩, state)

private def claims (key : ℕ) : ℕ → ℕ → List Claim
  | 0, _ => []
  | count + 1, state => (claim key state).1 :: claims key count (claim key state).2

/-- Generated claim set `index`: the key, then up to six claims. -/
private def generated (index : ℕ) : List Claim × ℕ :=
  let (choice, state) := draw (step (index + 1)) 10
  let key := if choice < 7 then 512 else if choice < 9 then 524288 else 1000
  let (count, state) := draw state 7
  (claims key count state, key)

/-- A layout whose `extensionTerms` are `512`, so its exact error sum is above `2^−119` and
`runBits` is 118: nine claims, one small table, and `W` of 27 variables under the key of `2^19`
blocks. -/
private def powerOfTwo : List Claim × ℕ :=
  (⟨1, some 0⟩ :: ⟨2 ^ 19, none⟩ :: ⟨2 ^ 19, none⟩ :: List.replicate 6 ⟨1, none⟩, 2 ^ 19)

private def generatedEntry (input : List Claim × ℕ) : Lean.Json :=
  let entry := [("claims", Lean.Json.arr (input.1.map fun c => Lean.Json.arr #[Lean.toJson c.blocks,
      match c.table with | some bound => Lean.toJson bound | none => Lean.Json.null]).toArray),
    ("key_blocks", Lean.toJson input.2)]
  match layoutNew input.1 input.2 with
  | .ok l => Lean.Json.mkObj (entry ++ [("layout", layout l),
      ("error", error (l.counts 320 4))])
  | .error e => Lean.Json.mkObj (entry ++ [("rejected", Lean.toJson e.name)])

/-- The claims of the proof of `crates/ns-compress/tests/unit/transcript_order.rs`
(`tests/support::instances`): their blocks and tables, under the key of 512 blocks, each with 128
public cells, 14 point variables and 4 matrices. -/
private def transcriptClaims : List Claim := [⟨61, some 2176⟩, ⟨100, some 300⟩, ⟨90, none⟩]

private def call : Call → Lean.Json
  | .absorb words => Lean.Json.arr #[Lean.toJson "absorb", Lean.toJson words]
  | .bytes => Lean.Json.arr #[Lean.toJson "bytes"]
  | .draw count => Lean.Json.arr #[Lean.toJson "draw", Lean.toJson count]

private def transcript : Lean.Json :=
  match layoutNew transcriptClaims 512 with
  | .error e => Lean.Json.mkObj [("rejected", Lean.toJson e.name)]
  | .ok l =>
    let shape : Shape :=
      { claims := transcriptClaims.map fun _ => (128, 14, 4)
        tables := (transcriptClaims.filter fun c => c.table.isSome).length
        spanVars := l.spanVars
        witnessVars := l.witnessVars
        keyBlockVars := l.keyVars - laneVars - rowVars }
    Lean.Json.mkObj [
      ("claims", Lean.Json.arr (transcriptClaims.map fun c => Lean.Json.arr #[Lean.toJson c.blocks,
        match c.table with | some bound => Lean.toJson bound | none => Lean.Json.null,
        Lean.toJson (128 : ℕ), Lean.toJson (14 : ℕ), Lean.toJson (4 : ℕ)]).toArray),
      ("key_blocks", Lean.toJson (512 : ℕ)),
      ("steps", Lean.Json.arr ((schedule shape).map fun step => Lean.Json.mkObj [
        ("label", Lean.toJson step.1.name),
        ("move", Lean.toJson (if step.1.move = .draw then "draw" else "recv")),
        ("calls", Lean.Json.arr (step.2.map call).toArray)]).toArray)]

def fixture : Lean.Json :=
  let depths := List.range (productionLightFoldCap + 1)
  Lean.Json.mkObj [
    ("schema", Lean.toJson (1 : ℕ)),
    ("modulus", Lean.toJson fieldOrder),
    ("lane_vars", Lean.toJson laneVars),
    ("row_vars", Lean.toJson rowVars),
    ("lookup_vars", Lean.toJson lookupVars),
    ("schedules", Lean.Json.arr (schedules.map fun s => Lean.Json.mkObj [
      ("key_vars", Lean.toJson s.keyVars),
      ("rows", Lean.toJson (s.rows.map fun r => [r.1, r.2]))]).toArray),
    ("blocks", Lean.toJson [productionBlocks 0, productionBlocks 1]),
    ("key_blocks", Lean.toJson ((keyBlocksFor (max (productionBlocks 0) (productionBlocks 1))).getD 0)),
    ("public_cells", Lean.toJson (320 : ℕ)),
    ("matrices", Lean.toJson (4 : ℕ)),
    ("bounds", Lean.toJson (depths.map productionGlobalParams.accumulatorBound)),
    ("production", Lean.Json.arr (([0, 1] : List (Fin 2)).flatMap (fun pc =>
      depths.flatMap fun d0 => depths.map fun d1 => production pc d0 d1)).toArray),
    ("transcript", transcript),
    ("generated", Lean.Json.arr ((powerOfTwo :: (List.range 3000).map generated).map
      generatedEntry).toArray)]

end NightstreamFPrime.Export.CompressionParity
