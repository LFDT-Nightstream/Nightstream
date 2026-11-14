import Lean
import NightstreamFPrime.Spec.Compression.Rust.Layout

/-! The compression parity file. For each production claim set, Lean's layout (`layoutNew`) and the
exact value of Rust `error_bits` that bounds the soundness error of the argument
(`runError (productionCounts …)`, `runCost_le_runError`, `error_le_runError`). Rust tests compare
them with `Layout::new` and `error_bits` (`crates/nightstream/tests/bounded_depth_lifecycle/`). -/

namespace NightstreamFPrime.Export.CompressionParity

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Compression.Rust
open NightstreamFPrime.Spec.Compression.Verifier (Slice fieldOrder)
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)

/-- The ring blocks of the light and the refresh relation of the reference application (Rust
`Relation::blocks`). The Rust test fails when the relations have other counts. -/
def productionBlocks : Fin 2 → ℕ := ![74530, 497376]

/-- The largest `m` with `value ≤ 2^−m`, for `0 < value ≤ 1`: the largest security minimum that
the exact value meets. -/
def floorBits (value : ℚ) : ℕ := Nat.log 2 (value.den / value.num.toNat)

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

private def error (value : ℚ) : Lean.Json :=
  Lean.Json.mkObj [
    ("numerator", Lean.toJson (toString value.num)),
    ("denominator", Lean.toJson (toString value.den)),
    ("bits", Lean.toJson (floorBits value))]

private def production (pc : Fin 2) (d0 d1 : ℕ) : Lean.Json :=
  let depths : Fin 2 → ℕ := ![d0, d1]
  let keyBlocks := (keyBlocksFor (max (productionBlocks 0) (productionBlocks 1))).getD 0
  let entry := [("pc", Lean.toJson ((pc : ℕ) + 1)), ("depths", Lean.toJson [d0, d1])]
  match layoutNew (productionClaims productionBlocks pc depths) keyBlocks with
  | .ok l => Lean.Json.mkObj (entry ++ [("layout", layout l),
      ("error", error (runError (productionCounts l.production depths)))])
  | .error e => Lean.Json.mkObj (entry ++ [("rejected", Lean.toJson (repr e).pretty)])

def fixture : Lean.Json :=
  let depths := List.range (productionLightFoldCap + 1)
  Lean.Json.mkObj [
    ("schema", Lean.toJson (1 : ℕ)),
    ("modulus", Lean.toJson fieldOrder),
    ("schedules", Lean.Json.arr (schedules.map fun s => Lean.Json.mkObj [
      ("key_vars", Lean.toJson s.keyVars),
      ("rows", Lean.toJson (s.rows.map fun r => [r.1, r.2]))]).toArray),
    ("blocks", Lean.toJson [productionBlocks 0, productionBlocks 1]),
    ("key_blocks", Lean.toJson ((keyBlocksFor (max (productionBlocks 0) (productionBlocks 1))).getD 0)),
    ("public_cells", Lean.toJson (320 : ℕ)),
    ("matrices", Lean.toJson (4 : ℕ)),
    ("bounds", Lean.toJson (depths.map productionGlobalParams.accumulatorBound)),
    ("production", Lean.Json.arr (([0, 1] : List (Fin 2)).flatMap (fun pc =>
      depths.flatMap fun d0 => depths.map fun d1 => production pc d0 d1)).toArray)]

end NightstreamFPrime.Export.CompressionParity
