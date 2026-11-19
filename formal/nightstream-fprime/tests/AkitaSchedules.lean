import NightstreamFPrime.Spec.Compression.Akita.Rows
import NightstreamFPrime.Spec.Compression.Akita.Shapes

/-! Each Akita schedule file of `crates/ns-compress/artifacts/` holds exactly its Lean rows
(`Spec/Compression/Akita/Rows.lean`), in the fields that `Row` names, and the key group of each
of its rows has the numbers of its Lean shape (`Spec/Compression/Akita/Shapes.lean`). Lake does
not track the files, so `scripts/validate.sh` runs this file again. -/

namespace AkitaSchedules

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita

def field (value : Json) (path : List String) : Except String Json :=
  path.foldlM (·.getObjVal? ·) value

def named {α : Type} (value : Json) (names : List (String × α)) : Except String α :=
  match names.find? (value == .str ·.1) with
  | some (_, found) => pure found
  | none => throw s!"{value} is not one of {names.map (·.1)}"

def group (value : Json) : Except String Group := do
  let method ← field value ["opening", "opening_method"]
  let route ← field value ["profile", "inner", "matrix", "security_route"]
  return {
    setup := !(← value.getObjVal? "setup_natural_len").isNull
    opening := ← (fun _ => Opening.packing) <$> method.getObjVal? "SubringCoefficientPacking" <|>
      named method [("EvaluationTrace", .trace)]
    norm := ← (fun _ => Norm.coefficient) <$> route.getObjVal? "Linf" <|>
      (fun _ => Norm.euclidean) <$> route.getObjVal? "L2" }

def level (value : Json) : Except String Level := do
  return {
    groups := (← (← (← field value ["params", "groups", "entries"]).getArr?).mapM group).toList
    compressed := ← named (← field value ["params", "payload_mode"])
      [("Compressed", true), ("Raw", false)]
    ringCheck := ← named (← field value ["params", "ring_relation_mode"])
      [("QuotientLift", RingCheck.quotientLift), ("ReducedEvaluation", .quotientFree)]
    chunks := ← (← field value ["params", "witness_chunk", "num_chunks"]).getNat?
    source := ← (← value.getObjVal? "input_witness_len").getNat?
    target := ← (← value.getObjVal? "output_witness_len").getNat? }

def row (value : Json) : Except String Row := do
  let folds ← (← field value ["schedule", "recursive_folds"]).getArr?
  return {
    levels := ← ((← field value ["schedule", "root"]) :: folds.toList).mapM level
    terminal := ← (← field value ["schedule", "terminal", "input_witness_len"]).getNat? }

/-- The numbers of a shape in the fields of a key group, with the widths of `A` and `B` and the
count of cells of the table. -/
def numbers (S : Shape) : Bool × List ℕ :=
  (S.chain.isSome, [S.innerDegree, S.outerDegree, S.blocks, S.blockLength, S.sourceBase,
    S.sourceDigits, S.innerRank, S.openingBase, S.openingDigits, S.certifiedBase, S.outerRank,
    S.slices, S.blockLength * S.sourceDigits,
    S.sliceLength * S.innerRank * S.ratio * S.openingDigits,
    S.blocks * S.blockLength * S.innerDegree])

def keyGroup (value : Json) : Except String (Bool × List ℕ) := do
  let params ← field value ["schedule", "root", "params"]
  let groups ← (← field params ["groups", "entries"]).getArr?
  let key ← (groups[0]?.map (·.getObjVal? "profile")).getD (throw "no key group")
  let read (path : List String) : Except String ℕ := do (← field key path).getNat?
  let outer := ["outer", "matrix", "sis_table_key"]
  return (← named (← params.getObjVal? "payload_mode") [("Compressed", true), ("Raw", false)],
    [← read ["inner", "matrix", "security_route", "Linf", "ring_dimension"],
      ← read (outer ++ ["ring_dimension"]),
      ← read ["blocks", "live_blocks"],
      ← read ["blocks", "positions_per_block"],
      2 ^ (← read ["inner", "digits", "log_basis"]),
      ← read ["inner", "digits", "num_digits"],
      ← read ["inner", "matrix", "output_rank"],
      2 ^ (← read ["outer", "digits", "log_basis"]),
      ← read ["outer", "digits", "num_digits"],
      (← read (outer ++ ["coeff_linf_bound"])) + 1,
      ← read ["outer", "matrix", "output_rank"],
      ← read ["outer_slice_count"],
      ← read ["inner", "matrix", "input_width"],
      ← read ["outer", "matrix", "input_width"],
      2 ^ (← read ["group", "num_vars"])])

def holds (path : System.FilePath) (rows : List Row) (shape : Shape) : IO Unit := do
  let text ← IO.FS.readFile path
  let entries ← IO.ofExcept do (← (← Json.parse text).getObjVal? "rows").getArr?
  unless (← IO.ofExcept (entries.toList.mapM row)) == rows do
    throw <| IO.userError s!"{path} does not hold the Lean rows"
  unless (← IO.ofExcept (entries.toList.mapM keyGroup)).all (· == numbers shape) do
    throw <| IO.userError s!"{path} does not hold the Lean shape of the key group"

#eval holds "../../crates/ns-compress/artifacts/key20.aks" [key20] Shape.key20
#eval holds "../../crates/ns-compress/artifacts/key30.aks" [key30W27, key30W26] Shape.key30

end AkitaSchedules
