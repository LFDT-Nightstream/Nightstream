import NightstreamFPrime.Spec.Compression.Akita.Rows

/-! Each Akita schedule file of `crates/ns-compress/artifacts/` holds exactly its Lean rows
(`Spec/Compression/Akita/Rows.lean`), in the fields that `Row` names. Lake does not track the
files, so `scripts/validate.sh` runs this file again. -/

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

def holds (path : System.FilePath) (rows : List Row) : IO Unit := do
  let text ← IO.FS.readFile path
  let found ← IO.ofExcept do
    (← (← (← Json.parse text).getObjVal? "rows").getArr?).toList.mapM row
  unless found == rows do throw <| IO.userError s!"{path} does not hold the Lean rows"

#eval holds "../../crates/ns-compress/artifacts/key20.aks" [key20]
#eval holds "../../crates/ns-compress/artifacts/key30.aks" [key30W27, key30W26]

end AkitaSchedules
