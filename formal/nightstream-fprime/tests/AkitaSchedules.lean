import NightstreamFPrime.Spec.Compression.Akita.Key20Instances
import NightstreamFPrime.Spec.Compression.Akita.Rows
import NightstreamFPrime.Spec.Compression.Akita.Shapes

/-! Each Akita schedule file of `crates/ns-compress/artifacts/` holds exactly its Lean rows
(`Spec/Compression/Akita/Rows.lean`), in the fields that `Row` names, and the key group of each
of its rows has the numbers of its Lean shape and of its Lean opening descriptor
(`Spec/Compression/Akita/Shapes.lean`). Each group of each level of `key20.aks` and its terminal
have the numbers of `Spec/Compression/Akita/Key20.lean`. Each matrix `A`, `B` and `D` of
`key20.aks` and `A` of its terminal is below an instance of the list `Key20.instances`
(`Spec/Compression/Akita/Key20Instances.lean`), and each instance of the list after the two
compression maps has the degree, the rank, the width, the norm and the priced bound of a matrix
of the file. The file does not hold the compression maps. Lake does not track the files, so
`scripts/validate.sh` runs this file again. -/

namespace AkitaSchedules

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita

def field (value : Json) (path : List String) : Except String Json :=
  path.foldlM (·.getObjVal? ·) value

def nat (value : Json) (path : List String) : Except String ℕ := do (← field value path).getNat?

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
    chunks := ← nat value ["params", "witness_chunk", "num_chunks"]
    source := ← nat value ["input_witness_len"]
    target := ← nat value ["output_witness_len"] }

def row (value : Json) : Except String Row := do
  let folds ← (← field value ["schedule", "recursive_folds"]).getArr?
  return {
    levels := ← ((← field value ["schedule", "root"]) :: folds.toList).mapM level
    terminal := ← nat value ["schedule", "terminal", "input_witness_len"] }

/-- The numbers of a shape and of an opening descriptor in the fields of a group, with the widths
of `A` and `B`. -/
def numbers (S : Shape) (O : Descriptor) : List ℕ :=
  [S.innerDegree, S.outerDegree, S.blocks, S.blockLength, S.sourceBase, S.sourceDigits,
    S.innerRank, S.openingBase, S.openingDigits, S.certifiedBase, S.outerRank, S.slices,
    S.blockLength * S.sourceDigits, S.sliceLength * S.innerRank * S.ratio * S.openingDigits,
    O.challengeMass, O.responseBase, O.responseDigits]

/-- The ring degree of `A` and the numbers of its norm route in the fields of an inner matrix:
the bound of the coefficient route, or `S_max`, the count of the coordinates of the direct norm
check and the squared bound of the Euclidean route. -/
def route (inner : Json) : Except String (ℕ × List ℕ) := do
  let route ← field inner ["matrix", "security_route"]
  match route.getObjVal? "Linf" with
  | .ok linf => return (← nat linf ["ring_dimension"], [← nat linf ["coeff_linf_bound"]])
  | .error _ =>
      let l2 ← route.getObjVal? "L2"
      return (← nat l2 ["table_key", "ring_dimension"], [← nat l2 ["response_l2_sq_cap"],
        ← nat l2 ["norm_proof_shape", "Direct", "physical_response_len"],
        ← nat l2 ["table_key", "collision_l2_sq"]])

/-- The fields of a group for `numbers`, the count of its variables and the numbers of its norm
route. -/
def groupNumbers (group : Json) : Except String (List ℕ × ℕ × List ℕ) := do
  let key ← group.getObjVal? "profile"
  let outer := ["outer", "matrix", "sis_table_key"]
  let challenge := ["opening", "fold_challenge_config"]
  let (degree, norm) ← route (← key.getObjVal? "inner")
  return ([degree,
      ← nat key (outer ++ ["ring_dimension"]),
      ← nat key ["blocks", "live_blocks"],
      ← nat key ["blocks", "positions_per_block"],
      2 ^ (← nat key ["inner", "digits", "log_basis"]),
      ← nat key ["inner", "digits", "num_digits"],
      ← nat key ["inner", "matrix", "output_rank"],
      2 ^ (← nat key ["outer", "digits", "log_basis"]),
      ← nat key ["outer", "digits", "num_digits"],
      (← nat key (outer ++ ["coeff_linf_bound"])) + 1,
      ← nat key ["outer", "matrix", "output_rank"],
      ← nat key ["outer_slice_count"],
      ← nat key ["inner", "matrix", "input_width"],
      ← nat key ["outer", "matrix", "input_width"],
      (← nat group (challenge ++ ["count_pm1"])) + 2 * (← nat group (challenge ++ ["count_pm2"])),
      2 ^ (← nat group ["opening", "log_basis_open"]),
      ← nat group ["opening", "num_digits_fold"]],
    ← nat key ["group", "num_vars"], norm)

def compressed (params : Json) : Except String Bool := do
  named (← params.getObjVal? "payload_mode") [("Compressed", true), ("Raw", false)]

/-- The key group of a row: the compression of its payload, its numbers, the count of cells of
its table and the bound of `A`. -/
def keyGroup (value : Json) : Except String (Bool × List ℕ) := do
  let params ← field value ["schedule", "root", "params"]
  let groups ← (← field params ["groups", "entries"]).getArr?
  let (shape, vars, norm) ← groupNumbers (← (groups[0]?.map pure).getD (throw "no key group"))
  return (← compressed params, shape ++ 2 ^ vars :: norm)

def rows (path : System.FilePath) : IO (List Json) := do
  let text ← IO.FS.readFile path
  return (← IO.ofExcept do (← (← Json.parse text).getObjVal? "rows").getArr?).toList

def holds (path : System.FilePath) (expected : List Row) (shape : Shape)
    (descriptor : Descriptor) : IO Unit := do
  let entries ← rows path
  unless (← IO.ofExcept (entries.mapM row)) == expected do
    throw <| IO.userError s!"{path} does not hold the Lean rows"
  unless (← IO.ofExcept (entries.mapM keyGroup)).all (· == (shape.chain.isSome,
      numbers shape descriptor ++ [shape.blocks * shape.blockLength * shape.innerDegree,
        descriptor.innerBound shape])) do
    throw <| IO.userError s!"{path} does not hold the Lean numbers of the key group"

/-- The numbers of the Euclidean route in the fields of `route`. The file has the squared bound
`64·Γ²·S_max` of the Module-SIS instance of `A` as the next power of two. -/
def squared (norm : Key20.Squared) : List ℕ :=
  [norm.limit, 2 ^ norm.vars, 2 ^ Nat.clog 2 (64 * norm.operator ^ 2 * norm.limit)]

/-- The numbers of a group of a level: `numbers`, the ring degree of its fold challenges, the
degree, the rank and the certified base of `D`, the compression of the payloads of `B` and `D`,
the count of variables of a table with its cells, and the numbers of its norm route. -/
def entry (S : Shape) (O : OpeningShape S) (fold : Descriptor) (norm : Option Key20.Squared) :
    List ℕ :=
  numbers S fold ++ [O.carrierDegree, O.openDegree, O.openRank, S.certifiedBase,
    S.chain.isSome.toNat, O.chain.isSome.toNat,
    Nat.clog 2 (S.blocks * S.blockLength * S.innerDegree)] ++
    (norm.map squared).getD [fold.innerBound S]

/-- The count of the columns of `D` for a group. -/
def columns (S : Shape) (O : OpeningShape S) : ℕ := S.blocks * O.subcolumns * S.openingDigits

/-- The count of cells of the next witness of a level with quotients: the cells of the flat
witness and `δ_quot` cells for each coefficient of each ring row, in ring elements of the degree
`next` of the next level. `degrees` is the sum of the degrees of the ring rows of the level. -/
def lifted (cells degrees next : ℕ) : ℕ :=
  (cells + Key20.quotientDigits * degrees + next - 1) / next * next

/-- The numbers of the levels of `Key20.lean`: for each level the numbers of its groups, the
width of `D`, the count of cells of its next witness and its count `μ` of variables. -/
def tower : List (List (List ℕ) × List ℕ) :=
  open Key20 in
  let groups := List.finRange 3
  let trace (L : Key20.Level) (norm : Option Squared) :=
    ([entry L.shape L.opening L.fold norm],
      [columns L.shape L.opening, L.opening.cells L.fold, L.vars])
  [(groups.map fun g => entry (root.shape g) (root.opening g) (root.fold g) none,
      [(groups.map fun g => columns (root.shape g) (root.opening g)).sum,
        lifted root.cells root.degreeSum level₁.shape.innerDegree, rootVars]),
    ([entry level₁.shape level₁.opening level₁.fold none],
      [columns level₁.shape level₁.opening,
        lifted (level₁.opening.cells level₁.fold) level₁.opening.degreeSum
          level₂.shape.innerDegree, level₁.vars]),
    trace level₂ none, trace level₃ norm₃, trace level₄ norm₄, trace level₅ norm₅]

/-- The numbers of the terminal of `Key20.lean`. -/
def terminal : List ℕ :=
  let S := Key20.terminal
  [S.blocks, S.blockLength, S.sourceBase, S.sourceDigits, S.innerRank, S.innerDegree,
    S.blockLength * S.sourceDigits] ++ squared Key20.normTerminal

/-- The fields of a group of a level for `entry`. The fold challenges of a level with evaluation
trace are in the ring of `A`. -/
def fileEntry (params group : Json) : Except String (List ℕ) := do
  let (shape, vars, norm) ← groupNumbers group
  let method ← field group ["opening", "opening_method"]
  let matrix := ["open_matrix", "sis_table_key"]
  let payload := (← compressed params).toNat
  return shape ++ [
    ← nat method ["SubringCoefficientPacking", "challenge_subring_dimension"] <|>
      pure (shape.headD 0),
    ← nat params (matrix ++ ["ring_dimension"]),
    ← nat params ["open_matrix", "output_rank"],
    (← nat params (matrix ++ ["coeff_linf_bound"])) + 1,
    payload, payload, vars] ++ norm

/-- The fields of the levels of a schedule for `tower`. The count of variables of a level is that
of the group of the next level, and for the last level it is the least count for the cells of the
terminal and a zero cell. -/
def fileTower (schedule : Json) : Except String (List (List (List ℕ) × List ℕ)) := do
  let folds := (← (← schedule.getObjVal? "recursive_folds").getArr?).toList
  let vars ← folds.mapM fun fold => do
    nat (← (← field fold ["params", "groups", "entries"]).getArrVal? 0)
      ["profile", "group", "num_vars"]
  let last := Nat.clog 2 ((← nat schedule ["terminal", "input_witness_len"]) + 1)
  (((← schedule.getObjVal? "root") :: folds).zip (vars ++ [last])).mapM fun (level, next) => do
    let params ← level.getObjVal? "params"
    let groups ← (← field params ["groups", "entries"]).getArr?
    return (← groups.toList.mapM (fileEntry params),
      [← nat params ["open_matrix", "input_width"], ← nat level ["output_witness_len"], next])

/-- The fields of a terminal for `terminal`. -/
def fileTerminal (value : Json) : Except String (List ℕ) := do
  let inner ← value.getObjVal? "inner"
  let (degree, norm) ← route inner
  return [← nat value ["blocks", "live_blocks"], ← nat value ["blocks", "positions_per_block"],
    2 ^ (← nat inner ["digits", "log_basis"]), ← nat inner ["digits", "num_digits"],
    ← nat inner ["matrix", "output_rank"], degree, ← nat inner ["matrix", "input_width"]] ++ norm

def towerHolds (path : System.FilePath) : IO Unit := do
  let [entry] ← rows path | throw <| IO.userError s!"{path} does not have one row"
  let schedule ← IO.ofExcept (entry.getObjVal? "schedule")
  let levels ← IO.ofExcept (fileTower schedule)
  unless levels == tower do
    throw <| IO.userError s!"{path} does not hold the Lean numbers of the levels: the file has
      {levels}, and Lean has {tower}"
  let last ← IO.ofExcept do fileTerminal (← schedule.getObjVal? "terminal")
  unless last == terminal do
    throw <| IO.userError s!"{path} does not hold the Lean numbers of the terminal: the file has
      {last}, and Lean has {terminal}"

/-- The numbers `(d, n, m, p, η)` of the Module-SIS instance of a matrix of the file. -/
def matrixNumbers (matrix : Json) (degree : ℕ) (euclidean : Bool) (bound : ℕ) :
    Except String (ℕ × ℕ × ℕ × Bool × ℕ) := do
  return (degree, ← nat matrix ["output_rank"], ← nat matrix ["input_width"], euclidean, bound)

/-- The instance of `A`: the priced bound is the last number of `route`, and the route is
Euclidean when it has three numbers. -/
def innerNumbers (inner : Json) : Except String (ℕ × ℕ × ℕ × Bool × ℕ) := do
  let (degree, norm) ← route inner
  matrixNumbers (← inner.getObjVal? "matrix") degree (norm.length == 3) (norm.getLastD 0)

/-- The instance of `B` or `D`. -/
def keyedNumbers (matrix : Json) : Except String (ℕ × ℕ × ℕ × Bool × ℕ) := do
  matrixNumbers matrix (← nat matrix ["sis_table_key", "ring_dimension"]) false
    (← nat matrix ["sis_table_key", "coeff_linf_bound"])

/-- The instances of the matrices of a schedule: `A_term`, and `D`, `A` and `B` of each group of
each level. -/
def fileInstances (schedule : Json) : Except String (List (ℕ × ℕ × ℕ × Bool × ℕ)) := do
  let folds := (← (← schedule.getObjVal? "recursive_folds").getArr?).toList
  let levels ← ((← schedule.getObjVal? "root") :: folds).mapM fun level => do
    let params ← level.getObjVal? "params"
    let groups ← (← field params ["groups", "entries"]).getArr?
    let matrices ← groups.toList.mapM fun group => do
      return [← innerNumbers (← field group ["profile", "inner"]),
        ← keyedNumbers (← field group ["profile", "outer", "matrix"])]
    return (← keyedNumbers (← params.getObjVal? "open_matrix")) :: matrices.flatten
  return (← innerNumbers (← field schedule ["terminal", "inner"])) :: levels.flatten

/-- `Setup.Instance.Below` on the numbers of two instances. -/
def below : ℕ × ℕ × ℕ × Bool × ℕ → ℕ × ℕ × ℕ × Bool × ℕ → Bool
  | (d, n, m, p, η), (d', n', m', p', η') =>
      d == d' && n == n' && p == p' && decide (m ≤ m') && decide (η ≤ η')

def instancesHold (path : System.FilePath) : IO Unit := do
  let [entry] ← rows path | throw <| IO.userError s!"{path} does not have one row"
  let views ← IO.ofExcept do fileInstances (← entry.getObjVal? "schedule")
  let listed := Key20.instances.map (·.key)
  unless views.all fun view => listed.any (below view) do
    throw <| IO.userError s!"{path} has a matrix that is not below a Lean instance: the file has
      {views}, and Lean has {listed}"
  unless (listed.drop 2).all views.contains do
    throw <| IO.userError s!"{path} does not hold each Lean instance: the file has {views}, and
      Lean has {listed}"

#eval holds "../../crates/ns-compress/artifacts/key20.aks" [key20] Shape.key20 Descriptor.key20
#eval holds "../../crates/ns-compress/artifacts/key30.aks" [key30W27, key30W26] Shape.key30
  Descriptor.key30
#eval towerHolds "../../crates/ns-compress/artifacts/key20.aks"
#eval instancesHold "../../crates/ns-compress/artifacts/key20.aks"

end AkitaSchedules
