import NightstreamFPrime.Spec.Compression.Akita.Key20Instances
import NightstreamFPrime.Spec.Compression.Akita.Key30Instances
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.Key30Opening
import NightstreamFPrime.Spec.Compression.Akita.Rows
import NightstreamFPrime.Spec.Compression.Akita.Shapes

/-! Each Akita schedule file of `crates/ns-compress/artifacts/` holds exactly its Lean rows
(`Spec/Compression/Akita/Rows.lean`), in the fields that `Row` names, and the key group of each
of its rows has the numbers of its Lean shape and of its Lean opening descriptor
(`Spec/Compression/Akita/Shapes.lean`). Each group of each level of `key20.aks` and its terminal
have the numbers of `Spec/Compression/Akita/Key20.lean`, and each group of each level of each row
of `key30.aks` and its terminal have the numbers of `Spec/Compression/Akita/Key30.lean`. The count
`output_witness_len` of a level is the end of its witness in the model with the last alignment of
the Akita code, and each alignment of the code before it is empty (`segmentsAligned`). The count
of rows over `E` of a level and its count `s_rel` are the numbers that the Akita code gives for
the fields of the file (`relation_matrix_row_count` and `relation_row_index_num_vars` in
`crates/akita-params/src/layout/params.rs`): the files do not hold these two numbers. Each
matrix `A`, `B` and `D` of `key20.aks` and `A` of its terminal is below an instance of the list
`Key20.instances` (`Spec/Compression/Akita/Key20Instances.lean`), and each instance of the list
after the two compression maps has the degree, the rank, the width, the norm and the priced bound
of a matrix of the file. The two rows of `key30.aks` have the same comparison with the list
`Key30.instances` (`Spec/Compression/Akita/Key30Instances.lean`). A file does not hold the
compression maps. The counts `count_pm1` and `count_pm2` of each group of each level and of the
terminal are the pair of `Key20.families` and `Key30.families`, the families of the decoders of
the Lean verifier (`Key20.families_code`, `Key30.families_code`). Lake does not track the files, so
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

/-- The numbers of a group of a level: the pair `(n₁, n₂)` of its challenge family, `numbers`,
the ring degree of its fold challenges, the degree, the rank and the certified base of `D`, the
compression of the payloads of `B` and `D`, the count of variables of a table with its cells, and
the numbers of its norm route. -/
def entry (family : ℕ × ℕ) (S : Shape) (O : OpeningShape S) (fold : Descriptor)
    (norm : Option Key20.Squared) : List ℕ :=
  family.1 :: family.2 :: numbers S fold ++ [O.carrierDegree, O.openDegree, O.openRank, S.certifiedBase,
    S.chain.isSome.toNat, O.chain.isSome.toNat,
    Nat.clog 2 (S.blocks * S.blockLength * S.innerDegree)] ++
    (norm.map squared).getD [fold.innerBound S]

/-- The count of the columns of `D` for a group. -/
def columns (S : Shape) (O : OpeningShape S) : ℕ := S.blocks * O.subcolumns * S.openingDigits

/-- The count of cells of the next witness of a level with quotients: the end `quotientRoom` of
the witness of Akita (76) in the model, and then the zeros of the last alignment of the Akita code
(revision `d50f700f`, `crates/akita-params/src/witness.rs` and `witness/tail.rs`), to a multiple
of `align`. -/
def aligned (room align : ℕ) : ℕ := (room + align - 1) / align * align

/-- The least ring degree of the rows of a fold before its compression rows: the block of the
alignments of the Akita code before the compression digits, and the last alignment of a level
with one group. -/
def block {S : Shape} (O : OpeningShape S) : ℕ :=
  min (min O.carrierDegree S.innerDegree) (min S.outerDegree O.openDegree)

/-- The block of a fold with opening groups: the least ring degree of their rows. -/
def groupsBlock {n : ℕ} (B : Batch (Fin n)) : ℕ :=
  (List.finRange n).foldr (fun g least => min least (block (B.opening g))) B.openDegree

/-- The last alignment of a level with two groups or more in the Akita code: the largest `d_A` of
the groups. -/
def groupsAlign {n : ℕ} (B : Batch (Fin n)) : ℕ :=
  (List.finRange n).foldr (fun g most => max most (B.shape g).innerDegree) 0

/-- The Akita code aligns the witness of a level before its compression digits to `block`, before
the digits of the first maps to `first`, and before the digits of the second maps to `second`
(`crates/akita-params/src/witness/tail.rs`). The model has no zeros there, so these positions of
`gaps` are multiples of these degrees. -/
def gapsAligned (G : Gaps) (block first second : ℕ) : Bool :=
  (G.main + G.mainCells) % block == 0 && (G.main + G.mainCells) % first == 0 &&
    (G.middle + G.mainCells + G.middleCells) % second == 0

/-- The alignments of the Akita code before the last one are empty at each level of the two files
with compression digits. The trace levels have no quotient digits (`gaps` with no digit). The
maps have the degrees 64 and 32 (`Key20.maps`). -/
def segmentsAligned : Bool :=
  open Key20 in
  gapsAligned (root.gaps quotientDigits) (groupsBlock root) 64 32 &&
    gapsAligned (level₁.opening.gaps level₁.fold quotientDigits) (block level₁.opening) 64 32 &&
    [level₂, level₃].all (fun L => gapsAligned (L.opening.gaps L.fold 0) (block L.opening) 64 32) &&
    Key30.rows.all (fun r =>
      gapsAligned ((Key30.root r).gaps Key30.quotientDigits₀) (groupsBlock (Key30.root r)) 64 32 &&
        gapsAligned ((Key30.level₁ r).gaps Key30.quotientDigits) (groupsBlock (Key30.level₁ r))
          64 32 &&
        gapsAligned ((Key30.level₂ r).opening.gaps (Key30.level₂ r).fold Key30.quotientDigits)
          (block (Key30.level₂ r).opening) 64 32 &&
        gapsAligned ((Key30.level₃ r).opening.gaps (Key30.level₃ r).fold Key30.quotientDigits)
          (block (Key30.level₃ r).opening) 64 32) &&
    gapsAligned (Key30.level₄.opening.gaps Key30.level₄.fold 0) (block Key30.level₄.opening) 64 32

/-- The numbers of the levels of `Key20.lean`: for each level the numbers of its groups, the
width of `D`, the count of cells of its next witness, its count `μ` of variables, its count of
rows over `E` before the scalar row and its count `s_rel`. -/
def tower : List (List (List ℕ) × List ℕ) :=
  open Key20 in
  let groups := List.finRange 3
  let trace (level : ℕ) (L : Key20.Level) (norm : Option Squared) :=
    ([entry (family level 0) L.shape L.opening L.fold norm],
      [columns L.shape L.opening, L.opening.cells L.fold, L.vars, L.opening.traceRowCount,
        L.batch])
  [(groups.map fun g => entry (family 0 g.val) (root.shape g) (root.opening g) (root.fold g) none,
      [(groups.map fun g => columns (root.shape g) (root.opening g)).sum,
        aligned (root.quotientRoom quotientDigits) (groupsAlign root), rootVars, root.rowCount,
        rootBatch]),
    ([entry (family 1 0) level₁.shape level₁.opening level₁.fold none],
      [columns level₁.shape level₁.opening,
        aligned (level₁.opening.quotientRoom level₁.fold quotientDigits) (block level₁.opening),
        level₁.vars, level₁.opening.rowCount, level₁.batch]),
    trace 2 level₂ none, trace 3 level₃ norm₃, trace 4 level₄ norm₄, trace 5 level₅ norm₅]

/-- The numbers of the levels of a row of `Key30.lean`, in the form of `tower`. -/
def tower30 (r : Key30.Row) : List (List (List ℕ) × List ℕ) :=
  open Key30 in
  let groups {n : ℕ} (level : ℕ) (B : Batch (Fin n)) (digits vars batch : ℕ) :=
    ((List.finRange n).map fun g =>
        entry (family level g.val) (B.shape g) (B.opening g) (B.fold g) none,
      [((List.finRange n).map fun g => columns (B.shape g) (B.opening g)).sum,
        aligned (B.quotientRoom digits) (groupsAlign B), vars, B.rowCount, batch])
  let packing (level : ℕ) (L : Key20.Level) :=
    ([entry (family level 0) L.shape L.opening L.fold none],
      [columns L.shape L.opening,
        aligned (L.opening.quotientRoom L.fold quotientDigits) (block L.opening), L.vars,
        L.opening.rowCount, L.batch])
  let trace (level : ℕ) (L : Key20.Level) (norm : Option Key20.Squared) :=
    ([entry (family level 0) L.shape L.opening L.fold norm],
      [columns L.shape L.opening, L.opening.cells L.fold, L.vars, L.opening.traceRowCount,
        L.batch])
  [groups 0 (root r) quotientDigits₀ rootVars rootBatch,
    groups 1 (level₁ r) quotientDigits vars₁ batch₁,
    packing 2 (level₂ r), packing 3 (level₃ r), trace 4 level₄ none, trace 5 level₅ (norm₅ r),
    trace 6 level₆ norm₆]

/-- The numbers of a terminal in the fields of `fileTerminal`. -/
def terminalNumbers (family : ℕ × ℕ) (S : Shape) (norm : Key20.Squared) : List ℕ :=
  [family.1, family.2, S.blocks, S.blockLength, S.sourceBase, S.sourceDigits, S.innerRank,
    S.innerDegree, S.blockLength * S.sourceDigits] ++ squared norm

/-- The counts `count_pm1` and `count_pm2` of a challenge family of the file. -/
def fileFamily (config : Json) : Except String (List ℕ) := do
  return [← nat config ["count_pm1"], ← nat config ["count_pm2"]]

/-- The fields of a group of a level for `entry`. The fold challenges of a level with evaluation
trace are in the ring of `A`. -/
def fileEntry (params group : Json) : Except String (List ℕ) := do
  let (shape, vars, norm) ← groupNumbers group
  let family ← fileFamily (← field group ["opening", "fold_challenge_config"])
  let method ← field group ["opening", "opening_method"]
  let matrix := ["open_matrix", "sis_table_key"]
  let payload := (← compressed params).toNat
  return family ++ shape ++ [
    ← nat method ["SubringCoefficientPacking", "challenge_subring_dimension"] <|>
      pure (shape.headD 0),
    ← nat params (matrix ++ ["ring_dimension"]),
    ← nat params ["open_matrix", "output_rank"],
    (← nat params (matrix ++ ["coeff_linf_bound"])) + 1,
    payload, payload, vars] ++ norm

/-- The count of rows of the relation matrix of a level in the Akita code
(`relation_matrix_row_count`): for each group the consistency row, the rows of `A` and the rows
of `B` of each slice, then the rows of `D`, and at a level with compression digits two rows for
each group and two rows for `D`. -/
def fileRows (params : Json) : Except String ℕ := do
  let groups ← (← field params ["groups", "entries"]).getArr?
  let each ← groups.toList.mapM fun group => do
    return 1 + (← nat group ["profile", "inner", "matrix", "output_rank"]) +
      (← nat group ["profile", "outer_slice_count"]) *
        (← nat group ["profile", "outer", "matrix", "output_rank"])
  return each.sum + (← nat params ["open_matrix", "output_rank"]) +
    (if ← compressed params then 2 * (groups.size + 1) else 0)

/-- The fields of the levels of a schedule for `tower`. The count of variables of a level is that
of the last group of the next level, which is the group of its witness, and for the last level it
is the least count for the cells of the terminal. The count `s_rel` is the least count for the
rows and the opening row (`relation_row_index_num_vars`). -/
def fileTower (schedule : Json) : Except String (List (List (List ℕ) × List ℕ)) := do
  let folds := (← (← schedule.getObjVal? "recursive_folds").getArr?).toList
  let vars ← folds.mapM fun fold => do
    let groups ← (← field fold ["params", "groups", "entries"]).getArr?
    nat (← (groups.back?.map pure).getD (throw "no group")) ["profile", "group", "num_vars"]
  let last := Nat.clog 2 (← nat schedule ["terminal", "input_witness_len"])
  (((← schedule.getObjVal? "root") :: folds).zip (vars ++ [last])).mapM fun (level, next) => do
    let params ← level.getObjVal? "params"
    let groups ← (← field params ["groups", "entries"]).getArr?
    let rows ← fileRows params
    return (← groups.toList.mapM (fileEntry params),
      [← nat params ["open_matrix", "input_width"], ← nat level ["output_witness_len"], next,
        rows, Nat.clog 2 (rows + 1)])

/-- The fields of a terminal for `terminal`. -/
def fileTerminal (value : Json) : Except String (List ℕ) := do
  let inner ← value.getObjVal? "inner"
  let (degree, norm) ← route inner
  return (← fileFamily (← value.getObjVal? "fold_challenge_config")) ++
    [← nat value ["blocks", "live_blocks"], ← nat value ["blocks", "positions_per_block"],
    2 ^ (← nat inner ["digits", "log_basis"]), ← nat inner ["digits", "num_digits"],
    ← nat inner ["matrix", "output_rank"], degree, ← nat inner ["matrix", "input_width"]] ++ norm

/-- Each row of the file has the Lean numbers `expected` of its levels and of its terminal. -/
def towerHolds (path : System.FilePath)
    (expected : List (List (List (List ℕ) × List ℕ) × List ℕ)) : IO Unit := do
  let entries ← rows path
  unless entries.length == expected.length do
    throw <| IO.userError s!"{path} does not have {expected.length} rows"
  for (entry, tower, terminal) in entries.zip expected do
    let schedule ← IO.ofExcept (entry.getObjVal? "schedule")
    let levels ← IO.ofExcept (fileTower schedule)
    unless levels == tower do
      throw <| IO.userError s!"{path} does not hold the Lean numbers of the levels: the file has
        {levels}, and Lean has {tower}"
    let last ← IO.ofExcept do fileTerminal (← schedule.getObjVal? "terminal")
    unless last == terminal do
      throw <| IO.userError s!"{path} does not hold the Lean numbers of the terminal: the file has
        {last}, and Lean has {terminal}"

/-- The first group of the level 1 of each row of the file is the setup prefix of
`Key30.shapeS`: the file has the count `2^ν_S` of its cells in `setup_natural_len`. -/
def setupHolds (path : System.FilePath) : IO Unit := do
  let S := Key30.shapeS
  for entry in ← rows path do
    let length ← IO.ofExcept do
      let folds ← (← field entry ["schedule", "recursive_folds"]).getArr?
      let level ← (folds[0]?.map pure).getD (throw "no level 1")
      let groups ← (← field level ["params", "groups", "entries"]).getArr?
      nat (← (groups[0]?.map pure).getD (throw "no group")) ["setup_natural_len"]
    unless length == 2 ^ Key30.setupVars ∧ length == S.blocks * S.blockLength * S.innerDegree do
      throw <| IO.userError s!"{path} does not hold the Lean numbers of the setup prefix"

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

/-- Each matrix of each row of the file is below an instance of the Lean list `listed`, and the
file holds each instance of the list after the two compression maps. -/
def instancesHold (path : System.FilePath) (listed : List (ℕ × ℕ × ℕ × Bool × ℕ)) : IO Unit := do
  let each ← (← rows path).mapM fun entry =>
    IO.ofExcept do fileInstances (← entry.getObjVal? "schedule")
  let views := each.flatten
  unless views.all fun view => listed.any (below view) do
    throw <| IO.userError s!"{path} has a matrix that is not below a Lean instance: the file has
      {views}, and Lean has {listed}"
  unless (listed.drop 2).all views.contains do
    throw <| IO.userError s!"{path} does not hold each Lean instance: the file has {views}, and
      Lean has {listed}"

#eval holds "../../crates/ns-compress/artifacts/key20.aks" [key20] Shape.key20 Descriptor.key20
#eval holds "../../crates/ns-compress/artifacts/key30.aks" [key30W26, key30W27] Shape.key30
  Descriptor.key30
#eval towerHolds "../../crates/ns-compress/artifacts/key20.aks"
  [(tower, terminalNumbers (Key20.family 6 0) Key20.terminal Key20.normTerminal)]
#eval towerHolds "../../crates/ns-compress/artifacts/key30.aks" <| Key30.rows.map fun r =>
  (tower30 r, terminalNumbers (Key30.family 7 0) Key30.terminal Key30.normTerminal)
#eval setupHolds "../../crates/ns-compress/artifacts/key30.aks"
#eval (do unless segmentsAligned do throw <| IO.userError "an alignment of the code is not empty"
  : IO Unit)
#eval instancesHold "../../crates/ns-compress/artifacts/key20.aks" (Key20.instances.map (·.key))
#eval instancesHold "../../crates/ns-compress/artifacts/key30.aks" (Key30.instances.map (·.key))

end AkitaSchedules
