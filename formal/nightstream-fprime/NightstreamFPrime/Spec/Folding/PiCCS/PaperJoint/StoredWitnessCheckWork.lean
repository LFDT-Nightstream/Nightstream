import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck

/-!
Charged execution of the existing stored-witness Boolean check. The driver
owns finite loops, stored-array reads, and streaming MLE recursion. Every
function-valued public/key/matrix read and every arithmetic primitive returns
its value and work from the same call. Primitive refinement and work bounds
remain explicit; neither opening validity nor a constant function cost is
assumed.

The counter measures named abstract operations, not wall time. Each finite
loop step charges control, index advance, and result construction. Its initial
result, terminal control, and final return cost three operations together.
Boolean sequencing charges dispatch, continuation construction, and return.
The MLE traversal charges its branch/leaf operations and every constructed
Boolean vertex. Compilation and concrete primitive clocks remain separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier FullOutputCoordinates PaperLinearAlgebra
open StoredWitnessProjection
open CheckedWitnessExtraction (openingMaps)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- The remaining implementation owners. All costs include their own calls
and representation work, including any key expansion inside commitment. -/
structure Program (shape : Shape) (carrier : Phi81Relation.Shape) where
  publicCheck : StoredProbe shape → Result Bool
  commitmentCheck : StoredWitness shape carrier → Fin shape.sourceCount → Result Bool
  publicInput : Fin shape.sourceCount → Fin carrier.publicWidth → Result F
  padEntry : Fin shape.coefficientCount → BooleanVertex shape.cubeVariables →
    Fin carrier.carrierWidth → Result F
  matrixEntry : Fin shape.matrixCount → Fin shape.coefficientCount →
    BooleanVertex shape.cubeVariables → Fin carrier.carrierWidth → Result F
  padClaim : StoredProbe shape → Fin shape.sourceCount → Fin shape.coefficientCount → Result K
  matrixClaim : StoredProbe shape → Fin shape.sourceCount → Fin shape.matrixCount →
    Fin shape.coefficientCount → Result K
  dotStep : F → F → F → Result F
  embed : F → Result K
  interpolate : K → K → K → Result K
  equalF : F → F → Result Bool
  equalK : K → K → Result Bool
  norm : F → Result Bool

structure PrimitiveBounds where
  publicCheck : Nat
  commitmentCheck : Nat
  publicInput : Nat
  padEntry : Nat
  matrixEntry : Nat
  padClaim : Nat
  matrixClaim : Nat
  dotStep : Nat
  embed : Nat
  interpolate : Nat
  equalF : Nat
  equalK : Nat
  norm : Nat

variable {shape : Shape} {carrier : Phi81Relation.Shape}

/-- Each bound constrains the work returned by the actual primitive call. -/
structure Bounded (program : Program shape carrier) (bounds : PrimitiveBounds) : Prop where
  publicCheck : ∀ probe, (program.publicCheck probe).work ≤ bounds.publicCheck
  commitmentCheck : ∀ stored source,
    (program.commitmentCheck stored source).work ≤ bounds.commitmentCheck
  publicInput : ∀ source column, (program.publicInput source column).work ≤ bounds.publicInput
  padEntry : ∀ coefficient vertex column,
    (program.padEntry coefficient vertex column).work ≤ bounds.padEntry
  matrixEntry : ∀ matrix coefficient vertex column,
    (program.matrixEntry matrix coefficient vertex column).work ≤ bounds.matrixEntry
  padClaim : ∀ probe source coefficient,
    (program.padClaim probe source coefficient).work ≤ bounds.padClaim
  matrixClaim : ∀ probe source matrix coefficient,
    (program.matrixClaim probe source matrix coefficient).work ≤ bounds.matrixClaim
  dotStep : ∀ accumulated matrix value,
    (program.dotStep accumulated matrix value).work ≤ bounds.dotStep
  embed : ∀ value, (program.embed value).work ≤ bounds.embed
  interpolate : ∀ low high point,
    (program.interpolate low high point).work ≤ bounds.interpolate
  equalF : ∀ left right, (program.equalF left right).work ≤ bounds.equalF
  equalK : ∀ left right, (program.equalK left right).work ≤ bounds.equalK
  norm : ∀ value, (program.norm value).work ≤ bounds.norm

/-- Primitive values refine the operations that the pure checker executes. -/
structure Correct {Commitment : Type*} [DecidableEq Commitment]
    {blockCount : Nat} (program : Program shape carrier)
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps) (width : Nat) : Prop where
  publicCheck : ∀ probe, (program.publicCheck probe).value =
    ProtocolPolynomial.FixedWidth.check extensionOps width (statement.verifierInput K.embed)
      probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
      (statement.projectOutput probe.view.response.fullOutput) probe.certificate
  commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).value =
    decide (commit (stored.get source).get = statement.commitments source)
  publicInput : ∀ source column, (program.publicInput source column).value =
    statement.publicInputs source column
  padEntry : ∀ coefficient vertex column, (program.padEntry coefficient vertex column).value =
    statement.matrixSource.coefficientMatrixOf baseOps
      (fun row column => statement.cubeLayout.paddedIdentityEntry
        baseOps.zero baseOps.one row column) coefficient vertex column
  matrixEntry : ∀ matrix coefficient vertex column,
    (program.matrixEntry matrix coefficient vertex column).value =
      statement.matrixSource.coefficientMatrix baseOps matrix coefficient vertex column
  padClaim : ∀ probe source coefficient, (program.padClaim probe source coefficient).value =
    probe.view.response.fullOutput.padCoordinate source coefficient
  matrixClaim : ∀ probe source matrix coefficient,
    (program.matrixClaim probe source matrix coefficient).value =
      probe.view.response.fullOutput.matrixCoordinate source matrix coefficient
  dotStep : ∀ accumulated matrix value, (program.dotStep accumulated matrix value).value =
    baseOps.add accumulated (baseOps.mul matrix value)
  embed : ∀ value, (program.embed value).value = K.embed value
  interpolate : ∀ low high point, (program.interpolate low high point).value =
    extensionOps.add low (extensionOps.mul point (extensionOps.sub high low))
  equalF : ∀ left right, (program.equalF left right).value = decide (left = right)
  equalK : ∀ left right, (program.equalK left right).value = decide (left = right)
  norm : ∀ value, (program.norm value).value = decide (centeredMagnitude value < params.ambientBound)

private def foldStep {Value : Type*} {count : Nat}
    (step : Value → Fin count → Result Value) (accumulated : Result Value)
    (index : Fin count) : Result Value :=
  let next := step accumulated.value index
  ⟨next.value, accumulated.work + next.work + 3⟩

private def fold {Value : Type*} {count : Nat}
    (step : Value → Fin count → Result Value) (initial : Value) : Result Value :=
  let result := Fin.foldl count (foldStep step) (⟨initial, 1⟩ : Result Value)
  ⟨result.value, result.work + 2⟩

private theorem foldCore_value {Value : Type*} : ∀ {count : Nat}
    (step : Value → Fin count → Result Value) (initial : Result Value),
    (Fin.foldl count (foldStep step) initial).value =
      Fin.foldl count (fun value index => (step value index).value) initial.value
  | 0, _, _ => by simp only [Fin.foldl_zero]
  | _ + 1, step, initial => by
      rw [Fin.foldl_succ, Fin.foldl_succ]
      exact foldCore_value (fun value index => step value index.succ) (foldStep step initial 0)

private theorem fold_value {Value : Type*} {count : Nat}
    (step : Value → Fin count → Result Value) (initial : Value) :
    (fold step initial).value = Fin.foldl count (fun value index => (step value index).value) initial :=
  foldCore_value step ⟨initial, 1⟩

private theorem foldCore_work_le {Value : Type*} (bound : Nat) : ∀ {count : Nat}
    (step : Value → Fin count → Result Value) (initial : Result Value),
    (∀ value index, (step value index).work ≤ bound) →
    (Fin.foldl count (foldStep step) initial).work ≤
      initial.work + count * (bound + 3)
  | 0, _, _, _ => by simp
  | count + 1, step, initial, bounded => by
      rw [Fin.foldl_succ]
      have tail := foldCore_work_le bound (fun value index => step value index.succ)
        (foldStep step initial 0)
        (fun value index => bounded value index.succ)
      have head := bounded initial.value 0
      exact le_trans tail (by simp only [foldStep, Nat.add_mul, Nat.one_mul]; omega)

private theorem fold_work_le {Value : Type*} {count : Nat}
    (step : Value → Fin count → Result Value) (initial : Value) (bound : Nat)
    (bounded : ∀ value index, (step value index).work ≤ bound) :
    (fold step initial).work ≤ count * (bound + 3) + 3 := by
  have total := foldCore_work_le bound step (⟨initial, 1⟩ : Result Value) bounded
  change (Fin.foldl count (foldStep step) ⟨initial, 1⟩).work ≤ 1 + count * (bound + 3) at total
  dsimp only [fold]
  omega

private def andThen (left : Result Bool) (right : Unit → Result Bool) : Result Bool :=
  if left.value then
    let next := right ()
    ⟨next.value, left.work + next.work + 3⟩
  else ⟨false, left.work + 3⟩

private theorem andThen_value (left : Result Bool) (right : Unit → Result Bool) :
    (andThen left right).value = (left.value && (right ()).value) := by
  cases accepted : left.value <;> simp [andThen, accepted]

private theorem andThen_work_le (left : Result Bool) (right : Unit → Result Bool) :
    (andThen left right).work ≤ left.work + (right ()).work + 3 := by
  cases accepted : left.value <;> simp [andThen, accepted]

private def allFin {count : Nat} (test : Fin count → Result Bool) : Result Bool :=
  fold (fun accepted index =>
    if accepted then
      let checked := test index
      ⟨checked.value, checked.work + 2⟩
    else ⟨false, 2⟩) true

private theorem allFin_value {count : Nat} (test : Fin count → Result Bool) :
    (allFin test).value =
      Fin.foldl count (fun accepted index => accepted && (test index).value) true := by
  rw [allFin, fold_value]
  congr 1
  funext accepted index
  cases accepted <;> rfl

private theorem allFin_work_le {count : Nat} (test : Fin count → Result Bool) (bound : Nat)
    (bounded : ∀ index, (test index).work ≤ bound) :
    (allFin test).work ≤ count * (bound + 5) + 3 := by
  apply fold_work_le _ _ (bound + 2)
  intro accepted index
  have called := bounded index
  cases accepted <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> omega

private def dot {count : Nat} (program : Program shape carrier)
    (left right : Fin count → Result F) : Result F :=
  fold (fun accumulated index =>
    let matrix := left index
    let value := right index
    let result := program.dotStep accumulated matrix.value value.value
    ⟨result.value, matrix.work + value.work + result.work + 1⟩) baseOps.zero

private def consRow {arity : Nat} (head : Bool)
    (rows : BooleanVertex (arity + 1) → Result K) (tail : BooleanVertex arity) : Result K :=
  let result := rows (.cons head tail)
  ⟨result.value, result.work + 2⟩

private def evaluateRows (program : Program shape carrier) : {arity : Nat} →
    (BooleanVertex arity → Result K) → List K → Result K
  | 0, rows, [] =>
      let result := rows .nil
      ⟨result.value, result.work + 3⟩
  | _ + 1, rows, coordinate :: coordinates =>
      let low := evaluateRows program (consRow false rows) coordinates
      let high := evaluateRows program (consRow true rows) coordinates
      let result := program.interpolate low.value high.value coordinate
      ⟨result.value, low.work + high.work + result.work + 4⟩
  | _, _, _ => ⟨extensionOps.zero, 2⟩

private def coefficient (program : Program shape carrier)
    (entry : BooleanVertex shape.cubeVariables → Fin carrier.carrierWidth → Result F)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (coordinates : List K) : Result K :=
  evaluateRows program (fun vertex =>
    let row := dot program (entry vertex) (read stored source)
    let result := program.embed row.value
    ⟨result.value, row.work + result.work + 1⟩) coordinates

private def prefixCoordinate (program : Program shape carrier)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (column : Fin carrier.publicWidth) : Result Bool :=
  let value := read stored source (carrier.publicColumn column)
  let expected := program.publicInput source column
  let checked := program.equalF value.value expected.value
  ⟨checked.value, value.work + expected.work + checked.work + 1⟩

private def normCoordinate (program : Program shape carrier)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (column : Fin carrier.carrierWidth) : Result Bool :=
  let value := read stored source column
  let checked := program.norm value.value
  ⟨checked.value, value.work + checked.work + 1⟩

private def opening (program : Program shape carrier)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) : Result Bool :=
  andThen (program.commitmentCheck stored source) fun _ =>
    andThen (allFin (prefixCoordinate program stored source)) fun _ =>
      allFin (normCoordinate program stored source)

private def compareEvaluation (program : Program shape carrier)
    (expected received : Result K) : Result Bool :=
  let checked := program.equalK expected.value received.value
  ⟨checked.value, expected.work + received.work + checked.work + 1⟩

private def evaluation (program : Program shape carrier) (probe : StoredProbe shape)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (coordinates : List K) : Result Bool :=
  andThen (allFin fun index : Fin shape.coefficientCount =>
    compareEvaluation program (coefficient program (program.padEntry index) stored source coordinates)
      (program.padClaim probe source index)) fun _ =>
    allFin fun matrix : Fin shape.matrixCount =>
      allFin fun index : Fin shape.coefficientCount =>
        compareEvaluation program
          (coefficient program (program.matrixEntry matrix index) stored source coordinates)
          (program.matrixClaim probe source matrix index)

/-- Three record projections and one result; the coordinate list is shared. -/
private def pointCoordinates (probe : StoredProbe shape) : Result (List K) :=
  ⟨probe.coins.roundPoint.coordinates, 1 + 1 + 1 + 1⟩

private def ambient (program : Program shape carrier) (probe : StoredProbe shape)
    (stored : StoredWitness shape carrier) : Result Bool :=
  let point := pointCoordinates probe
  let checked := allFin fun source : Fin shape.sourceCount =>
    andThen (opening program stored source) fun _ =>
      evaluation program probe stored source point.value
  ⟨checked.value, point.work + checked.work + 1⟩

/-- The public check executes first. Rejection skips ambient witness work. -/
def check (program : Program shape carrier)
    (candidate : StoredProbe shape × StoredWitness shape carrier) : Result Bool :=
  andThen (program.publicCheck candidate.1) fun _ => ambient program candidate.1 candidate.2

section Values

variable {Commitment : Type*} [DecidableEq Commitment] {blockCount width : Nat}
  (program : Program shape carrier)
  (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
  (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)
  (correct : Correct program commit params statement width)

include correct in
private theorem dot_value {count : Nat} (left right : Fin count → Result F) :
    (dot program left right).value = (canonicalFinIndices count).foldl
      (fun accumulated index => baseOps.add accumulated
        (baseOps.mul (left index).value (right index).value)) baseOps.zero := by
  rw [dot, fold_value]
  simp only [correct.dotStep]
  rw [Fin.foldl_eq_finRange_foldl]
  rfl

include correct in
private theorem evaluateRows_value {arity : Nat}
    (rows : BooleanVertex arity → Result K) (coordinates : List K) :
    (evaluateRows program rows coordinates).value =
      (BooleanTable.tabulate (fun vertex => (rows vertex).value)).evaluateCoordinates
        extensionOps coordinates := by
  induction arity generalizing coordinates with
  | zero => cases coordinates <;> rfl
  | succ arity inductionHypothesis =>
      cases coordinates <;>
        simp only [evaluateRows, correct.interpolate, inductionHypothesis, consRow,
          BooleanTable.tabulate, BooleanTable.evaluateCoordinates]

include correct in
private theorem coefficient_value
    (entry : BooleanVertex shape.cubeVariables → Fin carrier.carrierWidth → Result F)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (coordinates : List K) :
    (coefficient program entry stored source coordinates).value =
      (BooleanTable.tabulate (fun vertex => K.embed (matrixVectorAt baseOps
        (fun vertex column => (entry vertex column).value)
        ((view stored).assignments source) vertex))).evaluateCoordinates extensionOps coordinates := by
  rw [coefficient, evaluateRows_value program commit params statement correct]
  simp only [correct.embed, dot_value program commit params statement correct]
  rfl

include correct in
private theorem pad_value (probe : StoredProbe shape) (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) (index : Fin shape.coefficientCount) :
    (coefficient program (program.padEntry index) stored source probe.coins.roundPoint.coordinates).value =
      (StoredWitnessCheck.evaluations statement probe.view stored).padCoordinate source index := by
  rw [coefficient_value program commit params statement correct, StoredWitnessCheck.evaluations_eq_honestAt]
  simp only [correct.padEntry]
  rfl

include correct in
private theorem matrix_value (probe : StoredProbe shape) (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) (matrix : Fin shape.matrixCount) (index : Fin shape.coefficientCount) :
    (coefficient program (program.matrixEntry matrix index) stored source probe.coins.roundPoint.coordinates).value =
      (StoredWitnessCheck.evaluations statement probe.view stored).matrixCoordinate source matrix index := by
  rw [coefficient_value program commit params statement correct, StoredWitnessCheck.evaluations_eq_honestAt]
  simp only [correct.matrixEntry]
  rfl

include correct in
private theorem opening_value (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) :
    (opening program stored source).value = StoredWitnessCheck.openingCheck commit params statement stored source := by
  simp only [opening, andThen_value, allFin_value, prefixCoordinate, normCoordinate,
    correct.commitmentCheck, correct.equalF, correct.publicInput, correct.norm]
  rfl

include correct in
private theorem evaluation_value (probe : StoredProbe shape) (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) :
    (evaluation program probe stored source probe.coins.roundPoint.coordinates).value =
      StoredWitnessCheck.evaluationCheck (StoredWitnessCheck.evaluations statement probe.view stored)
        probe.view.response.fullOutput source := by
  simp only [evaluation, andThen_value, allFin_value, compareEvaluation, correct.equalK,
    correct.padClaim, correct.matrixClaim, pad_value program commit params statement correct,
    matrix_value program commit params statement correct]
  rfl

include correct in
private theorem ambient_value (probe : StoredProbe shape) (stored : StoredWitness shape carrier) :
    (ambient program probe stored).value = StoredWitnessCheck.ambientCheck commit params statement probe.view stored := by
  simp only [ambient, pointCoordinates, allFin_value, andThen_value,
    opening_value program commit params statement correct,
    evaluation_value program commit params statement correct]
  rfl

include correct in
/-- Erasing the actual invocation's work gives the already proved Boolean checker. -/
theorem check_value (candidate : StoredProbe shape × StoredWitness shape carrier) :
    (check program candidate).value =
      StoredWitnessCheck.check commit params statement width (candidate.1.view, candidate.2) := by
  simp only [check, andThen_value, correct.publicCheck, ambient_value program commit params statement correct]
  rfl

end Values

/-- A row visits every stored coefficient. Each step includes the two-array
read, the returned row term, and the finite-loop operations. -/
def rowWork (carrier : Phi81Relation.Shape) (bounds : PrimitiveBounds) (entryBound : Nat) : Nat :=
  carrier.carrierWidth * (entryBound + (1 + 1 + 1) + bounds.dotStep + 4) + 3

/-- A binary MLE traversal, including vertex construction along every path.
This symbolic upper bound depends on row count and depth, not emitted data. -/
def mleWork (arity rowBound interpolationBound : Nat) : Nat :=
  2 ^ arity * (rowBound + 3 + arity * (interpolationBound + 6))

def coefficientWork (shape : Shape) (carrier : Phi81Relation.Shape)
    (bounds : PrimitiveBounds) (entryBound : Nat) : Nat :=
  mleWork shape.cubeVariables (rowWork carrier bounds entryBound + bounds.embed + 1) bounds.interpolate

def openingWork (carrier : Phi81Relation.Shape) (bounds : PrimitiveBounds) : Nat :=
  bounds.commitmentCheck +
    (carrier.publicWidth * ((1 + 1 + 1) + bounds.publicInput + bounds.equalF + 6) + 3) +
    (carrier.carrierWidth * ((1 + 1 + 1) + bounds.norm + 6) + 3) + 6

def evaluationWork (shape : Shape) (carrier : Phi81Relation.Shape) (bounds : PrimitiveBounds) : Nat :=
  (shape.coefficientCount *
    (coefficientWork shape carrier bounds bounds.padEntry + bounds.padClaim + bounds.equalK + 6) + 3) +
  (shape.matrixCount *
    (shape.coefficientCount *
      (coefficientWork shape carrier bounds bounds.matrixEntry + bounds.matrixClaim + bounds.equalK + 6) + 8) + 3) + 3

/-- One public gate and the complete source loop. The last eleven operations
come from public sequencing, point reads, loop endpoints, and ambient return. -/
def workBound (shape : Shape) (carrier : Phi81Relation.Shape) (bounds : PrimitiveBounds) : Nat :=
  bounds.publicCheck + shape.sourceCount * (openingWork carrier bounds + evaluationWork shape carrier bounds + 8) + 11

section Work

variable (program : Program shape carrier) (bounds : PrimitiveBounds) (bounded : Bounded program bounds)

include bounded in
private theorem dot_work_le {count : Nat} (left right : Fin count → Result F)
    (leftBound rightBound : Nat)
    (leftBounded : ∀ index, (left index).work ≤ leftBound)
    (rightBounded : ∀ index, (right index).work ≤ rightBound) :
    (dot program left right).work ≤ count * (leftBound + rightBound + bounds.dotStep + 4) + 3 := by
  unfold dot
  apply fold_work_le _ _ (leftBound + rightBound + bounds.dotStep + 1)
  intro accumulated index
  have matrixWork := leftBounded index
  have valueWork := rightBounded index
  have arithmeticWork := bounded.dotStep accumulated (left index).value (right index).value
  dsimp only
  omega

include bounded in
private theorem evaluateRows_work_le : ∀ {arity : Nat}
    (rows : BooleanVertex arity → Result K) (coordinates : List K) (rowBound : Nat),
    (∀ vertex, (rows vertex).work ≤ rowBound) →
    (evaluateRows program rows coordinates).work ≤ mleWork arity rowBound bounds.interpolate
  | 0, rows, [], rowBound, rowsBounded => by
      have leaf := rowsBounded .nil
      dsimp only [evaluateRows, mleWork]
      omega
  | 0, _, _ :: _, _, _ => by simp only [evaluateRows, mleWork, Nat.pow_zero, Nat.zero_mul]; omega
  | arity + 1, _, [], rowBound, _ => by
      have positive : 1 ≤ 2 ^ (arity + 1) := Nat.one_le_two_pow
      have size : 2 ≤ rowBound + 3 + (arity + 1) * (bounds.interpolate + 6) := by omega
      exact Nat.mul_le_mul positive size
  | arity + 1, rows, coordinate :: coordinates, rowBound, rowsBounded => by
      have low := evaluateRows_work_le (consRow false rows) coordinates (rowBound + 2)
        (fun vertex => by
          have leaf := rowsBounded (.cons false vertex)
          dsimp only [consRow]
          omega)
      have high := evaluateRows_work_le (consRow true rows) coordinates (rowBound + 2)
        (fun vertex => by
          have leaf := rowsBounded (.cons true vertex)
          dsimp only [consRow]
          omega)
      have arithmeticWork := bounded.interpolate
        (evaluateRows program (consRow false rows) coordinates).value
        (evaluateRows program (consRow true rows) coordinates).value coordinate
      have positive : 1 ≤ 2 ^ arity := Nat.one_le_two_pow
      change (evaluateRows program (consRow false rows) coordinates).work +
          (evaluateRows program (consRow true rows) coordinates).work +
          (program.interpolate
            (evaluateRows program (consRow false rows) coordinates).value
            (evaluateRows program (consRow true rows) coordinates).value coordinate).work + 4 ≤ _
      dsimp only [mleWork] at low high ⊢
      rw [pow_succ]
      nlinarith

include bounded in
private theorem coefficient_work_le
    (entry : BooleanVertex shape.cubeVariables → Fin carrier.carrierWidth → Result F)
    (entryBound : Nat) (entryBounded : ∀ vertex column, (entry vertex column).work ≤ entryBound)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) (coordinates : List K) :
    (coefficient program entry stored source coordinates).work ≤ coefficientWork shape carrier bounds entryBound := by
  unfold coefficient coefficientWork
  apply evaluateRows_work_le program bounds bounded _ _ _
  intro vertex
  have row := dot_work_le program bounds bounded (entry vertex) (read stored source)
    entryBound (1 + 1 + 1) (entryBounded vertex) (fun _ => Nat.le_refl _)
  have embedded := bounded.embed (dot program (entry vertex) (read stored source)).value
  dsimp only
  unfold rowWork
  omega

include bounded in
private theorem prefixCoordinate_work_le (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) (column : Fin carrier.publicWidth) :
    (prefixCoordinate program stored source column).work ≤
      (1 + 1 + 1) + bounds.publicInput + bounds.equalF + 1 := by
  have publicWork := bounded.publicInput source column
  have comparison := bounded.equalF (read stored source (carrier.publicColumn column)).value
    (program.publicInput source column).value
  have accessed : (StoredWitnessProjection.read stored source (carrier.publicColumn column)).work =
      1 + 1 + 1 := rfl
  dsimp only [prefixCoordinate]
  rw [accessed]
  omega

include bounded in
private theorem normCoordinate_work_le (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) (column : Fin carrier.carrierWidth) :
    (normCoordinate program stored source column).work ≤ (1 + 1 + 1) + bounds.norm + 1 := by
  have comparison := bounded.norm (read stored source column).value
  have accessed : (StoredWitnessProjection.read stored source column).work = 1 + 1 + 1 := rfl
  dsimp only [normCoordinate]
  rw [accessed]
  omega

include bounded in
private theorem opening_work_le (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) :
    (opening program stored source).work ≤ openingWork carrier bounds := by
  have prefixes := allFin_work_le (prefixCoordinate program stored source)
    ((1 + 1 + 1) + bounds.publicInput + bounds.equalF + 1)
    (prefixCoordinate_work_le program bounds bounded stored source)
  have norms := allFin_work_le (normCoordinate program stored source)
    ((1 + 1 + 1) + bounds.norm + 1) (normCoordinate_work_le program bounds bounded stored source)
  have inside := andThen_work_le (allFin (prefixCoordinate program stored source))
    (fun _ => allFin (normCoordinate program stored source))
  have outside := andThen_work_le (program.commitmentCheck stored source) fun _ =>
    andThen (allFin (prefixCoordinate program stored source)) fun _ =>
      allFin (normCoordinate program stored source)
  have committed := bounded.commitmentCheck stored source
  dsimp only [opening]
  unfold openingWork
  simp only [Nat.add_assoc, Nat.reduceAdd] at prefixes norms ⊢
  omega

include bounded in
private theorem compareEvaluation_work_le (expected received : Result K) :
    (compareEvaluation program expected received).work ≤
      expected.work + received.work + bounds.equalK + 1 := by
  have compared := bounded.equalK expected.value received.value
  dsimp only [compareEvaluation]
  omega

include bounded in
private theorem evaluation_work_le (probe : StoredProbe shape) (stored : StoredWitness shape carrier)
    (source : Fin shape.sourceCount) (coordinates : List K) :
    (evaluation program probe stored source coordinates).work ≤ evaluationWork shape carrier bounds := by
  have padEach (index : Fin shape.coefficientCount) :
      (compareEvaluation program (coefficient program (program.padEntry index) stored source coordinates)
        (program.padClaim probe source index)).work ≤
        coefficientWork shape carrier bounds bounds.padEntry + bounds.padClaim + bounds.equalK + 1 := by
    have computed := coefficient_work_le program bounds bounded (program.padEntry index)
      bounds.padEntry (bounded.padEntry index) stored source coordinates
    have claimed := bounded.padClaim probe source index
    have compared := compareEvaluation_work_le program bounds bounded
      (coefficient program (program.padEntry index) stored source coordinates)
      (program.padClaim probe source index)
    omega
  have matrixEach (matrix : Fin shape.matrixCount) (index : Fin shape.coefficientCount) :
      (compareEvaluation program
        (coefficient program (program.matrixEntry matrix index) stored source coordinates)
        (program.matrixClaim probe source matrix index)).work ≤
        coefficientWork shape carrier bounds bounds.matrixEntry + bounds.matrixClaim + bounds.equalK + 1 := by
    have computed := coefficient_work_le program bounds bounded (program.matrixEntry matrix index)
      bounds.matrixEntry (bounded.matrixEntry matrix index) stored source coordinates
    have claimed := bounded.matrixClaim probe source matrix index
    have compared := compareEvaluation_work_le program bounds bounded
      (coefficient program (program.matrixEntry matrix index) stored source coordinates)
      (program.matrixClaim probe source matrix index)
    omega
  have pads := allFin_work_le _ _ padEach
  have matrices := allFin_work_le _ _ (fun matrix => allFin_work_le _ _ (matrixEach matrix))
  have combined := andThen_work_le
    (allFin fun index : Fin shape.coefficientCount =>
      compareEvaluation program (coefficient program (program.padEntry index) stored source coordinates)
        (program.padClaim probe source index))
    (fun _ => allFin fun matrix : Fin shape.matrixCount =>
      allFin fun index : Fin shape.coefficientCount =>
        compareEvaluation program
          (coefficient program (program.matrixEntry matrix index) stored source coordinates)
          (program.matrixClaim probe source matrix index))
  dsimp only [evaluation]
  unfold evaluationWork
  simp only [Nat.add_assoc, Nat.reduceAdd] at pads matrices ⊢
  omega

include bounded in
private theorem ambient_work_le (probe : StoredProbe shape) (stored : StoredWitness shape carrier) :
    (ambient program probe stored).work ≤
      shape.sourceCount * (openingWork carrier bounds + evaluationWork shape carrier bounds + 8) + 8 := by
  have each (source : Fin shape.sourceCount) :
      (andThen (opening program stored source) fun _ =>
        evaluation program probe stored source probe.coins.roundPoint.coordinates).work ≤
        openingWork carrier bounds + evaluationWork shape carrier bounds + 3 := by
    have opened := opening_work_le program bounds bounded stored source
    have evaluated := evaluation_work_le program bounds bounded probe stored source probe.coins.roundPoint.coordinates
    have combined := andThen_work_le (opening program stored source) fun _ =>
      evaluation program probe stored source probe.coins.roundPoint.coordinates
    omega
  have sources := allFin_work_le _ _ each
  dsimp only [ambient, pointCoordinates]
  simp only [Nat.add_assoc, Nat.reduceAdd] at sources ⊢
  omega

include bounded in
/-- Bound the executed checker, including rejected inputs, from actual
primitive bounds and the proved finite-loop and MLE counts. -/
theorem check_work_le (candidate : StoredProbe shape × StoredWitness shape carrier) :
    (check program candidate).work ≤ workBound shape carrier bounds := by
  have publicWork := bounded.publicCheck candidate.1
  have witnessWork := ambient_work_le program bounds bounded candidate.1 candidate.2
  have combined := andThen_work_le (program.publicCheck candidate.1) fun _ => ambient program candidate.1 candidate.2
  dsimp only [check]
  unfold workBound
  omega

end Work

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork
