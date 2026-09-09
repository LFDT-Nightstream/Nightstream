import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-!
Charged execution of SuperNeo v1.1 Appendix B.3 step 5. Each primitive returns
its value and work from the same call. Primitive work includes dispatch and
representation work, including any packing inside the assignment operations.

The inverse implementation receives only the scalar difference. It does not
receive a supplied inverse. Its correctness and work on unit inputs are
explicit primitive premises. Polynomial primitive bounds therefore remain
necessary for a polynomial runtime claim.

The driver charges one result return per coordinate and one list construction
per batch entry, including the final empty list. On a CompleteFork, the proved
opening theorem makes the paper's final source-membership test always true.
No additional computational acceptance gate is hidden in this program.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

open NightstreamFPrime.Spec
open PaperForkExtraction PaperForkAlgebra

universe uValue uScalar uAssignment uStructure uPublicInput uPoint uEvaluation uCommitment

/-- A primitive result and the work charged by that same invocation. -/
structure Result (Value : Type uValue) where
  value : Value
  work : Nat

/-- The four implementations executed by inverse-difference extraction. -/
structure Primitives (Scalar : Type uScalar) (Assignment : Type uAssignment) where
  scalarSub : Scalar → Scalar → Result Scalar
  unitInverse : Scalar → Result Scalar
  assignmentSub : Assignment → Assignment → Result Assignment
  scalarAction : Scalar → Assignment → Result Assignment

/-- Each implementation computes the corresponding existing algebra operation.
The inverse equality is required only on inputs proved to be units. -/
structure Correct {Scalar : Type uScalar} {Assignment : Type uAssignment}
    (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) : Prop where
  scalarSub : ∀ left right, (program.scalarSub left right).value = ring.sub left right
  unitInverse : ∀ value (unit : UnitWitness ring value),
    (program.unitInverse value).value = unit.inverse
  assignmentSub : ∀ left right,
    (program.assignmentSub left right).value = module.sub left right
  scalarAction : ∀ scalar assignment,
    (program.scalarAction scalar assignment).value = module.smul scalar assignment

/-- Bounds for the four primitive implementations at the selected input size. -/
structure PrimitiveBounds where
  scalarSub : Nat
  unitInverse : Nat
  assignmentSub : Nat
  scalarAction : Nat

/-- Work of one coordinate, including its result return. -/
def PrimitiveBounds.coordinateWork (bounds : PrimitiveBounds) : Nat :=
  bounds.scalarSub + bounds.unitInverse + bounds.assignmentSub + bounds.scalarAction + 1

/-- Bounds apply to returned invocation clocks, not to a free extractor total. -/
structure Bounded {Scalar : Type uScalar} {Assignment : Type uAssignment}
    (ring : CommutativeRingOps Scalar) (program : Primitives Scalar Assignment)
    (bounds : PrimitiveBounds) : Prop where
  scalarSub : ∀ left right, (program.scalarSub left right).work ≤ bounds.scalarSub
  unitInverse : ∀ value (_unit : UnitWitness ring value),
    (program.unitInverse value).work ≤ bounds.unitInverse
  assignmentSub : ∀ left right,
    (program.assignmentSub left right).work ≤ bounds.assignmentSub
  scalarAction : ∀ scalar assignment,
    (program.scalarAction scalar assignment).work ≤ bounds.scalarAction

/-- Execute the four primitives once, with each actual intermediate result. -/
def extract {Scalar : Type uScalar} {Assignment : Type uAssignment}
    (program : Primitives Scalar Assignment)
    (baseScalar forkScalar : Scalar) (baseAssignment forkAssignment : Assignment) :
    Result Assignment :=
  let delta := program.scalarSub baseScalar forkScalar
  let inverse := program.unitInverse delta.value
  let difference := program.assignmentSub baseAssignment forkAssignment
  let output := program.scalarAction inverse.value difference.value
  { value := output.value
    work := delta.work + inverse.work + difference.work + output.work + 1 }

/-- The charged program returns the same inverse-difference assignment. -/
theorem extract_value {Scalar : Type uScalar} {Assignment : Type uAssignment}
    (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) (correct : Correct ring module program)
    (baseScalar forkScalar : Scalar) (baseAssignment forkAssignment : Assignment)
    (unit : UnitWitness ring (ring.sub baseScalar forkScalar)) :
    (extract program baseScalar forkScalar baseAssignment forkAssignment).value =
      module.smul unit.inverse (module.sub baseAssignment forkAssignment) := by
  simp only [extract, correct.scalarAction, correct.assignmentSub, correct.scalarSub,
    correct.unitInverse _ unit]

/-- One coordinate charges only its executed primitive clocks and return. -/
theorem extract_work_le {Scalar : Type uScalar} {Assignment : Type uAssignment}
    (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) (correct : Correct ring module program)
    (bounds : PrimitiveBounds) (bounded : Bounded ring program bounds)
    (baseScalar forkScalar : Scalar) (baseAssignment forkAssignment : Assignment)
    (unit : UnitWitness ring (ring.sub baseScalar forkScalar)) :
    (extract program baseScalar forkScalar baseAssignment forkAssignment).work ≤
      bounds.coordinateWork := by
  have scalarWork := bounded.scalarSub baseScalar forkScalar
  have inverseWork := bounded.unitInverse (ring.sub baseScalar forkScalar) unit
  have assignmentWork := bounded.assignmentSub baseAssignment forkAssignment
  have actionWork := bounded.scalarAction
    (program.unitInverse (program.scalarSub baseScalar forkScalar).value).value
    (program.assignmentSub baseAssignment forkAssignment).value
  change (program.scalarSub baseScalar forkScalar).work +
      (program.unitInverse (program.scalarSub baseScalar forkScalar).value).work +
      (program.assignmentSub baseAssignment forkAssignment).work +
      (program.scalarAction
        (program.unitInverse (program.scalarSub baseScalar forkScalar).value).value
        (program.assignmentSub baseAssignment forkAssignment).value).work + 1 ≤ _
  rw [correct.scalarSub] at actionWork ⊢
  unfold PrimitiveBounds.coordinateWork
  omega

/-- Run each coordinate in order. Each cons and the final nil cost one step. -/
def collect {Assignment : Type uAssignment} : {count : Nat} →
    (Fin count → Result Assignment) → Result (List Assignment)
  | 0, _ => { value := [], work := 1 }
  | _ + 1, coordinate =>
      let head := coordinate 0
      let tail := collect (fun index => coordinate index.succ)
      { value := head.value :: tail.value
        work := head.work + tail.work + 1 }

theorem collect_values {Assignment : Type uAssignment} : ∀ {count : Nat}
    (coordinate : Fin count → Result Assignment),
    (collect coordinate).value = List.ofFn (fun index => (coordinate index).value)
  | 0, _ => rfl
  | _ + 1, coordinate => by
      change (coordinate 0).value ::
          (collect (fun index => coordinate index.succ)).value = _
      rw [List.ofFn_succ, collect_values]

theorem collect_work_le {Assignment : Type uAssignment} (bound : Nat) : ∀ {count : Nat}
    (coordinate : Fin count → Result Assignment),
    (∀ index, (coordinate index).work ≤ bound) →
    (collect coordinate).work ≤ count * (bound + 1) + 1
  | 0, _, _ => by simp only [collect, Nat.zero_mul, Nat.zero_add, Nat.le_refl]
  | count + 1, coordinate, bounded => by
      have head := bounded 0
      have tail := collect_work_le bound (fun index => coordinate index.succ)
        (fun index => bounded index.succ)
      change (coordinate 0).work +
          (collect (fun index => coordinate index.succ)).work + 1 ≤ _
      rw [Nat.add_mul]
      omega

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint}
  {Evaluation : Type uEvaluation} {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  {algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params}
  {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}
  (laws : ExtractionAlgebra semantics params algebra)
  (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
  (program : Primitives Scalar Assignment)
  (correct : Correct laws.ring laws.assignmentModule program)
  (fork : CompleteFork semantics params algebra batch)

/-- One ordered traversal of the existing complete coordinate fork. -/
def extractBatch : Result (List Assignment) :=
  collect fun coordinate => extract program
    (fork.base.challenges coordinate) ((fork.forks coordinate).challenges coordinate)
    fork.base.assignment (fork.forks coordinate).assignment

include strongSet correct

/-- Every returned coordinate equals the existing semantic extractor. -/
theorem coordinate_value (coordinate : Fin arity.total) :
    (extract program (fork.base.challenges coordinate)
      ((fork.forks coordinate).challenges coordinate)
      fork.base.assignment (fork.forks coordinate).assignment).value =
      extractedAssignment laws strongSet fork coordinate :=
  extract_value laws.ring laws.assignmentModule program correct _ _ _ _
    (fork.coordinateUnit laws strongSet coordinate)

/-- The batch keeps exact source order and the existing assignment representation. -/
theorem batch_values :
    (extractBatch program fork).value =
      List.ofFn (extractedAssignment laws strongSet fork) := by
  rw [extractBatch, collect_values]
  congr 1
  funext coordinate
  exact coordinate_value laws strongSet program correct fork coordinate

/-- All primitive calls and list traversal are charged for the whole batch. -/
theorem batch_work_le (bounds : PrimitiveBounds)
    (bounded : Bounded laws.ring program bounds) :
    (extractBatch program fork).work ≤
      arity.total * (bounds.coordinateWork + 1) + 1 := by
  apply collect_work_le
  intro coordinate
  exact extract_work_le laws.ring laws.assignmentModule program correct bounds bounded _ _ _ _
    (fork.coordinateUnit laws strongSet coordinate)

/-- B.3 step 7 is already true for every result produced from a CompleteFork.
The charged program therefore returns the same always-accepted assignments. -/
theorem coordinate_correctedAmbient (coordinate : Fin arity.total) :
    PaperCorrections.CorrectedAmbientHolds semantics params (batch.inputs coordinate)
      (extract program (fork.base.challenges coordinate)
        ((fork.forks coordinate).challenges coordinate)
        fork.base.assignment (fork.forks coordinate).assignment).value := by
  rw [coordinate_value laws strongSet program correct fork]
  exact completeFork_implies_correctedAmbientHolds semantics params arity algebra laws strongSet
    batch fork coordinate

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
