import Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction

/-!
Direct relaxed-binding collision reduction for two paper `Pi_RLC` forks.

Protocol: SuperNeo `Pi_RLC` weak reduction (Lemma 4 and Appendix D.5).
Phase: uniqueness of two successful coordinate-fork extractions.
Constraint family: none; this file emits no rows.

Owns: exact fork deltas and subtraction openings, their commitment equations,
their `2B` norm bounds, and the crossed-opening inequality obtained from two
different extracted assignment families.

Does not own: a caller-provided uniqueness bridge, source validity,
probabilistic forking, commitment hardness, concrete Phi81 algebra, Rust,
R1CS, row removal, or constraint counts.

Authority boundary: both public output families are computed by
`PaperForkExtraction.Response.output`.  A collision is constructed directly
from two complete forks at the same `PiRLC.phi`; it is never a premise.
-/

namespace Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

/-- The exact difference between the base challenge and one coordinate fork. -/
def forkDelta
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    (laws : ExtractionAlgebra semantics params algebra)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) : Scalar :=
  laws.ring.sub
    (fork.base.challenges coordinate)
    ((fork.forks coordinate).challenges coordinate)

/-- The exact difference between the base output assignment and one
coordinate-fork output assignment. -/
def forkOpening
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    (laws : ExtractionAlgebra semantics params algebra)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) : Assignment :=
  laws.assignmentModule.sub
    fork.base.assignment (fork.forks coordinate).assignment

/-- The only extra laws needed to instantiate Definition 4's collision
carrier from the paper extraction algebra. -/
structure RelaxedBindingLaws
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (laws : ExtractionAlgebra semantics params algebra)
    (ops : RelaxedBindingOps Assignment Commitment Scalar) : Prop where
  scaleAssignment_eq : forall scalar assignment,
    ops.scaleAssignment scalar assignment =
      laws.assignmentModule.smul scalar assignment
  scaleCommitment_eq : forall scalar commitment,
    ops.scaleCommitment scalar commitment =
      laws.commitmentModule.smul scalar commitment
  differenceChallenge_sub : forall left right,
    algebra.challengeValid left -> algebra.challengeValid right ->
      ops.differenceChallenge (laws.ring.sub left right)
  sub_norm : forall left right,
    semantics.normBounded params.bigB left ->
    semantics.normBounded params.bigB right ->
    semantics.normBounded (2 * params.bigB)
      (laws.assignmentModule.sub left right)

private theorem forkOpening_commitment
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    (laws : ExtractionAlgebra semantics params algebra)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) :
    semantics.commit (forkOpening laws fork coordinate) =
      laws.commitmentModule.smul (forkDelta laws fork coordinate)
        (batch.inputs coordinate).commitment := by
  unfold forkOpening forkDelta
  calc
    semantics.commit
        (laws.assignmentModule.sub
          fork.base.assignment (fork.forks coordinate).assignment) =
      laws.commitmentModule.sub
        (semantics.commit fork.base.assignment)
        (semantics.commit (fork.forks coordinate).assignment) :=
          laws.commitMap.map_sub _ _
    _ = laws.commitmentModule.sub
        (algebra.combineCommitment fork.base.challenges
          (fun index => (batch.inputs index).commitment))
        (algebra.combineCommitment (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).commitment)) := by
            rw [fork.baseSuccess.1.1, (fork.forkSuccess coordinate).1.1]
            rfl
    _ = laws.commitmentModule.sub
        (PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          fork.base.challenges
          (fun index => (batch.inputs index).commitment))
        (PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).commitment)) := by
            rw [laws.combineCommitment_eq, laws.combineCommitment_eq]
    _ = laws.commitmentModule.smul
        (laws.ring.sub
          (fork.base.challenges coordinate)
          ((fork.forks coordinate).challenges coordinate))
        (batch.inputs coordinate).commitment :=
      PaperForkAlgebra.coordinateIsolation laws.ring laws.commitmentModule
        laws.ringLaws laws.commitmentLaws fork.base.challenges
        (fork.forks coordinate).challenges
        (fun index => (batch.inputs index).commitment) coordinate
        (fork.agreeExcept coordinate)

private theorem forkOpening_norm
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    {ops : RelaxedBindingOps Assignment Commitment Scalar}
    (laws : ExtractionAlgebra semantics params algebra)
    (bindingLaws : RelaxedBindingLaws semantics params algebra laws ops)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) :
    semantics.normBounded (2 * params.bigB)
      (forkOpening laws fork coordinate) := by
  unfold forkOpening
  exact bindingLaws.sub_norm _ _
    fork.baseSuccess.1.2.2 (fork.forkSuccess coordinate).1.2.2

/-- One relaxed-binding collision receipt whose scalars and openings are
definitionally the deltas and output-opening differences of the two named
coordinate forks.  This is the execution-dependent carrier used by the
probabilistic reduction; it cannot be populated by an unrelated global
collision witness. -/
structure CoordinateForkCollisionReceipt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch₁ batch₂ : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    (laws : ExtractionAlgebra semantics params algebra)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (fork₁ : CompleteFork semantics params algebra batch₁)
    (fork₂ : CompleteFork semantics params algebra batch₂)
    (coordinate : Fin arity.total) : Prop where
  delta₁Valid : ops.differenceChallenge (forkDelta laws fork₁ coordinate)
  delta₂Valid : ops.differenceChallenge (forkDelta laws fork₂ coordinate)
  firstEquation :
    ops.scaleCommitment (forkDelta laws fork₁ coordinate)
        (batch₁.inputs coordinate).commitment =
      semantics.commit (forkOpening laws fork₁ coordinate)
  secondEquation :
    ops.scaleCommitment (forkDelta laws fork₂ coordinate)
        (batch₁.inputs coordinate).commitment =
      semantics.commit (forkOpening laws fork₂ coordinate)
  firstNorm : semantics.normBounded (2 * params.bigB)
    (forkOpening laws fork₁ coordinate)
  secondNorm : semantics.normBounded (2 * params.bigB)
    (forkOpening laws fork₂ coordinate)
  crossDifferent :
    ops.scaleAssignment (forkDelta laws fork₁ coordinate)
        (forkOpening laws fork₂ coordinate) ≠
      ops.scaleAssignment (forkDelta laws fork₂ coordinate)
        (forkOpening laws fork₁ coordinate)

namespace CoordinateForkCollisionReceipt

/-- Forget only the execution provenance while preserving the exact fork
values as the literal Definition-4 collision fields. -/
def toRelaxedBindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params}
    {batch₁ batch₂ : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity}
    {laws : ExtractionAlgebra semantics params algebra}
    {ops : RelaxedBindingOps Assignment Commitment Scalar}
    {fork₁ : CompleteFork semantics params algebra batch₁}
    {fork₂ : CompleteFork semantics params algebra batch₂}
    {coordinate : Fin arity.total}
    (receipt : CoordinateForkCollisionReceipt laws ops fork₁ fork₂ coordinate) :
    RelaxedBindingCollision semantics params ops
      (batch₁.inputs coordinate).commitment where
  delta₁ := forkDelta laws fork₁ coordinate
  delta₂ := forkDelta laws fork₂ coordinate
  opening₁ := forkOpening laws fork₁ coordinate
  opening₂ := forkOpening laws fork₂ coordinate
  delta₁Valid := receipt.delta₁Valid
  delta₂Valid := receipt.delta₂Valid
  firstEquation := receipt.firstEquation
  secondEquation := receipt.secondEquation
  firstNorm := receipt.firstNorm
  secondNorm := receipt.secondNorm
  crossDifferent := receipt.crossDifferent

end CoordinateForkCollisionReceipt

private theorem inverse_actions_equal_of_cross_actions_equal
    {Scalar : Type uScalar}
    {Assignment : Type uAssignment}
    (ring : PaperForkAlgebra.CommutativeRingOps Scalar)
    (module : PaperForkAlgebra.ModuleOps Scalar Assignment)
    (ringLaws : PaperForkAlgebra.CommutativeRingLaws ring)
    (moduleLaws : PaperForkAlgebra.ModuleLaws ring module)
    (delta₁ delta₂ : Scalar)
    (unit₁ : PaperForkAlgebra.UnitWitness ring delta₁)
    (unit₂ : PaperForkAlgebra.UnitWitness ring delta₂)
    (opening₁ opening₂ : Assignment)
    (crossEqual : module.smul delta₁ opening₂ =
      module.smul delta₂ opening₁) :
    module.smul unit₁.inverse opening₁ =
      module.smul unit₂.inverse opening₂ := by
  have opening₁_eq :
      opening₁ = module.smul unit₂.inverse
        (module.smul delta₁ opening₂) := by
    calc
      opening₁ = module.smul unit₂.inverse
          (module.smul delta₂ opening₁) :=
        (PaperForkAlgebra.inverseActionCancellation ring module moduleLaws
          delta₂ unit₂ opening₁).symm
      _ = module.smul unit₂.inverse
          (module.smul delta₁ opening₂) := by rw [← crossEqual]
  calc
    module.smul unit₁.inverse opening₁ =
      module.smul unit₁.inverse
        (module.smul unit₂.inverse
          (module.smul delta₁ opening₂)) := by rw [opening₁_eq]
    _ = module.smul (ring.mul unit₁.inverse unit₂.inverse)
        (module.smul delta₁ opening₂) :=
      (moduleLaws.mul_smul _ _ _).symm
    _ = module.smul
        (ring.mul (ring.mul unit₁.inverse unit₂.inverse) delta₁)
        opening₂ :=
      (moduleLaws.mul_smul _ _ _).symm
    _ = module.smul
        (ring.mul unit₂.inverse (ring.mul unit₁.inverse delta₁))
        opening₂ := by
          rw [ringLaws.mul_comm unit₁.inverse unit₂.inverse,
            ringLaws.mul_assoc]
    _ = module.smul (ring.mul unit₂.inverse ring.one) opening₂ := by
      rw [unit₁.inverse_mul]
    _ = module.smul unit₂.inverse opening₂ := by
      rw [ringLaws.mul_one]

/-- A difference at one explicit coordinate constructively produces the exact
collision receipt emitted by those two accepted forks.  No coordinate search
or unrelated collision witness occurs in this theorem. -/
theorem coordinate_differingExtractions_imply_collisionReceipt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (arity : BatchArity params)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws semantics params algebra laws ops)
    (leftBatch rightBatch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity)
    (leftFork : CompleteFork semantics params algebra leftBatch)
    (rightFork : CompleteFork semantics params algebra rightBatch)
    (samePhi : PiRLC.phi leftBatch.inputs = PiRLC.phi rightBatch.inputs)
    (coordinate : Fin arity.total)
    (different :
      extractedAssignment laws strongSet leftFork coordinate ≠
        extractedAssignment laws strongSet rightFork coordinate) :
    CoordinateForkCollisionReceipt laws ops leftFork rightFork coordinate := by
  let delta₁ := forkDelta laws leftFork coordinate
  let delta₂ := forkDelta laws rightFork coordinate
  let opening₁ := forkOpening laws leftFork coordinate
  let opening₂ := forkOpening laws rightFork coordinate
  let unit₁ := leftFork.coordinateUnit laws strongSet coordinate
  let unit₂ := rightFork.coordinateUnit laws strongSet coordinate
  have sameCommitment :
      (leftBatch.inputs coordinate).commitment =
        (rightBatch.inputs coordinate).commitment :=
    congrFun samePhi coordinate
  have firstOpeningCommitment :
      semantics.commit opening₁ =
        laws.commitmentModule.smul delta₁
          (leftBatch.inputs coordinate).commitment :=
    forkOpening_commitment laws leftFork coordinate
  have secondOpeningCommitment :
      semantics.commit opening₂ =
        laws.commitmentModule.smul delta₂
          (rightBatch.inputs coordinate).commitment :=
    forkOpening_commitment laws rightFork coordinate
  have crossDifferent :
      ops.scaleAssignment delta₁ opening₂ ≠
        ops.scaleAssignment delta₂ opening₁ := by
    intro crossEqual
    apply different
    unfold extractedAssignment
    apply inverse_actions_equal_of_cross_actions_equal laws.ring
      laws.assignmentModule laws.ringLaws laws.assignmentLaws
      delta₁ delta₂ unit₁ unit₂ opening₁ opening₂
    simpa only [bindingLaws.scaleAssignment_eq] using crossEqual
  exact {
    delta₁Valid := bindingLaws.differenceChallenge_sub _ _
      (leftFork.baseStrong coordinate)
      (leftFork.forkStrong coordinate coordinate)
    delta₂Valid := bindingLaws.differenceChallenge_sub _ _
      (rightFork.baseStrong coordinate)
      (rightFork.forkStrong coordinate coordinate)
    firstEquation := by
      rw [bindingLaws.scaleCommitment_eq]
      exact firstOpeningCommitment.symm
    secondEquation := by
      rw [bindingLaws.scaleCommitment_eq, sameCommitment]
      exact secondOpeningCommitment.symm
    firstNorm := forkOpening_norm laws bindingLaws leftFork coordinate
    secondNorm := forkOpening_norm laws bindingLaws rightFork coordinate
    crossDifferent := crossDifferent
  }

/-- Two differing operational extractions at the same commitment vector
construct a literal indexed `(2B, C)` relaxed-binding collision. -/
theorem samePhi_differingExtractions_imply_relaxedBindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (arity : BatchArity params)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws semantics params algebra laws ops)
    (leftBatch rightBatch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity)
    (leftFork : CompleteFork semantics params algebra leftBatch)
    (rightFork : CompleteFork semantics params algebra rightBatch)
    (samePhi : PiRLC.phi leftBatch.inputs = PiRLC.phi rightBatch.inputs)
    (different :
      (fun coordinate =>
        extractedAssignment laws strongSet leftFork coordinate) ≠
      (fun coordinate =>
        extractedAssignment laws strongSet rightFork coordinate)) :
    exists coordinate, Nonempty
      (RelaxedBindingCollision semantics params ops
        (leftBatch.inputs coordinate).commitment) := by
  classical
  have indexedDifferent : exists coordinate,
      extractedAssignment laws strongSet leftFork coordinate ≠
        extractedAssignment laws strongSet rightFork coordinate := by
    exact Classical.byContradiction fun noCoordinate => different (by
      funext coordinate
      exact Classical.byContradiction fun coordinateDifferent =>
        noCoordinate ⟨coordinate, coordinateDifferent⟩)
  rcases indexedDifferent with ⟨coordinate, extractedDifferent⟩
  let receipt := coordinate_differingExtractions_imply_collisionReceipt
    semantics params arity algebra laws strongSet ops bindingLaws leftBatch
    rightBatch leftFork rightFork samePhi coordinate extractedDifferent
  exact ⟨coordinate,
    ⟨CoordinateForkCollisionReceipt.toRelaxedBindingCollision receipt⟩⟩

end Nightstream.SuperNeo.Folding.PiRLC.PaperForkCollision
