import NightstreamFPrime.Spec.Folding.PiRLC
import NightstreamFPrime.Spec.Folding.PiRLC.PaperCorrections
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiRLC/PaperForkExtraction.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Operational coordinate-fork extraction for the paper `Pi_RLC` weak reduction.

Protocol: SuperNeo `Pi_RLC` (Lemma 4 and Appendix D.5).
Phase: deterministic extraction from one complete coordinate fork.
Constraint family: none; this file emits no rows.

Owns: a shared-system/shared-point input batch, prover responses whose public
outputs are computed by `PiRLC.combinedOutput`, the exact special-set fork
shape, the Appendix D.5 inverse-difference extractor, and corrected ambient
membership of every extracted source opening.

Does not own: the probabilistic forking lemma, transcript replay, relaxed
binding uniqueness, source-relation validity, concrete Phi81 algebra, Rust,
R1CS, row removal, or constraint counts.

Authority boundary: public outputs are definitions, not prover fields.
Successful response assignments must open those computed `CE(B)` outputs.  The only
additional premises are explicit ring/module homomorphism laws, strong-set
unit production, and universal coverage by the corrected strict ambient norm.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

open NightstreamFPrime.Spec

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uSource uTarget

/-- One fixed Π_RLC input batch.  Every source statement uses the shared
constraint system and evaluation point that the verifier passes to
`PiRLC.combinedOutput`. -/
structure InputBatch
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) where
  system : Structure
  point : Point
  inputs : Fin arity.total ->
    CE.Instance Structure PublicInput Point Evaluation Commitment
  sameSystem : forall index,
    (inputs index).constraintSystem = system
  samePoint : forall index, (inputs index).point = point
  evaluationCount : Nat
  evaluationsSize : forall index,
    (inputs index).evaluations.size = evaluationCount

/-- One prover response to one verifier challenge vector.  The public output
is intentionally absent and is computed below. -/
structure Response
    (Assignment : Type uAssignment)
    (Scalar : Type uScalar)
    (params : GlobalParams)
    (arity : BatchArity params) where
  challenges : Fin arity.total -> Scalar
  assignment : Assignment

namespace Response

/-- The verifier-owned public output for this response. -/
def output
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
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity)
    (response : Response Assignment Scalar params arity) :
    CE.Instance Structure PublicInput Point Evaluation Commitment :=
  PiRLC.combinedOutput algebra batch.system batch.point batch.inputs
    response.challenges

/-- Operational success: the response assignment opens the exact computed
public output in Π_RLC's `CE(B)` target relation.  Corrected ambient membership
is the extractor's source conclusion, not the verifier's target check. -/
def Success
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
    {arity : BatchArity params}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity)
    (response : Response Assignment Scalar params arity) : Prop :=
  CE.Holds semantics params (response.output algebra batch) response.assignment

end Response

/-- The strong-set fact used by Appendix D.5: two distinct valid challenges
have an invertible difference.  No separate nonzero premise is accepted by
the extractor. -/
structure StrongSetUnits
    {Scalar : Type uScalar}
    (ring : PaperForkAlgebra.CommutativeRingOps Scalar)
    (member : Scalar -> Prop) where
  differenceUnit : forall {left right},
    member left -> member right -> left ≠ right ->
      PaperForkAlgebra.UnitWitness ring (ring.sub left right)

/-- Preservation of subtraction and scalar action by one semantic map. -/
structure LinearMapLaws
    {Scalar : Type uScalar}
    {Source : Type uSource}
    {Target : Type uTarget}
    (source : PaperForkAlgebra.ModuleOps Scalar Source)
    (target : PaperForkAlgebra.ModuleOps Scalar Target)
    (map : Source -> Target) : Prop where
  map_sub : forall left right,
    map (source.sub left right) = target.sub (map left) (map right)
  map_smul : forall scalar value,
    map (source.smul scalar value) = target.smul scalar (map value)

/-- Paper algebra needed to turn two successful output openings into one
source opening.  Every verifier combination is identified with the canonical
finite combination from `PaperForkAlgebra`; semantic maps preserve the exact
assignment subtraction and action used by the extractor. -/
structure ExtractionAlgebra
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
      semantics params) where
  ring : PaperForkAlgebra.CommutativeRingOps Scalar
  ringLaws : PaperForkAlgebra.CommutativeRingLaws ring
  assignmentModule : PaperForkAlgebra.ModuleOps Scalar Assignment
  assignmentLaws : PaperForkAlgebra.ModuleLaws ring assignmentModule
  commitmentModule : PaperForkAlgebra.ModuleOps Scalar Commitment
  commitmentLaws : PaperForkAlgebra.ModuleLaws ring commitmentModule
  publicInputModule : PaperForkAlgebra.ModuleOps Scalar PublicInput
  publicInputLaws : PaperForkAlgebra.ModuleLaws ring publicInputModule
  evaluationModule : PaperForkAlgebra.ModuleOps Scalar Evaluation
  evaluationLaws : PaperForkAlgebra.ModuleLaws ring evaluationModule
  combineCommitment_eq : forall {count}
      (coefficients : Fin count -> Scalar)
      (values : Fin count -> Commitment),
    algebra.combineCommitment coefficients values =
      PaperForkAlgebra.linearCombination ring commitmentModule coefficients values
  combinePublicInput_eq : forall {count}
      (coefficients : Fin count -> Scalar)
      (values : Fin count -> PublicInput),
    algebra.combinePublicInput coefficients values =
      PaperForkAlgebra.linearCombination ring publicInputModule coefficients values
  semanticEvaluations_size_eq : forall system point left right,
    (semantics.evaluations system left point).size =
      (semantics.evaluations system right point).size
  combineEvaluations_size : forall {count}
      (coefficients : Fin count -> Scalar)
      (values : Fin count -> Array Evaluation)
      (expectedSize : Nat),
    0 < count ->
    (forall index, (values index).size = expectedSize) ->
      (algebra.combineEvaluations coefficients values).size =
        expectedSize
  combineEvaluations_getD : forall {count}
      (coefficients : Fin count -> Scalar)
      (values : Fin count -> Array Evaluation)
      (expectedSize index : Nat),
    0 < count ->
    (forall source, (values source).size = expectedSize) ->
    (algebra.combineEvaluations coefficients values).getD index
        evaluationModule.zero =
      PaperForkAlgebra.linearCombination ring evaluationModule coefficients
        (fun source => (values source).getD index evaluationModule.zero)
  commitMap : LinearMapLaws assignmentModule commitmentModule semantics.commit
  publicInputMap : LinearMapLaws assignmentModule publicInputModule
    semantics.projectPublicInput
  evaluationsMap : forall system point index,
    LinearMapLaws assignmentModule evaluationModule
      (fun assignment =>
        (semantics.evaluations system assignment point).getD index
          evaluationModule.zero)
  correctedNormCoverage : forall assignment,
    semantics.normBounded (PaperCorrections.correctedAmbientBoundFor params)
      assignment

/-- A base response and exactly one special-set fork per coordinate.  Each
fork changes its named coordinate and agrees everywhere else. -/
structure CompleteFork
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
    {arity : BatchArity params}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity) where
  base : Response Assignment Scalar params arity
  forks : Fin arity.total -> Response Assignment Scalar params arity
  baseSuccess : base.Success semantics params algebra batch
  forkSuccess : forall coordinate,
    (forks coordinate).Success semantics params algebra batch
  baseStrong : forall index, algebra.challengeValid (base.challenges index)
  forkStrong : forall coordinate index,
    algebra.challengeValid ((forks coordinate).challenges index)
  agreeExcept : forall coordinate,
    PaperForkAlgebra.AgreeExcept coordinate base.challenges
      (forks coordinate).challenges
  changed : forall coordinate,
    base.challenges coordinate ≠ (forks coordinate).challenges coordinate

namespace CompleteFork

/-- Unit witness for one fork delta, derived only from strong-set membership
and the fork's explicit changed-coordinate fact. -/
def coordinateUnit
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
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) :
    PaperForkAlgebra.UnitWitness laws.ring
      (laws.ring.sub
        (fork.base.challenges coordinate)
        ((fork.forks coordinate).challenges coordinate)) :=
  strongSet.differenceUnit
    (fork.baseStrong coordinate)
    (fork.forkStrong coordinate coordinate)
    (fork.changed coordinate)

end CompleteFork

/-- Appendix D.5's extracted assignment at one coordinate. -/
def extractedAssignment
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
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : CompleteFork semantics params algebra batch)
    (coordinate : Fin arity.total) : Assignment :=
  let unit := fork.coordinateUnit laws strongSet coordinate
  laws.assignmentModule.smul unit.inverse
    (laws.assignmentModule.sub
      fork.base.assignment (fork.forks coordinate).assignment)

private theorem extracted_map_eq
    {Scalar : Type uScalar}
    {Source : Type uSource}
    {Target : Type uTarget}
    (ring : PaperForkAlgebra.CommutativeRingOps Scalar)
    (ringLaws : PaperForkAlgebra.CommutativeRingLaws ring)
    (sourceModule : PaperForkAlgebra.ModuleOps Scalar Source)
    (targetModule : PaperForkAlgebra.ModuleOps Scalar Target)
    (targetLaws : PaperForkAlgebra.ModuleLaws ring targetModule)
    (map : Source -> Target)
    (mapLaws : LinearMapLaws sourceModule targetModule map)
    {count : Nat}
    (baseCoefficients forkCoefficients : Fin count -> Scalar)
    (inputValues : Fin count -> Target)
    (coordinate : Fin count)
    (agree : PaperForkAlgebra.AgreeExcept coordinate
      baseCoefficients forkCoefficients)
    (unit : PaperForkAlgebra.UnitWitness ring
      (ring.sub
        (baseCoefficients coordinate)
        (forkCoefficients coordinate)))
    (baseAssignment forkAssignment : Source)
    (baseImage : map baseAssignment =
      PaperForkAlgebra.linearCombination ring targetModule
        baseCoefficients inputValues)
    (forkImage : map forkAssignment =
      PaperForkAlgebra.linearCombination ring targetModule
        forkCoefficients inputValues) :
    map (sourceModule.smul unit.inverse
        (sourceModule.sub baseAssignment forkAssignment)) =
      inputValues coordinate := by
  calc
    map (sourceModule.smul unit.inverse
        (sourceModule.sub baseAssignment forkAssignment)) =
      targetModule.smul unit.inverse
        (targetModule.sub (map baseAssignment) (map forkAssignment)) := by
          rw [mapLaws.map_smul, mapLaws.map_sub]
    _ = targetModule.smul unit.inverse
        (targetModule.sub
          (PaperForkAlgebra.linearCombination ring targetModule
            baseCoefficients inputValues)
          (PaperForkAlgebra.linearCombination ring targetModule
            forkCoefficients inputValues)) := by
              rw [baseImage, forkImage]
    _ = targetModule.smul unit.inverse
        (targetModule.smul
          (ring.sub
            (baseCoefficients coordinate)
            (forkCoefficients coordinate))
          (inputValues coordinate)) := by
            rw [PaperForkAlgebra.coordinateIsolation ring targetModule
              ringLaws targetLaws baseCoefficients forkCoefficients
              inputValues coordinate agree]
    _ = inputValues coordinate :=
      PaperForkAlgebra.inverseActionCancellation ring targetModule targetLaws
        (ring.sub
          (baseCoefficients coordinate)
          (forkCoefficients coordinate))
        unit (inputValues coordinate)

/-- A complete operational coordinate fork extracts a corrected ambient CE
opening for every source input.  No source-validity or desired-opening premise
is accepted. -/
theorem completeFork_implies_correctedAmbientHolds
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
    (batch : InputBatch
      Structure PublicInput Point Evaluation Commitment params arity)
    (fork : CompleteFork semantics params algebra batch) :
    forall coordinate,
      PaperCorrections.CorrectedAmbientHolds semantics params
        (batch.inputs coordinate)
        (extractedAssignment laws strongSet fork coordinate) := by
  intro coordinate
  let unit := fork.coordinateUnit laws strongSet coordinate
  let extracted := extractedAssignment laws strongSet fork coordinate
  have baseCommitment :
      semantics.commit fork.base.assignment =
        PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          fork.base.challenges (fun index => (batch.inputs index).commitment) := by
    calc
      semantics.commit fork.base.assignment =
          algebra.combineCommitment fork.base.challenges
            (fun index => (batch.inputs index).commitment) := by
              exact fork.baseSuccess.1.1
      _ = PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          fork.base.challenges (fun index => (batch.inputs index).commitment) :=
        laws.combineCommitment_eq _ _
  have forkCommitment :
      semantics.commit (fork.forks coordinate).assignment =
        PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).commitment) := by
    calc
      semantics.commit (fork.forks coordinate).assignment =
          algebra.combineCommitment (fork.forks coordinate).challenges
            (fun index => (batch.inputs index).commitment) := by
              exact (fork.forkSuccess coordinate).1.1
      _ = PaperForkAlgebra.linearCombination laws.ring laws.commitmentModule
          (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).commitment) :=
        laws.combineCommitment_eq _ _
  have basePublicInput :
      semantics.projectPublicInput fork.base.assignment =
        PaperForkAlgebra.linearCombination laws.ring laws.publicInputModule
          fork.base.challenges (fun index => (batch.inputs index).publicInput) := by
    calc
      semantics.projectPublicInput fork.base.assignment =
          algebra.combinePublicInput fork.base.challenges
            (fun index => (batch.inputs index).publicInput) := by
              exact fork.baseSuccess.1.2.1
      _ = PaperForkAlgebra.linearCombination laws.ring laws.publicInputModule
          fork.base.challenges (fun index => (batch.inputs index).publicInput) :=
        laws.combinePublicInput_eq _ _
  have forkPublicInput :
      semantics.projectPublicInput (fork.forks coordinate).assignment =
        PaperForkAlgebra.linearCombination laws.ring laws.publicInputModule
          (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).publicInput) := by
    calc
      semantics.projectPublicInput (fork.forks coordinate).assignment =
          algebra.combinePublicInput (fork.forks coordinate).challenges
            (fun index => (batch.inputs index).publicInput) := by
              exact (fork.forkSuccess coordinate).1.2.1
      _ = PaperForkAlgebra.linearCombination laws.ring laws.publicInputModule
          (fork.forks coordinate).challenges
          (fun index => (batch.inputs index).publicInput) :=
        laws.combinePublicInput_eq _ _
  have extractedCommitment :
      semantics.commit extracted = (batch.inputs coordinate).commitment := by
    exact extracted_map_eq laws.ring laws.ringLaws laws.assignmentModule
      laws.commitmentModule laws.commitmentLaws
      semantics.commit laws.commitMap fork.base.challenges
      (fork.forks coordinate).challenges
      (fun index => (batch.inputs index).commitment) coordinate
      (fork.agreeExcept coordinate) unit fork.base.assignment
      (fork.forks coordinate).assignment baseCommitment forkCommitment
  have extractedPublicInput :
      semantics.projectPublicInput extracted =
        (batch.inputs coordinate).publicInput := by
    exact extracted_map_eq laws.ring laws.ringLaws laws.assignmentModule
      laws.publicInputModule laws.publicInputLaws
      semantics.projectPublicInput laws.publicInputMap fork.base.challenges
      (fork.forks coordinate).challenges
      (fun index => (batch.inputs index).publicInput) coordinate
      (fork.agreeExcept coordinate) unit fork.base.assignment
      (fork.forks coordinate).assignment basePublicInput forkPublicInput
  have baseEvaluationEquation :
      semantics.evaluations batch.system fork.base.assignment batch.point =
        algebra.combineEvaluations fork.base.challenges
          (fun source => (batch.inputs source).evaluations) := by
    exact fork.baseSuccess.2.2
  have forkEvaluationEquation :
      semantics.evaluations batch.system
          (fork.forks coordinate).assignment batch.point =
        algebra.combineEvaluations (fork.forks coordinate).challenges
          (fun source => (batch.inputs source).evaluations) := by
    exact (fork.forkSuccess coordinate).2.2
  have extractedEvaluations :
      semantics.evaluations batch.system extracted batch.point =
        (batch.inputs coordinate).evaluations := by
    apply Array.ext
    · calc
        (semantics.evaluations batch.system extracted batch.point).size =
            (semantics.evaluations batch.system fork.base.assignment
              batch.point).size :=
          laws.semanticEvaluations_size_eq _ _ _ _
        _ = (algebra.combineEvaluations fork.base.challenges
              (fun index => (batch.inputs index).evaluations)).size :=
          congrArg Array.size fork.baseSuccess.2.2
        _ = batch.evaluationCount :=
          laws.combineEvaluations_size fork.base.challenges
            (fun index => (batch.inputs index).evaluations)
            batch.evaluationCount arity.totalPositive batch.evaluationsSize
        _ = (batch.inputs coordinate).evaluations.size :=
          (batch.evaluationsSize coordinate).symm
    · intro index extractedLt sourceLt
      have baseEvaluationsAt :
          (semantics.evaluations batch.system fork.base.assignment
              batch.point).getD index laws.evaluationModule.zero =
            PaperForkAlgebra.linearCombination laws.ring
              laws.evaluationModule fork.base.challenges
              (fun source =>
                (batch.inputs source).evaluations.getD index
                  laws.evaluationModule.zero) := by
        calc
          (semantics.evaluations batch.system fork.base.assignment
              batch.point).getD index laws.evaluationModule.zero =
              (algebra.combineEvaluations fork.base.challenges
                (fun source => (batch.inputs source).evaluations)).getD
                  index laws.evaluationModule.zero := by
            rw [baseEvaluationEquation]
          _ = _ := laws.combineEvaluations_getD _ _ batch.evaluationCount _
            arity.totalPositive batch.evaluationsSize
      have forkEvaluationsAt :
          (semantics.evaluations batch.system
              (fork.forks coordinate).assignment batch.point).getD index
                laws.evaluationModule.zero =
            PaperForkAlgebra.linearCombination laws.ring
              laws.evaluationModule (fork.forks coordinate).challenges
              (fun source =>
                (batch.inputs source).evaluations.getD index
                  laws.evaluationModule.zero) := by
        calc
          (semantics.evaluations batch.system
              (fork.forks coordinate).assignment batch.point).getD index
                laws.evaluationModule.zero =
              (algebra.combineEvaluations
                (fork.forks coordinate).challenges
                (fun source => (batch.inputs source).evaluations)).getD
                  index laws.evaluationModule.zero := by
            rw [forkEvaluationEquation]
          _ = _ := laws.combineEvaluations_getD _ _ batch.evaluationCount _
            arity.totalPositive batch.evaluationsSize
      have extractedAt :=
        extracted_map_eq laws.ring laws.ringLaws laws.assignmentModule
          laws.evaluationModule laws.evaluationLaws
          (fun assignment =>
            (semantics.evaluations batch.system assignment batch.point).getD
              index laws.evaluationModule.zero)
          (laws.evaluationsMap batch.system batch.point index)
          fork.base.challenges (fork.forks coordinate).challenges
          (fun source =>
            (batch.inputs source).evaluations.getD index
              laws.evaluationModule.zero)
          coordinate (fork.agreeExcept coordinate) unit
          fork.base.assignment (fork.forks coordinate).assignment
          baseEvaluationsAt forkEvaluationsAt
      have extractedAt' :
          (semantics.evaluations batch.system extracted batch.point).getD
              index laws.evaluationModule.zero =
            (batch.inputs coordinate).evaluations.getD index
              laws.evaluationModule.zero := by
        simpa [extracted, extractedAssignment] using extractedAt
      simpa [Array.getD_eq_getD_getElem?,
        Array.getElem?_eq_getElem extractedLt,
        Array.getElem?_eq_getElem sourceLt] using extractedAt'
  have pointValid :
      semantics.evaluationPointValid batch.system batch.point :=
    fork.baseSuccess.2.1
  unfold PaperCorrections.CorrectedAmbientHolds
  refine ⟨⟨extractedCommitment, extractedPublicInput,
    laws.correctedNormCoverage extracted⟩, ?_, ?_⟩
  · rw [batch.sameSystem coordinate, batch.samePoint coordinate]
    exact pointValid
  · rw [batch.sameSystem coordinate, batch.samePoint coordinate]
    exact extractedEvaluations

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
