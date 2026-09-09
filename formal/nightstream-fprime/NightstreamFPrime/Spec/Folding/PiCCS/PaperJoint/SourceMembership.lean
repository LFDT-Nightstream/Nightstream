import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Paper authority: SuperNeo v1.1, Section 7.3, source CCS/CE relations.
Owns the exact source-membership meaning of the existing extraction conclusion.
The statement supplies one structure, fresh-then-running order, one prior
point, and the complete separate Pad and matrix evaluation families. The
witness supplies the same assignments used by StrongReduction.SourceHolds.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership

open NightstreamFPrime.Spec
open PaperLinearAlgebra
open StrongReduction
open UnifiedSources

universe uExtension uCommitment uPublicInput

variable {Extension : Type uExtension} {Commitment : Type uCommitment}
  {PublicInput : Type uPublicInput} {shape : Shape} {columns blockCount : Nat}
  {baseOps : InterpolationOps F}

/-- The selected fresh CCS statement, at its exact source position. -/
def freshInstance
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (fresh : Fin shape.freshCount) :
    CCS.Instance (RelationSource shape columns blockCount) PublicInput Commitment where
  constraintSystem := statement.relationSource
  commitment := statement.commitments (freshSourceIndex fresh)
  publicInput := statement.publicInputs (freshSourceIndex fresh)
  stage := .fresh

/-- The complete public evaluation family of one prior source. -/
def runningEvaluation
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (running : Fin shape.runningCount) : EvaluationFamily Extension shape where
  pad := fun coefficient => statement.claimedPadCoefficient ⟨running, coefficient⟩
  matrix := fun matrix coefficient =>
    statement.claimedMatrixCoefficient ⟨running, matrix, coefficient⟩

/-- Every prior CE statement has the same selected structure and point.
Its singleton evaluation array contains all Pad and matrix coefficients. -/
def runningInstance
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (running : Fin shape.runningCount) :
    CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables) (EvaluationFamily Extension shape) Commitment where
  constraintSystem := statement.relationSource
  commitment := statement.commitments (runningSourceIndex running)
  publicInput := statement.publicInputs (runningSourceIndex running)
  point := statement.priorPoint
  evaluations := #[runningEvaluation statement running]
  stage := .fresh

private theorem evaluation_ext (left right : EvaluationFamily Extension shape)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem runningEvaluations_eq_iff
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F → Extension) (openingMaps : OpeningMaps Commitment PublicInput columns)
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns) (running : Fin shape.runningCount) :
    (paperRelationSemantics baseOps extensionOps lift openingMaps).evaluations
        statement.relationSource (witness.assignments (runningSourceIndex running))
        statement.priorPoint = #[runningEvaluation statement running] ↔
      (∀ coefficient, PadEvaluationResidual.EvaluationClaimHolds baseOps extensionOps lift
        ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps).padData
        ⟨running, coefficient⟩) ∧
      (∀ matrix coefficient, MatrixEvaluationResidual.EvaluationClaimHolds baseOps extensionOps lift
        ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps).matrixData
        ⟨running, matrix, coefficient⟩) := by
  let inputs := (statement.sourceConnectedInputs witness).toUnifiedInputs baseOps
  constructor
  · intro equal
    constructor
    · intro coefficient
      have field := congrArg (fun values : Array (EvaluationFamily Extension shape) =>
        (values.getD 0 (runningEvaluation statement running)).pad coefficient) equal
      have computed := PadEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
        baseOps extensionOps extensionLaws lift inputs.padData ⟨running, coefficient⟩
      exact field.symm.trans computed
    · intro matrix coefficient
      have field := congrArg (fun values : Array (EvaluationFamily Extension shape) =>
        (values.getD 0 (runningEvaluation statement running)).matrix matrix coefficient) equal
      have computed := MatrixEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
        baseOps extensionOps extensionLaws lift inputs.matrixData ⟨running, matrix, coefficient⟩
      exact field.symm.trans computed
  · rintro ⟨pad, matrix⟩
    apply congrArg (fun evaluation : EvaluationFamily Extension shape => #[evaluation])
    apply evaluation_ext
    · funext coefficient
      have computed := PadEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
        baseOps extensionOps extensionLaws lift inputs.padData ⟨running, coefficient⟩
      exact computed.trans (pad coefficient).symm
    · funext matrixIndex coefficient
      have computed := MatrixEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
        baseOps extensionOps extensionLaws lift inputs.matrixData ⟨running, matrixIndex, coefficient⟩
      exact computed.trans (matrix matrixIndex coefficient).symm

/-- The existing strong-reduction conclusion is exactly the fresh CCS and
prior CE witness products. No input validity follows from a constructor or a
carried digest. The selected fresh bound is the existing b=2 profile. -/
theorem sourceHolds_iff_memberships
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F → Extension) (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams) (freshBound : params.b = 2)
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns) :
    SourceHolds extensionOps lift openingMaps params statement witness ↔
      (∀ fresh, CCS.Holds (paperRelationSemantics baseOps extensionOps lift openingMaps) params
        (freshInstance statement fresh) (witness.assignments (freshSourceIndex fresh))) ∧
      (∀ running, CE.Holds (paperRelationSemantics baseOps extensionOps lift openingMaps) params
        (runningInstance statement running) (witness.assignments (runningSourceIndex running))) := by
  constructor
  · intro source
    constructor
    · intro fresh
      exact ⟨source.1 (freshSourceIndex fresh), source.2.1 fresh⟩
    · intro running
      refine ⟨source.1 (runningSourceIndex running), trivial, ?_⟩
      exact (runningEvaluations_eq_iff extensionOps extensionLaws lift openingMaps
        statement witness running).mpr
          ⟨fun coefficient => source.2.2.2.1 ⟨running, coefficient⟩,
            fun matrix coefficient => source.2.2.2.2 ⟨running, matrix, coefficient⟩⟩
  · rintro ⟨fresh, running⟩
    have openings : ∀ source, Opening.Holds
        (paperRelationSemantics (shape := shape) (blockCount := blockCount)
          baseOps extensionOps lift openingMaps)
        params.b (statement.commitments source) (statement.publicInputs source)
        (witness.assignments source) := by
      intro source
      rcases source_eq_fresh_or_running source with ⟨index, rfl⟩ | ⟨index, rfl⟩
      · exact (fresh index).1
      · exact (running index).1
    refine ⟨openings, ?_, ?_, ?_, ?_⟩
    · intro index
      exact (fresh index).2
    · intro source column
      have bounded := (openings source).2.2 column
      simpa only [freshBound] using bounded
    · intro coordinate
      exact ((runningEvaluations_eq_iff extensionOps extensionLaws lift openingMaps
        statement witness coordinate.running).mp (running coordinate.running).2.2).1
          coordinate.coefficient
    · intro coordinate
      exact ((runningEvaluations_eq_iff extensionOps extensionLaws lift openingMaps
        statement witness coordinate.running).mp (running coordinate.running).2.2).2
          coordinate.matrix coordinate.coefficient

/-- The existing fixed-width extraction gate returns these exact CCS/CE
memberships on its successful branch. Its two failure events are unchanged. -/
theorem fixedWidthAcceptedProbe_implies_memberships_or_badEvent
    [DecidableEq Extension]
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F → Extension)
    (liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams) (freshBound : params.b = 2)
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
    (width : Nat) (degreeCovers : (statement.verifierInput lift).sumcheckDegreeBound ≤ width)
    (challengeSetSize : Nat) (probe : Probe Extension shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params statement probe witness)
    (accepted : probe.FixedWidthAccepted extensionOps lift statement width) :
    ((∀ fresh, CCS.Holds (paperRelationSemantics baseOps extensionOps lift openingMaps) params
        (freshInstance statement fresh) (witness.assignments (freshSourceIndex fresh))) ∧
      (∀ running, CE.Holds (paperRelationSemantics baseOps extensionOps lift openingMaps) params
        (runningInstance statement running) (witness.assignments (runningSourceIndex running)))) ∨
      SignedCoefficientObject.MixingRoot extensionOps
        ((statement.sourceProtocolData lift witness).toJointData extensionOps)
        probe.coins.alpha probe.coins.gamma ∨
      FixedWidthSumCheckFailure extensionOps lift statement width challengeSetSize probe witness := by
  rcases fixedWidthAcceptedProbe_extracts_source_or_badEvent baseLaws baseZero noZeroDivisors
      extensionOps extensionLaws extensionZeroLaws lift liftLaws openingMaps params freshBound
      statement constantLaw width degreeCovers challengeSetSize probe witness ambient accepted with
    source | failure
  · exact Or.inl ((sourceHolds_iff_memberships extensionOps extensionLaws lift openingMaps
      params freshBound statement witness).mp source)
  · exact Or.inr failure

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership
