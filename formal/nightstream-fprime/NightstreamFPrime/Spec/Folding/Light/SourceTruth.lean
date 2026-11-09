import NightstreamFPrime.Spec.Folding.Light.PiCCS
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership

/-! The premise of the light Π_CCS completeness (`Light.PiCCS.SemanticTruth`) from the source
relations (CW-10.4). The light Π_CCS bounds only the fresh sources' norms, because the light
fold does not decompose the running claim. So the fresh sources' CCS statements and the running
sources' evaluation claims give the truth, with no norm bound on the running sources
(`semanticTruth_of_sources`). This is `SourceMembership.sourceHolds_iff_memberships` without
the running sources' openings. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.PiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership

universe uExtension uCommitment uPublicInput

variable {Extension : Type uExtension} {Commitment : Type uCommitment}
  {PublicInput : Type uPublicInput} {shape : Shape} {columns blockCount : Nat}
  {baseOps : InterpolationOps F}

/-- The fresh sources' CCS statements at `b = 2` and the running sources' evaluation claims give
the light Π_CCS truth. -/
theorem semanticTruth_of_sources (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F → Extension) (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams) (freshBound : params.b = 2)
    (statement : Statement Extension Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns)
    (fresh : ∀ index, CCS.Holds (paperRelationSemantics baseOps extensionOps lift openingMaps)
      params (freshInstance statement index) (witness.assignments (freshSourceIndex index)))
    (running : ∀ index, (paperRelationSemantics baseOps extensionOps lift openingMaps).evaluations
        statement.relationSource (witness.assignments (runningSourceIndex index))
        statement.priorPoint = #[runningEvaluation statement index]) :
    SemanticTruth ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
      baseOps extensionOps lift := by
  refine ⟨fun index => (fresh index).2, ?_, ?_, ?_⟩
  · intro source isFresh column
    rcases source_eq_fresh_or_running source with ⟨index, rfl⟩ | ⟨index, rfl⟩
    · have bounded := (fresh index).1.2.2 column
      change centeredMagnitude (witness.assignments (freshSourceIndex index) column) < params.b
        at bounded
      rw [freshBound] at bounded
      exact bounded
    · exact absurd isFresh (by simp [runningSourceIndex])
  · intro coordinate
    exact ((runningEvaluations_eq_iff extensionOps extensionLaws lift openingMaps
      statement witness coordinate.running).mp (running coordinate.running)).1 coordinate.coefficient
  · intro coordinate
    exact ((runningEvaluations_eq_iff extensionOps extensionLaws lift openingMaps
      statement witness coordinate.running).mp (running coordinate.running)).2
        coordinate.matrix coordinate.coefficient

end NightstreamFPrime.Spec.Folding.Light.PiCCS
