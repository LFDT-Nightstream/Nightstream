import NightstreamFPrime.Export.Stage1.PiRLCWitnessBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Folding.PiRLC.PaperCompleteness

/-!
The prepared witness replay returns the assignment of the existing honest
PiRLC response for the selected Stage 1 algebra and fixed production key.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCWitnessHonestResponse

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1.PiRLCNonzero (SourceCount)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Export.Stage1.PiRLCWitnessBlock

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private def context : PiRLC.PaperCompleteness.Context
    (PaperAlgebra.Structure selectedShape.logicalWidth)
    (Phi81Relation.Assignment selectedShape)
    (Phi81Relation.PublicInput selectedShape)
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment RingF where
  semantics := PaperAlgebra.semantics Poseidon2HashChainV1Setup.productionAjtaiKey
  params := productionGlobalParams
  arity := Nifs.PaperProfile.arity
  algebra := PaperAlgebra.piRlcAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey
  evaluationCount := (PaperAlgebra.evaluationArity Poseidon2HashChainV1Setup.productionAjtaiKey).count
  evaluationsSize := (PaperAlgebra.evaluationArity Poseidon2HashChainV1Setup.productionAjtaiKey).evaluations_size

/-- The selected honest prover's assignment is definitionally the same
complete-carrier combination used by the executable block theorem. -/
theorem honestResponse_assignment
    (assignments : Fin SourceCount → Phi81Relation.Assignment selectedShape)
    (challenges : Fin SourceCount → RingF)
    (valid : ∀ source, Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (challenges source)) :
    (PiRLC.PaperCompleteness.honestResponse context assignments ⟨challenges, valid⟩).assignment =
      PiRLCFinite.combineAssignments challenges assignments := rfl

/-- The last prepared block is the corresponding block of the existing
honest response. No source-opening or expected-output premise is added. -/
theorem preparedWitnessBlockPartials_honestResponse
    (assignments : Fin SourceCount → Phi81Relation.Assignment selectedShape)
    (challenges : Fin SourceCount → RingF)
    (valid : ∀ source, Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (challenges source))
    (block : Fin (Phi81ColumnLayout.blockCount selectedShape.carrierWidth)) :
    ((preparedWitnessBlockPartials challenges (prepareWitnessActions challenges)
      (fun source => MaterializedRingF.ofRing
        (CarrierAction.assignmentBlock (assignments source) block))).map
      MaterializedRingF.toRing).getLast? =
      some (CarrierAction.assignmentBlock
        (PiRLC.PaperCompleteness.honestResponse context assignments
          ⟨challenges, valid⟩).assignment block) := by
  rw [honestResponse_assignment]
  exact preparedWitnessBlockPartials_getLast? challenges assignments block

end NightstreamFPrime.Export.Stage1.PiRLCWitnessHonestResponse
