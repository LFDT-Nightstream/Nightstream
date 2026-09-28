import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import NightstreamFPrime.Export.Stage1.PiDECStoredSplitHonestWitness

/-! The summed block contributions equal the selected honest child commitments. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCommitmentHonestMessages

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
  (StoredAssignment view)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev Claim := CE.Instance (PaperAlgebra.Structure selectedShape.logicalWidth)
  (Phi81Relation.PublicInput selectedShape) PaperAlgebra.Point
  PaperAlgebra.Evaluation PaperAlgebra.Commitment

/-- For the same successful stored split, sum every computed key-block
contribution for each child and row. The result is the existing honest
message commitment for the production key. No expected commitment is an
input, and no parent-opening or runtime premise is used. -/
theorem sum_contributions_honestMessages
    (parent : Claim)
    (parentWitness : StoredAssignment selectedShape.carrierWidth)
    (childWitnesses : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (success : StoredSplit.splitChecked parentWitness = some childWitnesses)
    (child : Fin productionGlobalParams.k)
    (row : Fin Poseidon2HashChainV1Setup.verifierRows) :
    (PiDECCommitmentFold.sum fun block =>
      (PiDECCommitmentBlock.contributions Poseidon2HashChainV1Setup.productionSetup
        row block (PiDECCommitmentFold.childBlocks
          (shape := selectedShape) childWitnesses block)).get child).get =
      (PiDEC.PaperVerifier.honestMessages
        (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
        parent (view parentWitness) child).commitment row := by
  have honest := (PiDECStoredSplitHonestWitness.splitChecked_honestWitness
    parent parentWitness childWitnesses success child).2.1
  have messageRow := congrArg
    (fun message : PiDEC.PaperVerifier.ChildMessage
        PaperAlgebra.Evaluation PaperAlgebra.Commitment => message.commitment row) honest
  dsimp only at messageRow
  calc
    _ = (PaperAlgebra.semantics Poseidon2HashChainV1Setup.productionAjtaiKey).commit
          (view (childWitnesses.get child)) row := by
      simp only [PaperAlgebra.semantics,
        Phi81Relation.PiRLCAlgebra.Commitment.commit,
        Poseidon2HashChainV1Setup.productionAjtaiKey,
        Poseidon2HashChainV1Setup.ajtaiKey,
        PerApplicationCanonicalPackage.commitmentKey]
      exact PiDECCommitmentFold.sum_contributions_eq_ajtaiRow
        (shape := selectedShape) Poseidon2HashChainV1Setup.productionSetup
        childWitnesses row child
    _ = _ := messageRow.symm

end NightstreamFPrime.Export.Stage1.PiDECCommitmentHonestMessages
