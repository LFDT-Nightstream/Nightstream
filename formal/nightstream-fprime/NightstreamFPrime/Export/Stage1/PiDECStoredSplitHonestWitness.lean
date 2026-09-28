import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier

/-!
Connect stored PiDEC digits to the selected honest witness and message
specification. The block theorem supports the streamed replay. Commitment
and evaluation execution remain separate obligations for those same digits.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECStoredSplitHonestWitness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
  (StoredAssignment view)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Lifecycle

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev Claim := CE.Instance (PaperAlgebra.Structure selectedShape.logicalWidth)
  (Phi81Relation.PublicInput selectedShape) PaperAlgebra.Point
  PaperAlgebra.Evaluation PaperAlgebra.Commitment

/-- All sixteen successful stored outputs are the exact private witnesses
used by the selected honest messages and children. The only premise is the
successful checked split of the supplied complete parent assignment. -/
theorem splitChecked_honestWitness
    (parent : Claim)
    (parentWitness : StoredAssignment selectedShape.carrierWidth)
    (childWitnesses : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (success : StoredSplit.splitChecked parentWitness = some childWitnesses) :
    let algebra := PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey
    let semantics := PaperAlgebra.semantics Poseidon2HashChainV1Setup.productionAjtaiKey
    ∀ child,
      view (childWitnesses.get child) = algebra.splitAssignment (view parentWitness) child ∧
      PiDEC.PaperVerifier.honestMessages algebra parent (view parentWitness) child =
        ({ commitment := semantics.commit (view (childWitnesses.get child))
           evaluations := semantics.evaluations parent.constraintSystem
             (view (childWitnesses.get child)) parent.point } :
          PiDEC.PaperVerifier.ChildMessage PaperAlgebra.Evaluation PaperAlgebra.Commitment) ∧
      PiDEC.childrenOf algebra parent (view parentWitness) child =
        ({ constraintSystem := parent.constraintSystem
           commitment := semantics.commit (view (childWitnesses.get child))
           publicInput := semantics.projectPublicInput (view (childWitnesses.get child))
           point := parent.point
           evaluations := semantics.evaluations parent.constraintSystem
             (view (childWitnesses.get child)) parent.point
           stage := .fresh } : Claim) := by
  dsimp only
  intro child
  have witnessExact : view (childWitnesses.get child) =
      (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey).splitAssignment
        (view parentWitness) child :=
    StoredSplit.splitChecked_assignment parentWitness childWitnesses success child
  refine ⟨witnessExact, ?_, ?_⟩
  · simp only [PiDEC.PaperVerifier.honestMessages, ← witnessExact]
  · simp only [PiDEC.childrenOf, ← witnessExact]
    rfl

/-- The same witness identity for the actual 54-coefficient block call.
Applying this at every complete-carrier block identifies all sixteen private
witnesses, without constructing a full-carrier vector in the executable. -/
theorem splitChecked_block_honestWitness
    (parentWitness : Phi81Relation.Assignment selectedShape)
    (block : Fin (Phi81ColumnLayout.blockCount selectedShape.carrierWidth))
    (childBlocks : Vector (StoredAssignment ringDegree) productionGlobalParams.k)
    (success : StoredSplit.splitChecked
        (Vector.ofFn (CarrierAction.assignmentBlock parentWitness block)) =
      some childBlocks) :
    ∀ child, view (childBlocks.get child) =
      CarrierAction.assignmentBlock
        ((PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey).splitAssignment
          parentWitness child) block := by
  intro child
  funext lane
  change (childBlocks.get child).get lane =
    Radix.splitScalar (parentWitness (CarrierAction.carrierColumn block lane)) child
  have value := StoredSplit.splitChecked_value
    (Vector.ofFn (CarrierAction.assignmentBlock parentWitness block))
    childBlocks success child lane
  change (childBlocks.get child).get lane =
    Radix.splitScalar
      ((Vector.ofFn (CarrierAction.assignmentBlock parentWitness block))[lane.val]) child at value
  rw [Vector.getElem_ofFn] at value
  exact value

end NightstreamFPrime.Export.Stage1.PiDECStoredSplitHonestWitness
