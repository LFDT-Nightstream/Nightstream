import NightstreamFPrime.Lifecycle.ProductionKey

/-! SuperNeo v1.2 Definitions 19–22 and Sections 7.3–7.5. -/

namespace NightstreamFPrime.Lifecycle.NifsProfile

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem selected_relation :
    canonicalStructure (publicFits := publicFits)
      (ProductionKey.key relation ajtai).relationSource = relation.system := by
  exact canonicalStructure_relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system

theorem selected_matrix
    (matrix : Fin productionProfile.ccsMatrices)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin logicalWidth) :
    (ProductionKey.key relation ajtai).matrixSource.matrices matrix vertex
        (Phi81CarrierLayout.embedLogical column) =
      relation.matrices matrix vertex column := by
  exact Phi81MatrixSource.source_matrix_embedLogical
    cubeVariables productionProfile.freshSources productionProfile.runningSources
    productionProfile.ccsMatrices logicalWidth relation.matrices
    ProductionRelation.polynomial matrix vertex column

theorem selected_shape :
    productionShape.cubeVariables = 28 ∧
    productionShape.matrixCount = 4 ∧
    productionShape.coefficientCount = 54 ∧
    productionShape.freshCount = 1 ∧
    productionShape.runningCount = 16 ∧
    productionShape.sourceCount = 17 ∧
    (FullShape logicalWidth publicFits).publicWidth = 270 ∧
    (FullShape logicalWidth publicFits).carrierWidth =
      Phi81CarrierLayout.carrierWidth logicalWidth ∧
    productionProfile.commitmentWidth = 22 ∧
    ProductionKey.degreeBound relation = 8 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem selected_arity :
    (ProductionKey.key relation ajtai).arity.freshCount = 1 ∧
    (ProductionKey.key relation ajtai).arity.mode.count
      (ProductionKey.key relation ajtai).params = 16 ∧
    (ProductionKey.key relation ajtai).arity.total = 17 ∧
    (ProductionKey.key relation ajtai).params.k = 16 := by
  exact ⟨Nifs.PaperProfile.arity_freshCount,
    Nifs.PaperProfile.arity_runningCount, Nifs.PaperProfile.arity_total,
    Nifs.PaperProfile.outputCount⟩

theorem shared_setup
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.key relation ajtai).openingMaps.commit assignment =
      (ProductionKey.key relation ajtai).piRlcSemantics.commit assignment ∧
    (ProductionKey.key relation ajtai).openingMaps.projectPublicInput assignment =
      (ProductionKey.key relation ajtai).piRlcSemantics.projectPublicInput assignment ∧
    (ProductionKey.key relation ajtai).piRlcSemantics = semantics ajtai ∧
    (ProductionKey.key relation ajtai).piRlcAlgebra = piRlcAlgebra ajtai ∧
    (ProductionKey.key relation ajtai).piDecAlgebra = piDecAlgebra ajtai := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem phases_preserve_relation
    (running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape)
    (proof : Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
      (ProductionKey.degreeBound relation))
    (challenges : Fin Nifs.PaperProfile.arity.total → RingF) :
    (∀ source, canonicalStructure (publicFits := publicFits)
      ((ProductionKey.key relation ajtai).piCcsOutputs running fresh proof source
        ).constraintSystem = relation.system) ∧
    canonicalStructure (publicFits := publicFits)
      ((ProductionKey.key relation ajtai).parentForChallenges
        running fresh proof challenges).constraintSystem = relation.system ∧
    (∀ child, canonicalStructure (publicFits := publicFits)
      (PiDEC.PaperVerifier.children
        (ProductionKey.key relation ajtai).piDecPublicInputSplit
        ((ProductionKey.key relation ajtai).piDecAttemptForParent proof
          ((ProductionKey.key relation ajtai).parentForChallenges
            running fresh proof challenges)) child).constraintSystem = relation.system) := by
  refine ⟨fun _ => ?_, ?_, fun _ => ?_⟩
  all_goals exact selected_relation relation ajtai

end NightstreamFPrime.Lifecycle.NifsProfile
