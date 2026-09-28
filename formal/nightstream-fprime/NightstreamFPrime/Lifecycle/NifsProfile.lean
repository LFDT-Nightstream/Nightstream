import NightstreamFPrime.Lifecycle.ProductionKey

/-!
SuperNeo v1.1 Definitions 18–21 and Sections 7.3–7.5.
The selected key fixes the relation, dimensions, and encoding across all
three phases. These are structural facts; they do not assume Ajtai hardness
or identify a caller-supplied key with the external verifier's selected key.
-/

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

/-- Recovering the logical structure from the selected NIFS key returns the
original matrices and fixed polynomial, including the logical carrier map. -/
theorem selected_relation :
    canonicalStructure (publicFits := publicFits)
      (ProductionKey.key relation ajtai).relationSource = relation.system := by
  exact canonicalStructure_relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system

/-- The selected matrix entries are preserved individually, rather than
being identified by a size or a digest. -/
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

/-- Exact production dimensions and the whole-ring public/carrier widths. -/
theorem selected_shape :
    productionShape.cubeVariables = 28 ∧
    productionShape.matrixCount = 14 ∧
    productionShape.coefficientCount = 54 ∧
    productionShape.freshCount = 1 ∧
    productionShape.runningCount = 16 ∧
    productionShape.sourceCount = 17 ∧
    (FullShape logicalWidth publicFits).publicWidth = 270 ∧
    (FullShape logicalWidth publicFits).carrierWidth =
      Phi81CarrierLayout.carrierWidth logicalWidth ∧
    productionProfile.commitmentWidth = 22 ∧
    ProductionKey.degreeBound relation = 9 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Counts in the actual verifier key match the paper profile, including the
PiRLC index type and the PiDEC output index type. -/
theorem selected_arity :
    (ProductionKey.key relation ajtai).arity.freshCount = 1 ∧
    (ProductionKey.key relation ajtai).arity.mode.count
      (ProductionKey.key relation ajtai).params = 16 ∧
    (ProductionKey.key relation ajtai).arity.total = 17 ∧
    (ProductionKey.key relation ajtai).params.k = 16 := by
  exact ⟨Nifs.PaperProfile.arity_freshCount,
    Nifs.PaperProfile.arity_runningCount, Nifs.PaperProfile.arity_total,
    Nifs.PaperProfile.outputCount⟩

/-- PiCCS and the two tail phases use the same commitment map and public
projection on each complete assignment. Both tail algebras are built from
this exact Ajtai key and this same semantic relation. -/
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

/-- Every phase keeps the selected logical relation, for arbitrary proof
messages and a challenge batch. No source or child can select new matrices. -/
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
