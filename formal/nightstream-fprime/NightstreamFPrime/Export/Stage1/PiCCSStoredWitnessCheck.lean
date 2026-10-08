import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
Owns the selected PiCCS source statement for the actual application matrix
source and the frozen Ajtai setup. `statement` reads the existing typed
fresh/running public fields and equals the statement that `ProductionKey.key`
selects.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open NightstreamFPrime.Lifecycle
open CheckedWitnessExtraction

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits

/-- The sole selected commitment map, including the actual indexed key expansion. -/
def commit : Phi81Relation.Assignment carrier → PaperAlgebra.Commitment :=
  (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey).commit

/-- Computable projection of the selected key's statement. Matrix entries
remain behind the existing selected relation's access function. -/
def statement (input : PiCCSInputCheck.Input) :
    Statement K PaperAlgebra.Commitment (Phi81Relation.PublicInput carrier)
      productionShape carrier.carrierWidth
      (Phi81ColumnLayout.blockCount carrier.carrierWidth) baseOps where
  cubeLayout := (Lifecycle.PiRLC.v1_2.InputBinding.relationSource PiDECInputCheck.relation).cubeLayout
  matrixSource := (Lifecycle.PiRLC.v1_2.InputBinding.relationSource PiDECInputCheck.relation).matrixSource
  commitments := PiCCSInputCheck.outputCommitments input
  publicInputs := PiCCSInputCheck.outputPublicInputs input
  priorPoint := (PiCCSInputCheck.running input).point
  claimedPadCoefficient := (PiCCSInputCheck.verifierInput input).claimedPadCoefficient
  claimedMatrixCoefficient := (PiCCSInputCheck.verifierInput input).claimedMatrixCoefficient

/-- The executable statement is the literal selected NIFS statement. -/
theorem statement_eq_key (input : PiCCSInputCheck.Input) :
    statement input =
      (ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) := by
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
