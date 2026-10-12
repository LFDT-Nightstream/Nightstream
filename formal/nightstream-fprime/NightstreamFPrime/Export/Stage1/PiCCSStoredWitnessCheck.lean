import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
Owns the PiCCS source statement of one application: its matrix source and the
Ajtai key of its commitment setup. `statement` reads the existing typed
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

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application)

/-- The commitment map of the application's setup, including the indexed key
expansion. -/
def commit : Phi81Relation.Assignment (carrier application) → PaperAlgebra.Commitment :=
  (PaperAlgebra.openingMaps (PerApplicationCanonicalPackage.commitmentKey setup)).commit

/-- Computable projection of the application key's statement. Matrix entries
remain behind the application relation's access function. -/
def statement (input : PiCCSInputCheck.Input) :
    Statement K PaperAlgebra.Commitment (Phi81Relation.PublicInput (carrier application))
      productionShape (carrier application).carrierWidth
      (Phi81ColumnLayout.blockCount (carrier application).carrierWidth) baseOps where
  cubeLayout := (Lifecycle.PiRLC.v1_2.InputBinding.relationSource
    (PerApplicationFixedPoint.relation application fits)).cubeLayout
  matrixSource := (Lifecycle.PiRLC.v1_2.InputBinding.relationSource
    (PerApplicationFixedPoint.relation application fits)).matrixSource
  commitments := PiCCSInputCheck.outputCommitments
    (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application) input
  publicInputs := PiCCSInputCheck.outputPublicInputs input
  priorPoint := (PiCCSInputCheck.running
    (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application) input).point
  claimedPadCoefficient := (PiCCSInputCheck.verifierInput input).claimedPadCoefficient
  claimedMatrixCoefficient := (PiCCSInputCheck.verifierInput input).claimedMatrixCoefficient

/-- The executable statement is the literal application NIFS statement. -/
theorem statement_eq_key (input : PiCCSInputCheck.Input) :
    statement application fits input =
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey setup)).statement
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) := by
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
