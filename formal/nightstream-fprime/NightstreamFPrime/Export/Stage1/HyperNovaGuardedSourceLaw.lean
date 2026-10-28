import NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw
import NightstreamFPrime.Export.Stage1.HyperNovaRealInput

/-!
Owns the decoded NIFS data of one visit of the history law: the exact source
input of a recursive payload (`inputs`) and the real NIFS output on the good
active branch (`realOutput`).
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaGuardedSourceLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open HyperNovaHistory (Statement Payload SourceResult)
open HyperNovaVisitedLaw (Visit goodActive)
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

attribute [local instance] Classical.propDecidable

private def inactiveInput : PiCCSInputCheck.Input where
  commitment := Vector.replicate _ 0
  publicInput := Vector.replicate _ 0
  rounds := Vector.replicate _ (Vector.replicate _ K.zero)
  evalK := Vector.replicate _ (Vector.replicate _ K.zero)
  evalA := Vector.replicate _ (Vector.replicate _ (Vector.replicate _ K.zero))
  running := {
    point := Vector.replicate _ K.zero
    commitments := Vector.replicate _ (Vector.replicate _ 0)
    publicInputs := Vector.replicate _ (Vector.replicate _ 0)
    evalK := Vector.replicate _ (Vector.replicate _ K.zero)
    evalA := Vector.replicate _ (Vector.replicate _ (Vector.replicate _ K.zero)) }

def inputs (visit : Visit application) : PiCCSInputCheck.Input :=
  match visit.1 with
  | some (_, .recursive payload) => HyperNovaHistory.sourceInput application fits payload
  | _ => inactiveInput

noncomputable def realOutput (visit : Visit application) :
    Option (NifsRealSuccess.RealOutput (PerApplicationFixedPoint.relation application fits)) :=
  if (goodActive application fits setup) visit then
    match visit.1 with
    | some (_, .recursive payload) => some (HyperNovaRealInput.output application fits setup payload)
    | _ => none
  else none

theorem realOutput_off (visit : Visit application) (inactive : ¬ goodActive application fits setup visit) :
    realOutput application fits setup visit = none := by
  simp only [realOutput, if_neg inactive]

end NightstreamFPrime.Export.Stage1.HyperNovaGuardedSourceLaw
