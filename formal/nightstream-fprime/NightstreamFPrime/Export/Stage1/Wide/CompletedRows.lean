import NightstreamFPrime.Export.Stage1.Wide.PrefixCompletedAssignment
import NightstreamFPrime.Export.Stage1.Wide.PiDECCompletedAssignment
import NightstreamFPrime.Export.Stage1.Wide.RunningCompletedAssignment
import NightstreamFPrime.Export.Stage1.Wide.ApplicationCompletedAssignment
import NightstreamFPrime.Export.Stage1.Wide.FinalBindings
import NightstreamFPrime.Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition

/-! All candidate rows accept one constructed assignment from the accepted
wide physical prefix and the application step. No old sampler is executed.
The semantic-step completion and carrier norm are separate obligations. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CompletedRows

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Poseidon2HashChainV1Package (application fits)

variable {width : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

/-- The complete candidate accepts the direct sampler/product witness and
the three-permutation application witness built from the same physical source. -/
theorem rowsZero (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : ProductionKey.LogicalRelation width publicFits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := publicFits))
    (template : Proof 9) (env : Env) (message : Fin 4 → F)
    (pilotAssumptions : Lifecycle.Pilot.Assumptions Layout.PilotProduction.interface Layout.PilotProduction.witnessOffset env)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width publicFits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := publicFits))
      Layout.Stage1.Wide.PiRLCInputs.phaseOffset env)
    (dAssumptions : PiDEC.v1_1.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiDECInputs.interface width publicFits) Layout.Stage1.Wide.PiDECInputs.phaseOffset env)
    (physical : Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation env)
    (next : Lifecycle.Stage1.NextPreimage.SpecHolds Layout.Stage1.NextPreimageInputs.sourceInterface
      Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset env)
    (step : (List.ofFn fun lane : Fin 4 => env (Layout.Stage1.ApplicationInputs.outputSourceColumn lane)) =
      application.step (List.ofFn fun lane : Fin 4 => env (Layout.Stage1.ApplicationInputs.inputSourceColumn lane))
        (List.ofFn message)) :
    (Stage1Plan.plan application compiled relation fits.package).RowsZero
      (SourceAssignment.assignment application env (ApplicationCompletedAssignment.suffix env message)) := by
  let suffix := ApplicationCompletedAssignment.suffix env message
  obtain ⟨throughD, running⟩ :=
    (Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation env).mp physical
  obtain ⟨throughR, dRows⟩ := (Layout.Stage1.Wide.PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation env).mp throughD
  obtain ⟨throughC, rRows⟩ := (Layout.Stage1.Wide.PilotPiCCSPiRLC.physicalHolds_iff relation env).mp throughR
  have cRows := ((Layout.Stage1.PilotPiCCS.physicalHolds_iff relation env).mp throughC).2
  apply (Stage1Plan.rows_iff application compiled relation fits.package _).mpr
  exact ⟨PrefixCompletedAssignment.rowsZero application env suffix relation ajtai template
      pilotAssumptions cAssumptions throughC,
    SourceAssignment.piRlc_complete application compiled env suffix,
    PiDECCompletedAssignment.rowsZero application env suffix relation ajtai template
      cAssumptions cRows rAssumptions rRows dAssumptions dRows,
    RunningCompletedAssignment.rowsZero application env suffix relation ajtai template cAssumptions cRows running,
    ApplicationCompletedAssignment.rowsZero env message step,
    FinalBindings.nextPreimage application env suffix next,
    FinalBindings.publicOutput application env suffix⟩

end NightstreamFPrime.Export.Stage1.Wide.CompletedRows
