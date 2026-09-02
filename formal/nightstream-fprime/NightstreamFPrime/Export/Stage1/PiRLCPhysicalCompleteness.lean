import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Layout.PiRLC.v1_1.Preservation

/-!
Owns the constructive bridge from the semantic PiRLC phase to its exact
physical rows in the final Stage 1 Spartan column order.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCPhysicalCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- A valid production PiRLC phase determines the exact remapped physical
rows. The completion changes only the PiRLC-local private interval. -/
theorem completePhysicalRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (phase :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
            NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
          8908425 ∧
        R1CS.RowsHold completed
          (NightstreamFPrime.Layout.Stage1.Spartan.remapRows
            (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
              (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
                (logicalWidth := Data.logicalWidth)
                (publicFits := Data.publicFits))
              NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)) := by
  rcases NightstreamFPrime.Layout.PiRLC.v1_1.physical_complete_production
      relation ajtai
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.inputShapes relation)
      assumptions phase with
    ⟨source, sourceAgrees, sourceRows⟩
  let completed :=
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval env source
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset 8908425
  refine ⟨completed,
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval_agreesOutside
      env source NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        8908425,
    ?_⟩
  apply NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold_copyMappedInterval
  · norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
  · norm_num [NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan,
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.Spartan.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
  · exact sourceAgrees
  · exact sourceRows

end NightstreamFPrime.Export.Stage1.PiRLCPhysicalCompleteness
