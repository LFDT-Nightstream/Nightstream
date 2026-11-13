import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Layout.PiRLC.v1_2.Preservation

namespace NightstreamFPrime.Export.Stage1.PiRLCPhysicalCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem completePhysicalRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.Assumptions relation
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.PackageOrder.pullback env))
    (phase :
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Semantics.PhaseHolds relation ajtai
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.PackageOrder.pullback env)) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.PackageOrder.sourceToPackage
            NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)
          4786265 ∧
        R1CS.RowsHold completed
          (NightstreamFPrime.Layout.Stage1.PackageOrder.remapRows
            (NightstreamFPrime.Layout.PiRLC.v1_2.physicalRows relation
              (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
                (logicalWidth := Data.logicalWidth)
                (publicFits := Data.publicFits))
              NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset)) := by
  rcases NightstreamFPrime.Layout.PiRLC.v1_2.physical_complete_production
      relation ajtai
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PackageOrder.pullback env)
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.inputShapes relation)
      assumptions phase with
    ⟨source, sourceAgrees, sourceRows⟩
  let completed :=
    NightstreamFPrime.Layout.Stage1.PackageOrder.copyMappedInterval env source
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset 4786265
  refine ⟨completed,
    NightstreamFPrime.Layout.Stage1.PackageOrder.copyMappedInterval_agreesOutside
      env source NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        4786265,
    ?_⟩
  apply NightstreamFPrime.Layout.Stage1.PackageOrder.remapRows_hold_copyMappedInterval
  · norm_num [NightstreamFPrime.Layout.Stage1.PackageOrder.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
  · rw [NightstreamFPrime.Layout.Stage1.PackageOrder.privateColumnCount_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PackageOrder.sourceToPackage,
      NightstreamFPrime.Layout.Stage1.PackageOrder.pilotSourceColumnCount,
      NightstreamFPrime.Layout.Stage1.PackageOrder.proofInputSourceStart,
      NightstreamFPrime.Layout.Stage1.PackageOrder.piCcsPhaseOffset,
      NightstreamFPrime.Layout.Stage1.PackageOrder.piCcsLocalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
  · exact sourceAgrees
  · exact sourceRows

end NightstreamFPrime.Export.Stage1.PiRLCPhysicalCompleteness
