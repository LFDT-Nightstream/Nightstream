import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds

/-!
Owns the zero-row package-to-accumulator soundness edge.

The canonical package already contains every PiCCS, PiRLC, and PiDEC row.
This module composes their existing package theorems into the exact SuperNeo
accumulator result. It adds no package component, column, row, or digest.
-/

namespace NightstreamFPrime.Export.Stage1.AccumulatorPackage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Satisfaction of the canonical package rows implies the exact deterministic
SuperNeo accumulator update. PiCCS, PiRLC, and PiDEC remain status open until
the final package identity and fixed-point gates are rerun. -/
theorem circuitPackage_implies_accumulatorHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (vk : KeyDigest) (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (piRlcAssumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) :
    NightstreamFPrime.Lifecycle.Stage1.Accumulator.Holds relation ajtai vk
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.running
        Data.logicalWidth Data.publicFits
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.fresh
        Data.logicalWidth Data.publicFits
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.proof relation
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.output relation
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) := by
  have piCcsPhase := Package.circuitPackage_implies_piCcsPhaseHolds relation
    ajtai
    (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.proof relation
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)) env holds
  have piRlcPhase := Package.circuitPackage_implies_piRlcPhaseHolds relation
    ajtai env holds piRlcAssumptions
  have piDecPhase := Package.circuitPackage_implies_piDecPhaseHolds relation
    ajtai env holds
      (NightstreamFPrime.Layout.Stage1.PiDECInputs.assumptions relation
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
  apply NightstreamFPrime.Layout.Stage1.AccumulatorSemantics.phases_imply_holds
    relation ajtai vk (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  · simpa [PiCCSInvocations.parentInterface,
      NightstreamFPrime.Layout.Stage1.AccumulatorInputs.piCcsInterface] using
      piCcsPhase
  · exact piRlcPhase
  · simpa [PiDECArithmetic.phaseInterface] using piDecPhase

end NightstreamFPrime.Export.Stage1.AccumulatorPackage
