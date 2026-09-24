import NightstreamFPrime.Export.Stage1.Wide.Stage1Plan
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! Counts of the assembled, unselected wide-sampler candidate. These do not
certify whole-package preservation or the matrix interpreter. -/

namespace NightstreamFPrime.Export.Stage1.Wide.HashChainCounts

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open Spec.Folding.PiCCS.PaperJoint
open Poseidon2HashChainV1Package

theorem applicationCoordinates : RetainedLayout.applicationCount application = 10742 := by
  rw [RetainedLayout.applicationCount, ApplicationSelectedBlocks.retainedCoordinateCount_eq,
    retainedApplicationWordCount]

theorem logicalCoordinates : RetainedLayout.logicalWidth application = 137341846 := by
  rw [RetainedLayout.logicalWidth_eq, applicationCoordinates]

theorem committedCoordinates :
    Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth application) = 137341872 := by
  rw [logicalCoordinates]
  rfl

theorem logicalRows (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    (Stage1Plan.plan application compiled (PerApplicationFixedPoint.seedRelation application fits)
      fits.package).rowCount = 3248956 := by
  rw [Stage1Plan.plan_rows, selectedApplicationRowCount]

end NightstreamFPrime.Export.Stage1.Wide.HashChainCounts
