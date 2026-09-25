import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-!
Owns structural projections from the canonical PiRLC combination schedule.
This module normalizes proof-carrying product indices without evaluating a
production invocation or row list.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationProjection

open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-- A combination index with the canonical numeric value has the exact
block-major, lane-major, cell-major coordinate. This statement is independent
of the proof term carried by `Fin`. -/
theorem coordinates_eq_of_val {blockCount cellCount : Nat}
    (index : Fin (CombinationStep.privateCount blockCount cellCount))
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (value : index.val =
      logicalIndex cellCount block.val lane.val cell.val) :
    CombinationStep.coordinates index =
      (block, lane, cell) := by
  have indexEq : index = CombinationStep.indexOf block lane cell := by
    apply Fin.ext
    rw [value, indexOf_val]
  rw [indexEq]
  simp [CombinationStep.coordinates, CombinationStep.indexOf]

end NightstreamFPrime.Export.Stage1.PiRLCCombinationProjection
