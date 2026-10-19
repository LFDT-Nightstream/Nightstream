import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

/-! Production PiDEC start values. Allocation consumers use PiDECStarts; this
module checks the current production profile values. -/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

theorem phaseStarts_eq :
    [phaseLogicalStart, phaseRowStart, phaseFreshStart] =
      [11464602, 11343524, 11464872] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [11464602, 11464602, 11464872, 11464872, 11464872, 11464872] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [11343524, 11343524, 11348384, 11349572, 11349680, 11350112] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [11464872, 11464872, 11464872, 11464872, 11464872, 11464872] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 11464602 + source ∧
      scalarRowStart source = 11343524 + source * 18 ∧
      scalarFreshStart source = 11464872 + source * 0 := by
  refine ⟨?_, rfl, rfl⟩
  change 11464602 + source * 1 = 11464602 + source
  rw [Nat.mul_one]

theorem finalBoundaries_eq :
    outputRowStart = 11350112 ∧ outputFreshStart = 11464872 := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Layout.Stage1.PiDECStarts

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    cumulativePhysicalRows relation =
        [11343524, 11348384, 11349572, 11349680, 11350112, 11350112,
          11377912] ∧
      cumulativePhysicalColumns relation =
        [11464602, 11464872, 11464872, 11464872, 11464872, 11464872,
          11464874] ∧
      cumulativeJointDomains relation =
        [11464602, 11464872, 11464872, 11464872, 11464872, 11464872,
          11464874] := by
  rcases PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq relation with ⟨rows, columns⟩
  have joint : PilotPiCCSPiRLCPiDEC.cumulativeJointDomains relation =
      List.zipWith max
        ((PiDEC.v1_2.cumulativeFrom 0 PiDEC.v1_2.exactRowDeltas).map
          (PiDECStarts.phaseRowStart + ·))
        ((PiDEC.v1_2.cumulativeFrom 0 PiDEC.v1_2.exactPhysicalColumnDeltas).map
          (PilotPiCCSPiRLCPiDEC.piDecOffset + ·)) := by
    rw [PilotPiCCSPiRLCPiDEC.cumulativeJointDomains, rows, columns]
  refine ⟨?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, rows, physicalRowCount_eq relation]
    rfl
  · rw [cumulativePhysicalColumns, columns, physicalColumnCount_eq relation]
    rfl
  · rw [cumulativeJointDomains, joint, jointDomain_eq relation]
    decide

end NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition
