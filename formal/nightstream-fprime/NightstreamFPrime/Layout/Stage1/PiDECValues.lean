import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

/-! Production PiDEC start values. Allocation consumers use PiDECStarts; this
module checks the current production profile values. -/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

theorem phaseStarts_eq :
    [phaseLogicalStart, phaseRowStart, phaseFreshStart] =
      [27409630, 27209746, 27409900] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [27409630, 27409630, 27409900, 27409900, 27409900, 27409900] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [27209746, 27209746, 27232426, 27233614, 27233722, 27235234] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [27409900, 27409900, 27427720, 27427720, 27427720, 27427720] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 27409630 + source ∧
      scalarRowStart source = 27209746 + source * 84 ∧
      scalarFreshStart source = 27409900 + source * 66 := by
  refine ⟨?_, rfl, rfl⟩
  change 27409630 + source * 1 = 27409630 + source
  rw [Nat.mul_one]

theorem finalBoundaries_eq :
    outputRowStart = 27235234 ∧ outputFreshStart = 27427720 := by
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
        [27209746, 27232426, 27233614, 27233722, 27235234, 27235234,
          27284593] ∧
      cumulativePhysicalColumns relation =
        [27409630, 27427720, 27427720, 27427720, 27427720, 27427720,
          27427722] ∧
      cumulativeJointDomains relation =
        [27409630, 27427720, 27427720, 27427720, 27427720, 27427720,
          27427722] := by
  rcases PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq relation with ⟨rows, columns⟩
  have joint : PilotPiCCSPiRLCPiDEC.cumulativeJointDomains relation =
      List.zipWith max
        ((PiDEC.v1_1.cumulativeFrom 0 PiDEC.v1_1.exactRowDeltas).map
          (PiDECStarts.phaseRowStart + ·))
        ((PiDEC.v1_1.cumulativeFrom 0 PiDEC.v1_1.exactPhysicalColumnDeltas).map
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
