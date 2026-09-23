import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

/-! Production PiDEC start values. Allocation consumers use PiDECStarts; this
module checks the unchanged default profile values. -/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

theorem phaseStarts_eq :
    [phaseLogicalStart, phaseRowStart, phaseFreshStart] =
      [28470790, 28295335, 28471060] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [28470790, 28470790, 28471060, 28471060, 28471060, 28471060] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [28295335, 28295335, 28318015, 28319203, 28319311, 28320823] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [28471060, 28471060, 28488880, 28488880, 28488880, 28488880] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 28470790 + source ∧
      scalarRowStart source = 28295335 + source * 84 ∧
      scalarFreshStart source = 28471060 + source * 66 := by
  refine ⟨?_, rfl, rfl⟩
  change 28470790 + source * 1 = 28470790 + source
  rw [Nat.mul_one]

theorem finalBoundaries_eq :
    outputRowStart = 28320823 ∧ outputFreshStart = 28488880 := by
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
        [28295335, 28318015, 28319203, 28319311, 28320823, 28320823,
          28666318] ∧
      cumulativePhysicalColumns relation =
        [28470790, 28488880, 28488880, 28488880, 28488880, 28488880,
          28785018] ∧
      cumulativeJointDomains relation =
        [28470790, 28488880, 28488880, 28488880, 28488880, 28488880,
          28785018] := by
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
