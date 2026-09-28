import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

/-! Production PiDEC start values. Allocation consumers use PiDECStarts; this
module checks the current production profile values. -/

namespace NightstreamFPrime.Layout.Stage1.PiDECStarts

theorem phaseStarts_eq :
    [phaseLogicalStart, phaseRowStart, phaseFreshStart] =
      [28097016, 27897132, 28097286] := by
  rfl

theorem childLogicalStarts_eq :
    [inputLogicalStart, publicInputLogicalStart, commitmentLogicalStart,
      evalKLogicalStart, evalALogicalStart, outputLogicalStart] =
    [28097016, 28097016, 28097286, 28097286, 28097286, 28097286] := by
  rfl

theorem childRowStarts_eq :
    [inputRowStart, publicInputRowStart, commitmentRowStart, evalKRowStart,
      evalARowStart, outputRowStart] =
    [27897132, 27897132, 27919812, 27921000, 27921108, 27922620] := by
  rfl

theorem childFreshStarts_eq :
    [inputFreshStart, publicInputFreshStart, commitmentFreshStart,
      evalKFreshStart, evalAFreshStart, outputFreshStart] =
    [28097286, 28097286, 28115106, 28115106, 28115106, 28115106] := by
  rfl

theorem scalarStarts_eq (source : Nat) :
    scalarLogicalStart source = 28097016 + source ∧
      scalarRowStart source = 27897132 + source * 84 ∧
      scalarFreshStart source = 28097286 + source * 66 := by
  refine ⟨?_, rfl, rfl⟩
  change 28097016 + source * 1 = 28097016 + source
  rw [Nat.mul_one]

theorem finalBoundaries_eq :
    outputRowStart = 27922620 ∧ outputFreshStart = 28115106 := by
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
        [27897132, 27919812, 27921000, 27921108, 27922620, 27922620,
          28268115] ∧
      cumulativePhysicalColumns relation =
        [28097016, 28115106, 28115106, 28115106, 28115106, 28115106,
          28411244] ∧
      cumulativeJointDomains relation =
        [28097016, 28115106, 28115106, 28115106, 28115106, 28115106,
          28411244] := by
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
