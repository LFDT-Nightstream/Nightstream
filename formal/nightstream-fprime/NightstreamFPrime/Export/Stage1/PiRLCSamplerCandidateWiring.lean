import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRetainedGeometry

/-!
Provides shared candidate forms for digest decoding and First54. The forms
use the existing First54 reject and symbol slots and retain the existing
coordinates for all other digest-lane values.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerCandidateWiring

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryRetainedBlocks
open PiRLCSamplerOrdinaryRetainedGeometry (piRlcGeometry)

def candidate (descriptor : Lane) (part : Fin 2) :
    PiRLCFirst54DirectSchedule.Candidate where
  source := descriptor.source
  round := ⟨descriptor.round.val * 8 + descriptor.lane.val * 2 + part.val, by
    have roundLt : descriptor.round.val < 8 := descriptor.round.isLt
    have laneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
    have partLt := part.isLt
    change _ < 64
    omega⟩

def rejectPosition (part : Fin 2) : Fin logicalCountPerLane :=
  ⟨82 + part.val * 17, by have bound := part.isLt; change _ < 100; omega⟩

def symbolPosition (part : Fin 2) : Fin logicalCountPerLane :=
  ⟨67 + part.val * 17, by have bound := part.isLt; change _ < 100; omega⟩

def localForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin logicalCountPerLane) : SparseForm logicalWidth :=
  (logicalBlock program).form
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry)
    (logicalSlot descriptor position)

def logicalForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin logicalCountPerLane) : SparseForm logicalWidth :=
  let inputs := PiRLCRetainedInputs.first54Inputs (piRlcGeometry geometry)
  if position.val = 82 then inputs.reject (candidate descriptor 0)
  else if position.val = 99 then inputs.reject (candidate descriptor 1)
  else if position.val = 67 then inputs.symbol (candidate descriptor 0)
  else if position.val = 84 then inputs.symbol (candidate descriptor 1)
  else localForm geometry descriptor position

/-- Every decoder reject output is the selector's exact physical form. -/
theorem logicalForm_reject {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (part : Fin 2) :
    logicalForm geometry descriptor (rejectPosition part) =
      (PiRLCRetainedInputs.first54Inputs (piRlcGeometry geometry)).reject
        (candidate descriptor part) := by
  fin_cases part <;> simp [logicalForm, rejectPosition]

/-- Every decoder symbol output is the selector's exact physical form. -/
theorem logicalForm_symbol {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (part : Fin 2) :
    logicalForm geometry descriptor (symbolPosition part) =
      (PiRLCRetainedInputs.first54Inputs (piRlcGeometry geometry)).symbol
        (candidate descriptor part) := by
  fin_cases part <;> simp [logicalForm, symbolPosition]

private theorem candidate_coordinates (descriptor : Lane) (part : Fin 2) :
    (candidate descriptor part).round.val / 8 = descriptor.round.val ∧
    (candidate descriptor part).round.val % 8 / 2 = descriptor.lane.val ∧
    (candidate descriptor part).round.val % 2 = part.val := by
  have laneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
  have partLt := part.isLt
  simp only [candidate]
  omega

theorem reject_source (descriptor : Lane) (part : Fin 2) :
    logicalSource descriptor (rejectPosition part) =
      (candidate descriptor part).rejectColumn := by
  have coordinates := candidate_coordinates descriptor part
  simp only [PiRLCFirst54DirectSchedule.Candidate.rejectColumn,
    PiRLCFirst54Invocations.rejectSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart, coordinates.1, coordinates.2.1,
    coordinates.2.2, logicalSource, rejectPosition]
  simp only [candidate, NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.auxiliaryCount]
  omega

theorem symbol_source (descriptor : Lane) (part : Fin 2) :
    logicalSource descriptor (symbolPosition part) =
      (candidate descriptor part).symbolColumn := by
  have coordinates := candidate_coordinates descriptor part
  simp only [PiRLCFirst54DirectSchedule.Candidate.symbolColumn,
    PiRLCFirst54Invocations.remainderSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart, coordinates.1, coordinates.2.1,
    coordinates.2.2, logicalSource, symbolPosition]
  simp only [candidate, NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount,
    NightstreamFPrime.Gadgets.Sampling.Candidate16Five.auxiliaryCount]
  omega

private theorem reject_block_source (program : Lifecycle.Stage1.Application.Program)
    (descriptor : Lane) (part : Fin 2) :
    (PiRLCFirst54RetainedBlocks.rejectBlock program).source
        (PiRLCFirst54DirectSchedule.candidateIndex (candidate descriptor part)) =
      (logicalBlock program).source (logicalSlot descriptor (rejectPosition part)) := by
  rw [PiRLCFirst54RetainedBlocks.rejectBlock_source,
    PiRLCFirst54DirectSchedule.candidate_candidateIndex, logicalBlock_source]
  apply Fin.ext
  simp only [PiRLCFirst54DirectPlan.retainedRejectColumn,
    PiRLCFirst54DirectPlan.packageColumn, PiRLCFirst54DirectPlan.prefixColumn,
    FieldSuffixBlock.baseColumn, PiRLCProductPlan.baseColumn,
    ProductRetainedBlock.baseColumn, RunningTransitionRetainedBlocks.packageSourceColumn,
    PiRLCRetainedPreservation.baseSourceColumn, PiRLCProductPlan.mappedPackageColumn,
    PiRLCProductPlan.shiftedPackageColumn, reject_source]

private theorem symbol_block_source (program : Lifecycle.Stage1.Application.Program)
    (descriptor : Lane) (part : Fin 2) :
    (PiRLCFirst54RetainedBlocks.symbolBlock program).source
        (PiRLCFirst54DirectSchedule.candidateIndex (candidate descriptor part)) =
      (logicalBlock program).source (logicalSlot descriptor (symbolPosition part)) := by
  rw [PiRLCFirst54RetainedBlocks.symbolBlock_source,
    PiRLCFirst54DirectSchedule.candidate_candidateIndex, logicalBlock_source]
  apply Fin.ext
  simp only [PiRLCFirst54DirectPlan.retainedSymbolColumn,
    PiRLCFirst54DirectPlan.packageColumn, PiRLCFirst54DirectPlan.prefixColumn,
    FieldSuffixBlock.baseColumn, PiRLCProductPlan.baseColumn,
    ProductRetainedBlock.baseColumn, RunningTransitionRetainedBlocks.packageSourceColumn,
    PiRLCRetainedPreservation.baseSourceColumn, PiRLCProductPlan.mappedPackageColumn,
    PiRLCProductPlan.shiftedPackageColumn, symbol_source]

/-- Canonical witness construction still supplies the same logical values.
This encoding premise is used for completeness, not accepted-row soundness. -/
theorem logicalForm_eval {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment NightstreamFPrime.Spec.F logicalWidth)
    (source : Fin (sourceWidth program) → NightstreamFPrime.Spec.F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment source)
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (logicalForm geometry descriptor position).eval assignment =
      source ((logicalBlock program).source (logicalSlot descriptor position)) := by
  have reject (part : Fin 2) :
      ((PiRLCRetainedInputs.first54Inputs (piRlcGeometry geometry)).reject
        (candidate descriptor part)).eval assignment =
      source ((logicalBlock program).source (logicalSlot descriptor (rejectPosition part))) := by
    rw [PiRLCRetainedInputs.first54Inputs, LowNormBlock.Block.form_eval _ _ _ _ _ encodes.reject,
      reject_block_source]
  have symbol (part : Fin 2) :
      ((PiRLCRetainedInputs.first54Inputs (piRlcGeometry geometry)).symbol
        (candidate descriptor part)).eval assignment =
      source ((logicalBlock program).source (logicalSlot descriptor (symbolPosition part))) := by
    rw [PiRLCRetainedInputs.first54Inputs, LowNormBlock.Block.form_eval _ _ _ _ _ encodes.symbol,
      symbol_block_source]
  unfold logicalForm
  split
  · have same : position = rejectPosition 0 := Fin.ext (by assumption)
    rw [same]
    exact reject 0
  · split
    · have same : position = rejectPosition 1 := Fin.ext (by assumption)
      rw [same]
      exact reject 1
    · split
      · have same : position = symbolPosition 0 := Fin.ext (by assumption)
        rw [same]
        exact symbol 0
      · split
        · have same : position = symbolPosition 1 := Fin.ext (by assumption)
          rw [same]
          exact symbol 1
        · exact LowNormBlock.Block.form_eval _ _ _ _ _ encodes.logical _

end NightstreamFPrime.Export.Stage1.PiRLCSamplerCandidateWiring
