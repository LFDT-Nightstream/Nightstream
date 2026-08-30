import NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectPlan
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler

/-!
Owns the semantic bridge from the direct First54 matrix plan to the canonical
PiRLC sampler selector. It does not define another selector relation or change
the production candidate order.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectBridge

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def samplerStart (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : Nat :=
  PiRLCStarts.samplerSourceLogicalStart source.val

def selectorStart (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : Nat :=
  PiRLCStarts.selectorLogicalStart source.val

@[simp] private theorem eval_zero (env : Env) : (0 : Expr).eval env = 0 := by
  rfl

theorem selectorOffset_eq
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    Sampler.selectorOffset (samplerStart source) = selectorStart source := by
  norm_num [Sampler.selectorOffset, Sampler.windowBase, samplerStart,
    selectorStart, Sampler.entryPrivateCount, Sampler.digestRoundCount,
    DigestWindow.logicalPrivateCount, PiRLCStarts.selectorLogicalStart]

theorem acceptedExpr_eq
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    (Sampler.selectorInterface interface coordinate
        (samplerStart candidate.source)).accepted
        (selectorStart candidate.source) candidate.round =
      1 - Expr.var candidate.rejectColumn := by
  rcases candidate with ⟨source, round⟩
  have roundBound := round.isLt
  simp only [Sampler.selectorInterface, PiRLCFirst54DirectSchedule.Candidate.rejectColumn,
    PiRLCFirst54Invocations.rejectSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart, DigestWindow.reject,
    DigestWindow.laneOffset, DigestWindow.laneOf, DigestWindow.partOf,
    DigestLane.reject, DigestLane.decoderOffset, Candidate16Five.rejectExpr,
    Sampler.candidateRound, Sampler.candidatePosition, Sampler.windowOffset,
    Sampler.windowBase, samplerStart, PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart, PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart]
  simp only [First54.candidateCount, PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.roundCount] at roundBound
  norm_num [Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
    DigestLane.logicalPrivateCount, CanonicalU64.auxiliaryCount,
    Candidate16Five.auxiliaryCount,
    NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest]
      at roundBound ⊢

theorem symbolExpr_eq
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    (Sampler.selectorInterface interface coordinate
        (samplerStart candidate.source)).symbol
        (selectorStart candidate.source) candidate.round =
      Expr.var candidate.symbolColumn := by
  rcases candidate with ⟨source, round⟩
  have roundBound := round.isLt
  simp only [Sampler.selectorInterface, PiRLCFirst54DirectSchedule.Candidate.symbolColumn,
    PiRLCFirst54Invocations.remainderSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart, DigestWindow.remainder,
    DigestWindow.laneOffset, DigestWindow.laneOf, DigestWindow.partOf,
    DigestLane.remainder, DigestLane.decoderOffset,
    Candidate16Five.remainderExpr, Sampler.candidateRound,
    Sampler.candidatePosition, Sampler.windowOffset, Sampler.windowBase,
    samplerStart, PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart]
  simp only [First54.candidateCount, PiRLCFirst54DirectSchedule.roundCount,
    PiRLCFirst54Invocations.roundCount] at roundBound
  norm_num [Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
    DigestLane.logicalPrivateCount, CanonicalU64.auxiliaryCount,
    Candidate16Five.auxiliaryCount,
    NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest]
      at roundBound ⊢

theorem acceptedValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCFirst54DirectPlan.acceptedValue program base candidate =
      ((Sampler.selectorInterface interface coordinate
        (samplerStart candidate.source)).accepted
          (selectorStart candidate.source) candidate.round).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rw [acceptedExpr_eq]
  simp [PiRLCFirst54DirectPlan.acceptedValue,
    PiRLCFirst54DirectPlan.rejectValue, Expr.eval_sub]

theorem symbolValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCFirst54DirectPlan.symbolValue program base candidate =
      ((Sampler.selectorInterface interface coordinate
        (samplerStart candidate.source)).symbol
          (selectorStart candidate.source) candidate.round).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rw [symbolExpr_eq]
  rfl

theorem positionOutputValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    PiRLCFirst54DirectPlan.positionOutputValue program base descriptor =
      (First54Step.output
        (First54.positionOffset (selectorStart descriptor.candidate.source)
          descriptor.candidate.round.val) descriptor.slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rfl

theorem priorPositionValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54Step.slotCount) :
    PiRLCFirst54DirectPlan.priorPositionValue program base candidate slot =
      (First54.priorPosition (selectorStart candidate.source)
        candidate.round.val slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rcases candidate with ⟨source, ⟨round, roundBound⟩⟩
  cases round with
  | zero =>
      by_cases first : slot.val = 0 <;>
        simp [PiRLCFirst54DirectPlan.priorPositionValue,
          First54.priorPosition, First54.initialPosition, first,
          eval_zero]
  | succ previous =>
      simp [PiRLCFirst54DirectPlan.priorPositionValue,
        First54.priorPosition,
        PiRLCFirst54DirectSchedule.Position.priorPositionColumn,
        PiRLCFirst54Invocations.previousPositionSourceStart,
        PiRLCFirst54Invocations.positionSourceStart, selectorStart,
        First54Step.output]

theorem outputValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    PiRLCFirst54DirectPlan.outputValue program base descriptor =
      (First54ValueStep.output
        (First54.valueOffset (selectorStart descriptor.candidate.source)
          descriptor.candidate.round.val) descriptor.slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rfl

theorem priorValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    PiRLCFirst54DirectPlan.priorValue program base descriptor =
      (First54.priorOutput (selectorStart descriptor.candidate.source)
        descriptor.candidate.round.val descriptor.slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rcases descriptor with ⟨⟨source, ⟨round, roundBound⟩⟩, slot⟩
  cases round with
  | zero =>
      simp [PiRLCFirst54DirectPlan.priorValue, First54.priorOutput,
        eval_zero]
  | succ previous =>
      simp [PiRLCFirst54DirectPlan.priorValue, First54.priorOutput,
        PiRLCFirst54DirectSchedule.Value.priorValueColumn,
        PiRLCFirst54Invocations.previousValueSourceStart,
        PiRLCFirst54Invocations.valueSourceStart, selectorStart,
        First54ValueStep.output]

theorem finalValue_eq_eval
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    PiRLCFirst54DirectPlan.finalValue program base
        ⟨source.val, by simpa using source.isLt⟩ =
      (First54.finalFull (selectorStart source)).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rfl

theorem priorPositionValues_eq
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    PiRLCFirst54DirectPlan.priorPositionValue program base candidate =
      fun current =>
        (First54.priorPosition (selectorStart candidate.source)
          candidate.round.val current).eval
          (PiRLCFirst54DirectPlan.baseEnv program base) := by
  funext current
  exact priorPositionValue_eq_eval program base candidate current

theorem priorValues_eq
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    (fun current => PiRLCFirst54DirectPlan.priorValue program base
      ⟨candidate, current⟩) =
      fun current =>
        (First54.priorOutput (selectorStart candidate.source)
          candidate.round.val current).eval
          (PiRLCFirst54DirectPlan.baseEnv program base) := by
  funext current
  exact priorValue_eq_eval program base ⟨candidate, current⟩

theorem positionUpdate_eq
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54Step.slotCount) :
    First54Step.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate) slot =
      First54Step.update
        (((Sampler.selectorInterface interface coordinate
          (samplerStart candidate.source)).accepted
            (selectorStart candidate.source) candidate.round).eval
          (PiRLCFirst54DirectPlan.baseEnv program base))
        (fun current =>
          (First54.priorPosition (selectorStart candidate.source)
            candidate.round.val current).eval
            (PiRLCFirst54DirectPlan.baseEnv program base)) slot := by
  rw [acceptedValue_eq_eval program base interface coordinate candidate,
    priorPositionValues_eq program base candidate]

theorem valueUpdate_eq
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54ValueStep.outputCount) :
    First54ValueStep.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.symbolValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
        (fun current => PiRLCFirst54DirectPlan.priorValue program base
          ⟨candidate, current⟩) slot =
      First54ValueStep.update
        (((Sampler.selectorInterface interface coordinate
          (samplerStart candidate.source)).accepted
            (selectorStart candidate.source) candidate.round).eval
          (PiRLCFirst54DirectPlan.baseEnv program base))
        (((Sampler.selectorInterface interface coordinate
          (samplerStart candidate.source)).symbol
            (selectorStart candidate.source) candidate.round).eval
          (PiRLCFirst54DirectPlan.baseEnv program base))
        (fun current =>
          (First54.priorPosition (selectorStart candidate.source)
            candidate.round.val current).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
        (fun current =>
          (First54.priorOutput (selectorStart candidate.source)
            candidate.round.val current).eval
            (PiRLCFirst54DirectPlan.baseEnv program base)) slot := by
  rw [acceptedValue_eq_eval program base interface coordinate candidate,
    symbolValue_eq_eval program base interface coordinate candidate,
    priorPositionValues_eq program base candidate,
    priorValues_eq program base candidate]

/-- The direct source recurrence is the canonical First54 child
specification at the production sampler offset. -/
theorem sourceHolds_implies_specHolds
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (holds : PiRLCFirst54DirectPlan.SourceHolds program base source) :
    First54.SpecHolds
      (Sampler.selectorInterface interface coordinate (samplerStart source))
      (selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base) := by
  refine ⟨?_, ?_, ?_⟩
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Position := ⟨candidate, slot⟩
    have roundLt : round.val < First54.candidateCount := by
      simpa [PiRLCFirst54DirectSchedule.roundCount,
        PiRLCFirst54Invocations.roundCount] using round.isLt
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt roundLt]
    have semantic := holds.position round slot
    change PiRLCFirst54DirectPlan.positionOutputValue program base descriptor =
      First54Step.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
        slot at semantic
    simp only [First54.positionInterface]
    rw [roundEq]
    calc
      (First54Step.output
          (First54.positionOffset (selectorStart source) round.val) slot).eval
          (PiRLCFirst54DirectPlan.baseEnv program base) =
          PiRLCFirst54DirectPlan.positionOutputValue program base descriptor :=
        (positionOutputValue_eq_eval program base descriptor).symm
      _ = First54Step.update
          (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
          (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
          slot := semantic
      _ = First54Step.update
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).accepted
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorPosition (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base)) slot :=
        positionUpdate_eq program base interface coordinate candidate slot
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Value := ⟨candidate, slot⟩
    have roundLt : round.val < First54.candidateCount := by
      simpa [PiRLCFirst54DirectSchedule.roundCount,
        PiRLCFirst54Invocations.roundCount] using round.isLt
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt roundLt]
    have semantic := holds.value round slot
    change PiRLCFirst54DirectPlan.outputValue program base descriptor =
      First54ValueStep.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.symbolValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
        (fun current => PiRLCFirst54DirectPlan.priorValue program base
          ⟨candidate, current⟩) slot at semantic
    simp only [First54.valueInterface]
    rw [roundEq]
    calc
      (First54ValueStep.output
          (First54.valueOffset (selectorStart source) round.val) slot).eval
          (PiRLCFirst54DirectPlan.baseEnv program base) =
          PiRLCFirst54DirectPlan.outputValue program base descriptor :=
        (outputValue_eq_eval program base descriptor).symm
      _ = First54ValueStep.update
          (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
          (PiRLCFirst54DirectPlan.symbolValue program base candidate)
          (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
          (fun current => PiRLCFirst54DirectPlan.priorValue program base
            ⟨candidate, current⟩) slot := semantic
      _ = First54ValueStep.update
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).accepted
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).symbol
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorPosition (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorOutput (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base)) slot :=
        valueUpdate_eq program base interface coordinate candidate slot
  · rw [← finalValue_eq_eval program base source]
    exact holds.full

/-- The canonical First54 child specification recovers the direct source
recurrence without changing candidate order or offsets. -/
theorem specHolds_implies_sourceHolds
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (specification : First54.SpecHolds
      (Sampler.selectorInterface interface coordinate (samplerStart source))
      (selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base)) :
    PiRLCFirst54DirectPlan.SourceHolds program base source := by
  refine ⟨?_, ?_, ?_⟩
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Position := ⟨candidate, slot⟩
    have roundLt : round.val < First54.candidateCount := by
      simpa [PiRLCFirst54DirectSchedule.roundCount,
        PiRLCFirst54Invocations.roundCount] using round.isLt
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt roundLt]
    have canonical := specification.position round slot
    simp only [First54.positionInterface] at canonical
    rw [roundEq] at canonical
    change PiRLCFirst54DirectPlan.positionOutputValue program base descriptor =
      First54Step.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
        slot
    calc
      PiRLCFirst54DirectPlan.positionOutputValue program base descriptor =
          (First54Step.output
            (First54.positionOffset (selectorStart source) round.val) slot).eval
            (PiRLCFirst54DirectPlan.baseEnv program base) :=
        positionOutputValue_eq_eval program base descriptor
      _ = First54Step.update
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).accepted
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorPosition (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base)) slot := canonical
      _ = First54Step.update
          (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
          (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
          slot :=
        (positionUpdate_eq program base interface coordinate candidate slot).symm
  · intro round slot
    let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
    let descriptor : PiRLCFirst54DirectSchedule.Value := ⟨candidate, slot⟩
    have roundLt : round.val < First54.candidateCount := by
      simpa [PiRLCFirst54DirectSchedule.roundCount,
        PiRLCFirst54Invocations.roundCount] using round.isLt
    have roundEq : First54.candidateIndex round.val = round := by
      apply Fin.ext
      simp [First54.candidateIndex, Nat.mod_eq_of_lt roundLt]
    have canonical := specification.value round slot
    simp only [First54.valueInterface] at canonical
    rw [roundEq] at canonical
    change PiRLCFirst54DirectPlan.outputValue program base descriptor =
      First54ValueStep.update
        (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
        (PiRLCFirst54DirectPlan.symbolValue program base candidate)
        (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
        (fun current => PiRLCFirst54DirectPlan.priorValue program base
          ⟨candidate, current⟩) slot
    calc
      PiRLCFirst54DirectPlan.outputValue program base descriptor =
          (First54ValueStep.output
            (First54.valueOffset (selectorStart source) round.val) slot).eval
            (PiRLCFirst54DirectPlan.baseEnv program base) :=
        outputValue_eq_eval program base descriptor
      _ = First54ValueStep.update
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).accepted
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (((Sampler.selectorInterface interface coordinate
            (samplerStart source)).symbol
              (selectorStart source) round).eval
            (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorPosition (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base))
          (fun current =>
            (First54.priorOutput (selectorStart source) round.val current).eval
              (PiRLCFirst54DirectPlan.baseEnv program base)) slot := canonical
      _ = First54ValueStep.update
          (PiRLCFirst54DirectPlan.acceptedValue program base candidate)
          (PiRLCFirst54DirectPlan.symbolValue program base candidate)
          (PiRLCFirst54DirectPlan.priorPositionValue program base candidate)
          (fun current => PiRLCFirst54DirectPlan.priorValue program base
            ⟨candidate, current⟩) slot :=
        (valueUpdate_eq program base interface coordinate candidate slot).symm
  · calc
      PiRLCFirst54DirectPlan.finalValue program base
          ⟨source.val, by simpa using source.isLt⟩ =
          (First54.finalFull (selectorStart source)).eval
            (PiRLCFirst54DirectPlan.baseEnv program base) :=
        finalValue_eq_eval program base source
      _ = 1 := specification.full

/-- Under the explicit local custody assumptions, the direct First54 matrix
rows are exactly all 17 canonical selector child specifications. -/
theorem rowsZero_iff_all_specHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (one : assignment inputs.oneColumn = 1)
    (preserves : PiRLCFirst54DirectPlan.Preserves inputs assignment base
      (PiRLCFirst54DirectPlan.honestProducts program base)) :
    (PiRLCFirst54DirectPlan.plan inputs).RowsZero assignment ↔
      ∀ source, First54.SpecHolds
        (Sampler.selectorInterface interface coordinate (samplerStart source))
        (selectorStart source)
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  constructor
  · intro rowsZero source
    apply sourceHolds_implies_specHolds program base interface coordinate source
    exact PiRLCFirst54DirectPlan.rowsZero_implies_sourceHolds inputs assignment
      base (PiRLCFirst54DirectPlan.honestProducts program base) one preserves
      rowsZero source
  · intro specifications
    apply PiRLCFirst54DirectPlan.sourceHolds_imply_rowsZero inputs assignment
      base one preserves
    intro source
    exact specHolds_implies_sourceHolds program base interface coordinate source
      (specifications source)

/-- Vanishing direct rows imply the existing high-level bounded-sampler
relation for each canonical production selector. -/
theorem rowsZero_implies_relationHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (one : assignment inputs.oneColumn = 1)
    (preserves : PiRLCFirst54DirectPlan.Preserves inputs assignment base
      (PiRLCFirst54DirectPlan.honestProducts program base))
    (rowsZero : (PiRLCFirst54DirectPlan.plan inputs).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (assumptions : First54.Assumptions
      (Sampler.selectorInterface interface coordinate (samplerStart source))
      (selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base)) :
    First54.RelationHolds
      (Sampler.selectorInterface interface coordinate (samplerStart source))
      (selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base) := by
  apply First54.parentCoverage
    (Sampler.selectorInterface interface coordinate (samplerStart source))
    (selectorStart source) (PiRLCFirst54DirectPlan.baseEnv program base)
    assumptions
  exact (rowsZero_iff_all_specHolds inputs assignment base interface coordinate
    one preserves).mp rowsZero source

end NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectBridge
