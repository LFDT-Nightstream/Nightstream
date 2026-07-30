import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredTernary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness.Chunking
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

/-!
Honest-witness construction boundary for the bounded selective fixed-point
`y_zcol` projection rows.

Owns: the reverse direction of the exact coefficient bridge: an active
selector together with the decoded rewrite and retained equations satisfies
all materialized rows. It also records the exact centered-ternary slot
layout needed by the concrete witness materializer.

Does not own: source/direct semantic agreement, derivation of the rewrite
equations, producer authority, the projection security event, production
conformance, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `selected.rewrite_complete` | each independent recurrence satisfies its exact compact row | derived from checked coefficients |
| `selected.retained_complete` | each honest A/B/C check satisfies its retained row | direct dataflow |
| `selected.rows_complete` | the two exact row classes cover all decoded rows | artifact-checked |

The row theorem does not assume decoded equality or selected-row
satisfaction. Its premises are precisely the independent equations consumed
by the compact rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness.Chunking
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

private theorem rewritePoint_generalSelector
    (assignment : Nat → Nat) (step : DecodedRewriteStep) :
    rewritePoint assignment step Role.generalSelector.index = 0 := by
  rfl

private theorem rewritePoint_evalSelector {assignment : Nat → Nat}
    (step : DecodedRewriteStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    rewritePoint assignment step Role.evalSelector.index = 1 := by
  exact evalSteadySelectorLinearForm selectorOne

private theorem retainedPoint_generalSelector {assignment : Nat → Nat}
    (step : DecodedRetainedStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    retainedPoint assignment step Role.generalSelector.index = 1 := by
  exact evalSteadySelectorLinearForm selectorOne

private theorem retainedPoint_evalSelector
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.evalSelector.index = 0 := by
  rfl

private theorem retainedPoint_bit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.bit.index = 0 := by
  rfl

private theorem retainedPoint_sboxInput
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.sboxInput.index = 0 := by
  rfl

private theorem retainedPoint_centeredUnit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.centeredUnit.index = 0 := by
  rfl

private theorem retainedPoint_canonicalDigit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalDigit.index = 0 := by
  rfl

private theorem retainedPoint_canonicalBorrow
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalBorrow.index = 0 := by
  rfl

private theorem retainedPoint_canonicalNextBorrow
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalNextBorrow.index = 0 := by
  rfl

private theorem retainedPoint_canonicalBoundDigit
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.canonicalBoundDigit.index = 0 := by
  rfl

private theorem retainedPoint_evalTailRight
    (assignment : Nat → Nat) (step : DecodedRetainedStep) :
    retainedPoint assignment step Role.evalTailRight.index = 0 := by
  rfl

private theorem evaluateRewritePoint {assignment : Nat → Nat}
    (step : DecodedRewriteStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    evaluate (rewritePoint assignment step) =
      -(outputValue assignment step.output +
          (-sourceValue assignment step.base +
            -previousValue assignment step.previous)) +
        factorSum assignment step.factors := by
  have canonicalZero :
      canonicalResidual (rewritePoint assignment step) = 0 :=
    canonicalResidual_zero_of_generalSelector_zero _
      (rewritePoint_generalSelector assignment step)
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    rewritePoint_generalSelector, rewritePoint_evalSelector step selectorOne,
    Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero, Fin.one_mul,
    Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]
  change
    ((((-(evalLinearForm assignment (rewriteCLinearForm step)) +
          evalLinearForm assignment (factorLeftLinearFormAt step.factors 0) *
            evalLinearForm assignment (factorRightLinearFormAt step.factors 0)) +
        evalLinearForm assignment (factorLeftLinearFormAt step.factors 1) *
          evalLinearForm assignment (factorRightLinearFormAt step.factors 1)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 2) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 2)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 3) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 3)) +
      evalLinearForm assignment (factorLeftLinearFormAt step.factors 4) *
        evalLinearForm assignment (factorRightLinearFormAt step.factors 4) = _
  rw [evalRewriteCLinearForm,
    evalFactorPairAt assignment step.factors 0,
    evalFactorPairAt assignment step.factors 1,
    evalFactorPairAt assignment step.factors 2,
    evalFactorPairAt assignment step.factors 3,
    evalFactorPairAt assignment step.factors 4]
  simp only [factorSum, Lean.Grind.Fin.add_assoc]

private theorem evaluateRetainedPoint {assignment : Nat → Nat}
    (step : DecodedRetainedStep)
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1) :
    evaluate (retainedPoint assignment step) =
      sourceValue assignment step.a * sourceValue assignment step.b +
        -sourceValue assignment step.c := by
  have canonicalZero :
      canonicalResidual (retainedPoint assignment step) = 0 :=
    canonicalResidual_zero_of_classPorts_zero _
      (retainedPoint_canonicalDigit assignment step)
      (retainedPoint_canonicalBorrow assignment step)
      (retainedPoint_canonicalNextBorrow assignment step)
      (retainedPoint_canonicalBoundDigit assignment step)
      (retainedPoint_evalTailRight assignment step)
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    retainedPoint_generalSelector step selectorOne,
    retainedPoint_evalSelector, retainedPoint_bit, retainedPoint_sboxInput,
    retainedPoint_centeredUnit, retainedPoint_canonicalDigit,
    retainedPoint_canonicalBorrow, retainedPoint_canonicalNextBorrow,
    retainedPoint_canonicalBoundDigit, retainedPoint_evalTailRight,
    Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero, Fin.one_mul,
    Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]
  change
    evalLinearForm assignment (sourceLinearForm step.a) *
          evalLinearForm assignment (sourceLinearForm step.b) +
        -evalLinearForm assignment (sourceLinearForm step.c) = _
  rw [evalSourceLinearForm, evalSourceLinearForm, evalSourceLinearForm]

/-- One independent rewrite recurrence constructs a satisfying compact row.
This is the completeness converse of `rewritePairStepHolds`. -/
theorem rewritePairSatisfied {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    {pair : DecodedRow × DecodedRewriteStep}
    (pairMember : pair ∈ rewritePairs)
    (holds : StepHolds assignment pair.2) :
    NatRowSatisfied pair.1 assignment := by
  have matching := rewriteCoefficientsExact pair pairMember
  have pointZero : evaluate (rewritePoint assignment pair.2) = 0 := by
    rw [evaluateRewritePoint pair.2 selectorOne]
    unfold StepHolds at holds
    rw [holds]
    grind
  unfold NatRowSatisfied
  rw [← residual_fieldAssignment_eq_natResidual]
  unfold Materialized.Semantics.residual
  rw [rewriteRowPoint_eq assignment pair.1 pair.2 matching]
  exact pointZero

/-- One independent retained A/B/C equation constructs its satisfying
physical row. This is the completeness converse of `retainedPairHolds`. -/
theorem retainedPairSatisfied {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    {pair : DecodedRow × DecodedRetainedStep}
    (pairMember : pair ∈ retainedPairs)
    (holds : RetainedHolds assignment pair.2) :
    NatRowSatisfied pair.1 assignment := by
  have matching := retainedCoefficientsExact pair pairMember
  have pointZero : evaluate (retainedPoint assignment pair.2) = 0 := by
    rw [evaluateRetainedPoint pair.2 selectorOne]
    unfold RetainedHolds at holds
    rw [holds]
    grind
  unfold NatRowSatisfied
  rw [← residual_fieldAssignment_eq_natResidual]
  unfold Materialized.Semantics.residual
  rw [retainedRowPoint_eq assignment pair.1 pair.2 matching]
  exact pointZero

theorem rewriteRowsSatisfied_of_stepsHold {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    (stepsHold : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step) :
    RowsSatisfied Materialized.Artifact.rewriteRows assignment := by
  intro row rowMember
  have mappedMember : row ∈ rewritePairs.map Prod.fst := by
    rw [rewritePairRowsExact]
    exact rowMember
  rcases List.mem_map.mp mappedMember with ⟨pair, pairMember, rowEq⟩
  subst row
  apply rewritePairSatisfied selectorOne pairMember
  apply stepsHold pair.2
  have : pair.2 ∈ rewritePairs.map Prod.snd :=
    List.mem_map.mpr ⟨pair, pairMember, rfl⟩
  rw [rewritePairStepsExact] at this
  exact this

theorem retainedRowsSatisfied_of_stepsHold {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    (stepsHold : ∀ step ∈ decodedRetainedSteps,
      RetainedHolds assignment step) :
    RowsSatisfied Materialized.Artifact.retainedRows assignment := by
  intro row rowMember
  have mappedMember : row ∈ retainedPairs.map Prod.fst := by
    rw [retainedPairRowsExact]
    exact rowMember
  rcases List.mem_map.mp mappedMember with ⟨pair, pairMember, rowEq⟩
  subst row
  apply retainedPairSatisfied selectorOne pairMember
  apply stepsHold pair.2
  have : pair.2 ∈ retainedPairs.map Prod.snd :=
    List.mem_map.mpr ⟨pair, pairMember, rfl⟩
  rw [retainedPairStepsExact] at this
  exact this

private def decodedRowChunkLengths : List Nat :=
  [250, 250, 250, 250, 250, 4]

private theorem decodedRowChunkLengthsExact :
    decodedRowChunkLengths.sum =
      Materialized.Artifact.decodedRows.length := by
  rw [Materialized.Artifact.rowCount]
  decide

private theorem decodedRowChunksWithinCertificateLimit :
    ∀ length ∈ decodedRowChunkLengths, length ≤ 256 := by
  decide

private def decodedRowClassifiedCheck (row : DecodedRow) : Bool :=
  (Materialized.Checked.rewriteSteps.map RawRewriteStep.emittedRow).contains
      row.emittedRow.val ||
    (Materialized.Checked.retainedSteps.map RawRetainedStep.emittedRow).contains
      row.emittedRow.val

private def decodedRowClassificationChunkChecks : List Bool :=
  chunkChecks
    (splitByLengths decodedRowChunkLengths
      Materialized.Artifact.decodedRows)
    decodedRowClassifiedCheck

set_option maxRecDepth 100000 in
private theorem decodedRowClassificationChunkChecks_true :
    decodedRowClassificationChunkChecks.all (fun value => value) = true := by
  native_decide

private theorem everyDecodedRowClassified :
    ∀ row ∈ Materialized.Artifact.decodedRows,
      row ∈ Materialized.Artifact.rewriteRows ∨
        row ∈ Materialized.Artifact.retainedRows := by
  intro row member
  have checked := check_eq_true_of_chunkChecks
    decodedRowChunkLengths Materialized.Artifact.decodedRows
      decodedRowClassifiedCheck decodedRowChunkLengthsExact
      decodedRowClassificationChunkChecks_true row member
  cases rewrite :
      (Materialized.Checked.rewriteSteps.map RawRewriteStep.emittedRow).contains
        row.emittedRow.val with
  | true =>
    left
    apply List.mem_filter.mpr
    exact ⟨member, rewrite⟩
  | false =>
    right
    apply List.mem_filter.mpr
    have retained :
        (Materialized.Checked.retainedSteps.map RawRetainedStep.emittedRow).contains
          row.emittedRow.val = true := by
      unfold decodedRowClassifiedCheck at checked
      rw [rewrite] at checked
      exact checked
    exact ⟨member, retained⟩

/-- Exact honest-completeness boundary for all selected rows. The
premises are semantic rewrite/final-check equations, never row acceptance. -/
theorem selectedRowsSatisfied_of_stepsHold {assignment : Nat → Nat}
    (selectorOne : assignment Materialized.Checked.steadySelectorColumn = 1)
    (rewritesHold : ∀ step ∈ decodedRewriteSteps,
      StepHolds assignment step)
    (retainedHold : ∀ step ∈ decodedRetainedSteps,
      RetainedHolds assignment step) :
    RowsSatisfied Materialized.Artifact.decodedRows assignment := by
  have rewriteRows := rewriteRowsSatisfied_of_stepsHold selectorOne rewritesHold
  have retainedRows := retainedRowsSatisfied_of_stepsHold selectorOne retainedHold
  intro row rowMember
  rcases everyDecodedRowClassified row rowMember with rewrite | retained
  · exact rewriteRows row rewrite
  · exact retainedRows row retained

/-! ## Exact centered-ternary slot materializer -/

inductive SlotOwner where
  | source (column : Nat)
  | derived (compilerIndex : Nat)
deriving DecidableEq

structure EncodingSlot where
  owner : SlotOwner
  start : Nat
deriving DecidableEq

def sourceEncodingSlots : List EncodingSlot :=
  SourceDecode.decoded.slots.map fun slot =>
    { owner := .source slot.column, start := slot.start }

def derivedEncodingSlots : List EncodingSlot :=
  decodedDerivedSlots.map fun slot =>
    { owner := .derived slot.compilerIndex, start := slot.start }

def encodingSlots : List EncodingSlot :=
  sourceEncodingSlots ++ derivedEncodingSlots

private def sourceSlotChunkLengths : List Nat :=
  List.replicate 44 250 ++ [22]

set_option maxRecDepth 100000 in
private theorem sourceSlotCountExact :
    SourceDecode.decoded.slots.length = 11022 := by
  native_decide

private theorem sourceSlotChunkLengthsExact :
    sourceSlotChunkLengths.sum = SourceDecode.decoded.slots.length := by
  rw [sourceSlotCountExact]
  decide

private theorem sourceSlotChunksWithinCertificateLimit :
    ∀ length ∈ sourceSlotChunkLengths, length ≤ 256 := by
  intro length member
  simp only [sourceSlotChunkLengths, List.mem_append,
    List.mem_replicate, List.mem_singleton] at member
  rcases member with ⟨_, equal⟩ | equal <;> omega

private theorem sourceSlotChunkSizesExact :
    (splitByLengths sourceSlotChunkLengths
      SourceDecode.decoded.slots).map List.length =
        sourceSlotChunkLengths := by
  apply splitByLengths_lengths
  exact Nat.le_of_eq sourceSlotChunkLengthsExact

private def sourceSlotWidthCheck
    (slot : SourceDecode.DecodedSourceSlot) : Bool :=
  decide (slot.width = digitCount)

private def sourceSlotWidthChunkChecks : List Bool :=
  chunkChecks
    (splitByLengths sourceSlotChunkLengths SourceDecode.decoded.slots)
    sourceSlotWidthCheck

set_option maxRecDepth 100000 in
private theorem sourceSlotWidthChunkChecks_true :
    sourceSlotWidthChunkChecks.all (fun value => value) = true := by
  native_decide

/-- Exact artifact fact: every retained source slot uses the balanced
`digitCount`-coordinate vocabulary. -/
theorem sourceSlotWidthsExact :
    ∀ slot ∈ SourceDecode.decoded.slots, slot.width = digitCount := by
  intro slot member
  exact of_decide_eq_true (check_eq_true_of_chunkChecks
    sourceSlotChunkLengths SourceDecode.decoded.slots
      sourceSlotWidthCheck sourceSlotChunkLengthsExact
      sourceSlotWidthChunkChecks_true slot member)

private def encodingSlotChunkLengths : List Nat :=
  List.replicate 48 250 ++ [2]

set_option maxRecDepth 100000 in
private theorem encodingSlotCountExact : encodingSlots.length = 12002 := by
  native_decide

private theorem encodingSlotChunkLengthsExact :
    encodingSlotChunkLengths.sum = encodingSlots.length := by
  rw [encodingSlotCountExact]
  decide

private theorem encodingSlotChunksWithinCertificateLimit :
    ∀ length ∈ encodingSlotChunkLengths, length ≤ 256 := by
  intro length member
  simp only [encodingSlotChunkLengths, List.mem_append,
    List.mem_replicate, List.mem_singleton] at member
  rcases member with ⟨_, equal⟩ | equal <;> omega

private theorem encodingSlotChunkSizesExact :
    (splitByLengths encodingSlotChunkLengths encodingSlots).map List.length =
      encodingSlotChunkLengths := by
  apply splitByLengths_lengths
  exact Nat.le_of_eq encodingSlotChunkLengthsExact

private def encodingSlotStartAfterCheck (slot : EncodingSlot) : Bool :=
  decide (Materialized.Checked.steadySelectorColumn + 1 ≤ slot.start)

private def encodingSlotStartAfterChunkChecks : List Bool :=
  chunkChecks (splitByLengths encodingSlotChunkLengths encodingSlots)
    encodingSlotStartAfterCheck

set_option maxRecDepth 100000 in
private theorem encodingSlotStartAfterChunkChecks_true :
    encodingSlotStartAfterChunkChecks.all (fun value => value) = true := by
  native_decide

private theorem encodingSlotStartAfter (slot : EncodingSlot)
    (member : slot ∈ encodingSlots) :
    Materialized.Checked.steadySelectorColumn + 1 ≤ slot.start := by
  exact of_decide_eq_true (check_eq_true_of_chunkChecks
    encodingSlotChunkLengths encodingSlots encodingSlotStartAfterCheck
      encodingSlotChunkLengthsExact encodingSlotStartAfterChunkChecks_true
      slot member)

/-! The materializer uses a persistent coordinate index. Building the map is
linear in the checked slot coordinates; subsequent lookup and certification
do not rescan the concrete slot list. -/

def insertSlotCoordinates
    (index : Std.HashMap Nat (SlotOwner × Nat))
    (slot : EncodingSlot) : Std.HashMap Nat (SlotOwner × Nat) :=
  (List.range digitCount).foldl (fun index digit =>
    index.insert (slot.start + digit) (slot.owner, digit)) index

def encodingCoordinateMap : Std.HashMap Nat (SlotOwner × Nat) :=
  encodingSlots.foldl insertSlotCoordinates {}

def locateSlotDigit (column : Nat) : Option (SlotOwner × Nat) :=
  encodingCoordinateMap[column]?

private def encodingSlotLookupCheckIn
    (coordinateMap : Std.HashMap Nat (SlotOwner × Nat))
    (slot : EncodingSlot) : Bool :=
    (List.range digitCount).all fun digit =>
      decide (coordinateMap[slot.start + digit]? =
        some (slot.owner, digit))

private def encodingCoordinateCertificateCheck : Bool :=
  let index := encodingCoordinateMap
  decide (index.size = encodingSlots.length * digitCount) &&
    (chunkChecks (splitByLengths encodingSlotChunkLengths encodingSlots)
      (encodingSlotLookupCheckIn index)).all (fun value => value)

set_option maxRecDepth 100000 in
private theorem encodingCoordinateCertificateCheck_true :
    encodingCoordinateCertificateCheck = true := by
  native_decide

private theorem encodingCoordinateCertificate :
    encodingCoordinateMap.size = encodingSlots.length * digitCount ∧
      (chunkChecks (splitByLengths encodingSlotChunkLengths encodingSlots)
        (encodingSlotLookupCheckIn encodingCoordinateMap)).all
          (fun value => value) = true := by
  have checked :
      decide
          (encodingCoordinateMap.size =
            encodingSlots.length * digitCount) = true ∧
        (chunkChecks
          (splitByLengths encodingSlotChunkLengths encodingSlots)
          (encodingSlotLookupCheckIn encodingCoordinateMap)).all
            (fun value => value) = true := by
    simpa only [encodingCoordinateCertificateCheck,
      Bool.and_eq_true] using encodingCoordinateCertificateCheck_true
  exact ⟨of_decide_eq_true checked.1, checked.2⟩

private theorem locateSlotDigit_at (slot : EncodingSlot)
    (member : slot ∈ encodingSlots) (digit : Fin digitCount) :
    locateSlotDigit (slot.start + digit.val) =
      some (slot.owner, digit.val) := by
  have slotChecked := check_eq_true_of_chunkChecks
    encodingSlotChunkLengths encodingSlots
      (encodingSlotLookupCheckIn encodingCoordinateMap)
      encodingSlotChunkLengthsExact encodingCoordinateCertificate.2
      slot member
  unfold encodingSlotLookupCheckIn at slotChecked
  exact of_decide_eq_true
    ((List.all_eq_true.mp slotChecked) digit.val
      (List.mem_range.mpr digit.isLt))

def encodedLookupValue (values : SlotOwner → Nat) :
    Option (SlotOwner × Nat) → Nat
  | none => 0
  | some (owner, index) =>
      if indexLt : index < digitCount then
        finiteEncode (values owner) ⟨index, indexLt⟩
      else
        0

/-- Deterministic final-relation assignment for all retained and derived
balanced words. The distinguished constant and selector coordinates are set
directly and are disjoint from every checked word interval. -/
def materializeAssignment (values : SlotOwner → Nat) : Nat → Nat :=
  fun column =>
    if column = Materialized.Checked.constantOneColumn then 1
    else if column = Materialized.Checked.steadySelectorColumn then 1
    else encodedLookupValue values (locateSlotDigit column)

theorem materializeAssignment_constantOne (values : SlotOwner → Nat) :
    materializeAssignment values Materialized.Checked.constantOneColumn = 1 := by
  simp [materializeAssignment]

theorem materializeAssignment_selectorOne (values : SlotOwner → Nat) :
    materializeAssignment values
      Materialized.Checked.steadySelectorColumn = 1 := by
  have distinct : Materialized.Checked.steadySelectorColumn ≠
      Materialized.Checked.constantOneColumn := by
    native_decide
  simp [materializeAssignment, distinct]

theorem materializeAssignment_at
    (values : SlotOwner → Nat) (slot : EncodingSlot)
    (member : slot ∈ encodingSlots) (digit : Fin digitCount) :
    materializeAssignment values (slot.start + digit.val) =
      finiteEncode (values slot.owner) digit := by
  have startBound :
      Materialized.Checked.steadySelectorColumn + 1 ≤ slot.start :=
    encodingSlotStartAfter slot member
  have notConstant : slot.start + digit.val ≠
      Materialized.Checked.constantOneColumn := by
    have distinguished := Materialized.Checked.distinguishedColumns
    omega
  have notSelector : slot.start + digit.val ≠
      Materialized.Checked.steadySelectorColumn := by
    omega
  unfold materializeAssignment
  rw [if_neg notConstant, if_neg notSelector,
    locateSlotDigit_at slot member digit]
  change (if indexLt : digit.val < digitCount then
      finiteEncode (values slot.owner) ⟨digit.val, indexLt⟩ else 0) = _
  rw [dif_pos digit.isLt]

private theorem finiteEncode_value_canonical (source : Nat)
    (digit : Fin digitCount) : finiteEncode source digit < goldilocksP := by
  rcases finiteEncode_alphabet source digit with negative | zero | one
  · rw [negative]
    decide
  · rw [zero]
    decide
  · rw [one]
    decide

private theorem encodedLookupValue_canonical (values : SlotOwner → Nat) :
    ∀ located, encodedLookupValue values located < goldilocksP := by
  intro located
  cases located with
  | none =>
      simpa [encodedLookupValue] using (show 0 < goldilocksP by decide)
  | some located =>
      rcases located with ⟨owner, index⟩
      simp only [encodedLookupValue]
      split
      · exact finiteEncode_value_canonical _ _
      · decide

theorem materializeAssignment_canonical (values : SlotOwner → Nat) :
    AssignmentCanonical (materializeAssignment values) := by
  intro column
  simp only [materializeAssignment]
  split
  · decide
  · split
    · decide
    · exact encodedLookupValue_canonical values _

private theorem lcEval_materializedWord
    (values : SlotOwner → Nat) (slot : EncodingSlot)
    (member : slot ∈ encodingSlots)
    (valueCanonical : values slot.owner < goldilocksP) :
    lcEval (materializeAssignment values)
        (SourceDecode.slotExpansionTerms slot.start digitCount) =
      values slot.owner := by
  have folded :=
    Nightstream.Implementation.R1CS.ShiftedTernarySound.foldl_range_eq_lowValue
      (fun index => materializeAssignment values (slot.start + index))
      0 digitCount
  have decoded := decodeFiniteWord_finiteEncode valueCanonical
  have lowValuesEqual :
      Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (fun index => materializeAssignment values (slot.start + index))
          digitCount =
        Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
          (wordAt (finiteEncode (values slot.owner))) digitCount := by
    apply Nightstream.Implementation.R1CS.ShiftedTernaryComplete.lowValue_congr
    intro index indexLt
    rw [materializeAssignment_at values slot member ⟨index, indexLt⟩]
    simp [wordAt, indexLt]
  calc
    lcEval (materializeAssignment values)
          (SourceDecode.slotExpansionTerms slot.start digitCount) =
        Nightstream.Implementation.R1CS.ShiftedTernarySound.lowValue
            (fun index => materializeAssignment values (slot.start + index))
            digitCount % goldilocksP := by
          simpa [lcEval, SourceDecode.slotExpansionTerms,
            SourceDecode.slotRadix, digitCount, List.foldl_map] using
            congrArg (fun value => value % goldilocksP) folded
    _ = decodeFiniteWord (finiteEncode (values slot.owner)) := by
          rw [lowValuesEqual]
          rfl
    _ = values slot.owner := decoded

theorem sourceSlot_decodes
    (values : SlotOwner → Nat) (slot : SourceDecode.DecodedSourceSlot)
    (member : slot ∈ SourceDecode.decoded.slots)
    (valueCanonical : values (.source slot.column) < goldilocksP) :
    lcEval (materializeAssignment values) slot.expansionTerms =
      values (.source slot.column) := by
  let encoded : EncodingSlot :=
    { owner := .source slot.column, start := slot.start }
  have encodedMember : encoded ∈ encodingSlots := by
    apply List.mem_append_left
    exact List.mem_map.mpr ⟨slot, member, rfl⟩
  have width := sourceSlotWidthsExact slot member
  simpa [encoded, SourceDecode.DecodedSourceSlot.expansionTerms, width] using
    lcEval_materializedWord values encoded encodedMember valueCanonical

theorem derivedSlot_decodes
    (values : SlotOwner → Nat) (slot : DecodedDerivedSlot)
    (member : slot ∈ decodedDerivedSlots)
    (valueCanonical : values (.derived slot.compilerIndex) < goldilocksP) :
    lcEval (materializeAssignment values)
        (SourceDecode.slotExpansionTerms slot.start slot.width) =
      values (.derived slot.compilerIndex) := by
  let encoded : EncodingSlot :=
    { owner := .derived slot.compilerIndex, start := slot.start }
  have encodedMember : encoded ∈ encodingSlots := by
    apply List.mem_append_right
    exact List.mem_map.mpr ⟨slot, member, rfl⟩
  simpa [encoded, slot.balancedWidth] using
    lcEval_materializedWord values encoded encodedMember valueCanonical

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness
