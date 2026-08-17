import Mathlib.Data.List.Sort
import Lean.Elab.Tactic.Omega
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-!
Contract: structural validation of the normalized production PiRLC
family-body row-owner ledger.

Assurance tier: artifact-checked for property
`FPRIME-PIRLC-FAMILY-BODY-ROW-LEDGER-COVER` under the supported
Goldilocks `b = 2`, `k_rho = 16` profile.

Owns exact source-row family partitions, rewrite-identifier partitions, and
the emitted-row cover. Linear-definition source rows and rewrite identifiers
are the checked complements of the retained, Poseidon2, and shifted-ternary
owners.

Does not own source-row semantics, port images, matrix actions, assignment
values, selector authority, decoder soundness, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-- A half-open row interval. Compact certificates use intervals only; they
never allocate one Boolean value per generated row. -/
structure Span where
  start : Nat
  stop : Nat
  deriving DecidableEq, Repr

def Span.contains (span : Span) (row : Nat) : Prop :=
  span.start ≤ row ∧ row < span.stop

private def fixedSpan (run : RawFixedRun) : Span :=
  { start := run.start, stop := run.start + run.length }

private def retainedSourceSpan (run : RawRetainedRun) : Span :=
  { start := run.sourceStart, stop := run.sourceStart + run.length }

private def retainedEmittedSpan (run : RawRetainedRun) : Span :=
  { start := run.emittedStart, stop := run.emittedStart + run.length }

private def rewriteSourceSpan (batch : RawRewriteBatch) : Span :=
  { start := batch.sourceStart
    stop := batch.sourceStart + batch.sourceStride * (batch.count - 1) +
      batch.sourceWidth }

private def rewriteEmittedSpan (batch : RawRewriteBatch) : Span :=
  { start := batch.emittedStart
    stop := batch.emittedStart + batch.emittedStride * (batch.count - 1) +
      batch.emittedWidth }

private def rewriteIdSpan (batch : RawRewriteBatch) : Span :=
  { start := batch.rewriteStart, stop := batch.rewriteStart + batch.count }

private def sourceLimit (raw : RawLedger) (arm : Nat) : Nat :=
  if arm = 0 then raw.evenSourceRows else raw.oddSourceRows

def fixedRunCount (raw : RawLedger) (family : RawFixedFamily) : Nat :=
  (raw.fixedRuns.filter fun run => run.family == family).length

def fixedRowCount (raw : RawLedger) (family : RawFixedFamily) : Nat :=
  (raw.fixedRuns.filter (fun run => run.family == family) |>.map fun run =>
    run.length).sum

def retainedRunCount (raw : RawLedger) (arm : Nat) : Nat :=
  (raw.retainedRuns.filter fun run => run.arm == arm).length

def retainedRowCount (raw : RawLedger) (arm : Nat) : Nat :=
  (raw.retainedRuns.filter (fun run => run.arm == arm) |>.map fun run =>
    run.length).sum

def rewriteBatchCount (raw : RawLedger) (kind : RawRewriteKind) : Nat :=
  (raw.rewriteBatches.filter fun batch => batch.kind == kind).length

def rewriteInstanceCount
    (raw : RawLedger) (kind : RawRewriteKind) (arm : Option Nat := none) : Nat :=
  (raw.rewriteBatches.filter (fun batch =>
      batch.kind == kind && arm.all fun selected => batch.arm == selected) |>.map
        fun batch => batch.count).sum

def rewriteSourceRowCount
    (raw : RawLedger) (kind : RawRewriteKind) (arm : Option Nat := none) : Nat :=
  (raw.rewriteBatches.filter (fun batch =>
      batch.kind == kind && arm.all fun selected => batch.arm == selected) |>.map
        fun batch => batch.count * batch.sourceWidth).sum

def rewriteEmittedRowCount (raw : RawLedger) (kind : RawRewriteKind) : Nat :=
  (raw.rewriteBatches.filter (fun batch => batch.kind == kind) |>.map fun batch =>
    batch.count * batch.emittedWidth).sum

private def sortSpans (spans : List Span) : List Span :=
  spans.insertionSort fun left right => left.start ≤ right.start

/-- A structural exact cover. Each interval starts at the preceding interval's
end, every interval is nonempty, and the last interval ends at `terminal`. -/
def ContiguousFrom : Nat → Nat → List Span → Bool
  | cursor, terminal, [] => decide (cursor = terminal)
  | cursor, terminal, span :: spans =>
      decide (span.start = cursor) && decide (cursor < span.stop) &&
        ContiguousFrom span.stop terminal spans

/-- A structural separation certificate. Gaps are permitted, but overlap and
out-of-range intervals are not. -/
def OrderedWithin : Nat → Nat → List Span → Bool
  | lower, upper, [] => decide (lower ≤ upper)
  | lower, upper, span :: spans =>
      decide (lower ≤ span.start) && decide (span.start < span.stop) &&
        OrderedWithin span.stop upper spans

def ExactCover (terminal : Nat) (spans : List Span) : Bool :=
  ContiguousFrom 0 terminal (sortSpans spans)

def SeparatedWithin (terminal : Nat) (spans : List Span) : Bool :=
  OrderedWithin 0 terminal (sortSpans spans)

private theorem contiguousFrom_covers
    {cursor terminal row : Nat} {spans : List Span}
    (valid : ContiguousFrom cursor terminal spans = true)
    (lower : cursor ≤ row) (upper : row < terminal) :
    ∃ span ∈ spans, span.contains row := by
  revert cursor
  induction spans with
  | nil =>
      intro cursor valid lower
      rw [ContiguousFrom] at valid
      have terminalExact : cursor = terminal := of_decide_eq_true valid
      omega
  | cons span spans ih =>
      intro cursor valid lower
      rw [ContiguousFrom] at valid
      have first := Bool.and_eq_true_iff.mp valid
      have second := Bool.and_eq_true_iff.mp first.1
      have start : span.start = cursor := of_decide_eq_true second.1
      have positive : cursor < span.stop := of_decide_eq_true second.2
      have tail : ContiguousFrom span.stop terminal spans = true := first.2
      by_cases here : row < span.stop
      · exact ⟨span, by simp, by
          constructor <;> omega⟩
      · obtain ⟨owner, member, owned⟩ := ih tail (by omega)
        exact ⟨owner, List.mem_cons_of_mem span member, owned⟩

/-- Generic kernel theorem: a compact contiguous interval certificate covers
every row in the declared domain. -/
theorem exactCover_covers
    {terminal row : Nat} {spans : List Span}
    (valid : ExactCover terminal spans = true) (bounded : row < terminal) :
    ∃ span ∈ spans, span.contains row := by
  obtain ⟨span, member, owned⟩ :=
    contiguousFrom_covers valid (Nat.zero_le row) bounded
  exact ⟨span, by simpa [sortSpans] using member, owned⟩

private theorem orderedWithin_member_start
    {lower upper : Nat} {spans : List Span}
    (valid : OrderedWithin lower upper spans = true)
    {span : Span} (member : span ∈ spans) :
    lower ≤ span.start := by
  revert lower span
  induction spans with
  | nil =>
      intro lower valid span member
      simp at member
  | cons head spans ih =>
      intro lower valid span member
      rw [OrderedWithin] at valid
      have first := Bool.and_eq_true_iff.mp valid
      have second := Bool.and_eq_true_iff.mp first.1
      have headStart : lower ≤ head.start := of_decide_eq_true second.1
      have headNonempty : head.start < head.stop :=
        of_decide_eq_true second.2
      have tail : OrderedWithin head.stop upper spans = true := first.2
      simp only [List.mem_cons] at member
      rcases member with same | member
      · simpa [same] using headStart
      · exact le_trans headStart <| le_trans (Nat.le_of_lt headNonempty)
          (ih tail member)

/-- Generic kernel theorem: an ordered compact interval certificate makes all
later intervals start after the current interval ends. -/
theorem orderedWithin_pairwise
    {lower upper : Nat} {spans : List Span}
    (valid : OrderedWithin lower upper spans = true) :
    spans.Pairwise fun left right => left.stop ≤ right.start := by
  revert lower
  induction spans with
  | nil =>
      intro lower valid
      exact .nil
  | cons head spans ih =>
      intro lower valid
      rw [OrderedWithin] at valid
      have first := Bool.and_eq_true_iff.mp valid
      have tail : OrderedWithin head.stop upper spans = true := first.2
      exact .cons
        (fun next member => orderedWithin_member_start tail member)
        (ih tail)

private def fixedArmValid (run : RawFixedRun) : Bool :=
  match run.family with
  | .armDomain => decide (run.arm = some 0 ∨ run.arm = some 1)
  | _ => decide (run.arm = none)

private def fixedRunValid (raw : RawLedger) (run : RawFixedRun) : Bool :=
  decide (0 < run.length) && decide (run.start + run.length ≤ raw.rows) &&
    fixedArmValid run

private def retainedRunValid (raw : RawLedger) (run : RawRetainedRun) : Bool :=
  decide (run.arm < 2) && decide (0 < run.length) &&
    decide (run.sourceStart + run.length ≤ sourceLimit raw run.arm) &&
    decide (run.emittedStart + run.length ≤ raw.rows)

private def rewriteFamilyValid (batch : RawRewriteBatch) : Bool :=
  match batch.kind with
  | .poseidon2 =>
      decide (batch.sourceWidth = 600 ∧
        (batch.emittedWidth = 86 ∨
          (batch.emittedWidth = 90 ∧ batch.count = 1)))
  | .shiftedTernaryCanonical =>
      decide (batch.sourceWidth = 124 ∧ batch.emittedWidth = 21)
  | .linearDefinition => false

private def rewriteBatchValid (raw : RawLedger) (batch : RawRewriteBatch) : Bool :=
  decide (0 < batch.count) && decide (batch.arm < 2) &&
    decide (batch.rewriteStride = 1) &&
    decide (batch.count = 1 ∧ batch.sourceStride = 0 ∨
      1 < batch.count ∧ batch.sourceWidth ≤ batch.sourceStride) &&
    decide (batch.count = 1 ∧ batch.emittedStride = 0 ∨
      1 < batch.count ∧ batch.emittedStride = batch.emittedWidth) &&
    decide ((rewriteSourceSpan batch).stop ≤ sourceLimit raw batch.arm) &&
    decide ((rewriteEmittedSpan batch).stop ≤ raw.rows) &&
    decide ((rewriteIdSpan batch).stop ≤ raw.rewriteCount) &&
    rewriteFamilyValid batch

private def fixedRunsValid (raw : RawLedger) : Bool :=
  raw.fixedRuns.all (fixedRunValid raw)

private def retainedRunsValid (raw : RawLedger) : Bool :=
  raw.retainedRuns.all (retainedRunValid raw)

private def rewriteBatchesValid (raw : RawLedger) : Bool :=
  raw.rewriteBatches.all (rewriteBatchValid raw)

private def sourceSpans (raw : RawLedger) (arm : Nat) : List Span :=
  (raw.retainedRuns.filter (fun run => run.arm == arm) |>.map
      retainedSourceSpan) ++
    (raw.rewriteBatches.filter (fun batch => batch.arm == arm) |>.map
      rewriteSourceSpan)

private def emittedSpans (raw : RawLedger) : List Span :=
  raw.fixedRuns.map fixedSpan ++
    raw.retainedRuns.map retainedEmittedSpan ++
    raw.rewriteBatches.map rewriteEmittedSpan

private def rewriteIdSpans (raw : RawLedger) (arm : Nat) : List Span :=
  (raw.rewriteBatches.filter (fun batch => batch.arm == arm)).map rewriteIdSpan

private def exactFixedCensus (raw : RawLedger) : Bool :=
  decide (fixedRunCount raw .selectorDomain = 1 ∧ fixedRowCount raw .selectorDomain = 2 ∧
    fixedRunCount raw .sharedDomain = 1 ∧ fixedRowCount raw .sharedDomain = 18360 ∧
    fixedRunCount raw .armDomain = 2 ∧ fixedRowCount raw .armDomain = 1408 ∧
    fixedRunCount raw .oneHot = 1 ∧ fixedRowCount raw .oneHot = 1 ∧
    fixedRunCount raw .publicPadding = 1 ∧ fixedRowCount raw .publicPadding = 7 ∧
    fixedRunCount raw .privatePadding = 1 ∧ fixedRowCount raw .privatePadding = 52 ∧
    fixedRunCount raw .ringPadding = 1 ∧ fixedRowCount raw .ringPadding = 22)

private def exactRetainedCensus (raw : RawLedger) : Bool :=
  decide (retainedRunCount raw 0 = 11 ∧ retainedRowCount raw 0 = 54545 ∧
    retainedRunCount raw 1 = 11 ∧ retainedRowCount raw 1 = 54545)

private def exactRewriteCensus (raw : RawLedger) : Bool :=
  decide (rewriteBatchCount raw .poseidon2 = 38 ∧
    rewriteInstanceCount raw .poseidon2 = 3762 ∧
    rewriteInstanceCount raw .poseidon2 (some 0) = 1880 ∧
    rewriteInstanceCount raw .poseidon2 (some 1) = 1882 ∧
    rewriteSourceRowCount raw .poseidon2 (some 0) = 1128000 ∧
    rewriteSourceRowCount raw .poseidon2 (some 1) = 1129200 ∧
    rewriteEmittedRowCount raw .poseidon2 = 323548 ∧
    rewriteBatchCount raw .shiftedTernaryCanonical = 2 ∧
    rewriteInstanceCount raw .shiftedTernaryCanonical = 1836 ∧
    rewriteInstanceCount raw .shiftedTernaryCanonical (some 0) = 918 ∧
    rewriteInstanceCount raw .shiftedTernaryCanonical (some 1) = 918 ∧
    rewriteSourceRowCount raw .shiftedTernaryCanonical (some 0) = 113832 ∧
    rewriteSourceRowCount raw .shiftedTernaryCanonical (some 1) = 113832 ∧
    rewriteEmittedRowCount raw .shiftedTernaryCanonical = 38556 ∧
    rewriteBatchCount raw .linearDefinition = 0)

private def headerExact (raw : RawLedger) : Bool :=
  decide (raw.schemaVersion = supportedSchemaVersion ∧
    raw.rows = 491046 ∧ raw.columns = 8858862 ∧
    raw.evenSourceRows = 1300897 ∧ raw.oddSourceRows = 1302097 ∧
    raw.rewriteCount = 14638 ∧
    raw.evenLinearDefinitionCount = 4520 ∧
    raw.oddLinearDefinitionCount = 4520 ∧
    raw.fixedRuns.length = 8 ∧ raw.retainedRuns.length = 22 ∧
    raw.rewriteBatches.length = 40)

private def sourceCountPartitionExact (raw : RawLedger) : Bool :=
  decide (retainedRowCount raw 0 +
      rewriteSourceRowCount raw .poseidon2 (some 0) +
      rewriteSourceRowCount raw .shiftedTernaryCanonical (some 0) +
      raw.evenLinearDefinitionCount = raw.evenSourceRows ∧
    retainedRowCount raw 1 +
      rewriteSourceRowCount raw .poseidon2 (some 1) +
      rewriteSourceRowCount raw .shiftedTernaryCanonical (some 1) +
      raw.oddLinearDefinitionCount = raw.oddSourceRows)

/-- The compact proof obligations used to establish exact ledger ownership.
No field contains an expanded row set or witness assignment. -/
structure LedgerValidFor (raw : RawLedger) : Prop where
  header : headerExact raw = true
  fixedRuns : fixedRunsValid raw = true
  retainedRuns : retainedRunsValid raw = true
  rewriteBatches : rewriteBatchesValid raw = true
  fixedCensus : exactFixedCensus raw = true
  retainedCensus : exactRetainedCensus raw = true
  rewriteCensus : exactRewriteCensus raw = true
  evenSourceSeparated :
    SeparatedWithin raw.evenSourceRows (sourceSpans raw 0) = true
  oddSourceSeparated :
    SeparatedWithin raw.oddSourceRows (sourceSpans raw 1) = true
  sourceCountPartition : sourceCountPartitionExact raw = true
  evenRewriteIds :
    ContiguousFrom 0 2798 (sortSpans (rewriteIdSpans raw 0)) = true
  oddRewriteIds :
    ContiguousFrom 7318 10118 (sortSpans (rewriteIdSpans raw 1)) = true
  evenLinearRewriteIds : 2798 + raw.evenLinearDefinitionCount = 7318
  oddLinearRewriteIds : 10118 + raw.oddLinearDefinitionCount = raw.rewriteCount
  emittedCover : ExactCover raw.rows (emittedSpans raw) = true

def LedgerValid : Prop := LedgerValidFor ledger

/-- Exact profile dimensions, established by direct projection from the
109-line generated artifact. -/
theorem dimensions_exact :
    ledger.evenSourceRows = 1300897 ∧ ledger.oddSourceRows = 1302097 ∧
      ledger.rewriteCount = 14638 ∧ ledger.rows = 491046 ∧
      ledger.columns = 8858862 := by
  decide

/-- Exact compact input lengths for the structural leaf certificates. -/
theorem certificate_input_lengths_exact :
    ledger.fixedRuns.length = 8 ∧ ledger.retainedRuns.length = 22 ∧
      ledger.rewriteBatches.length = 40 ∧
      (sourceSpans ledger 0).length = 31 ∧
      (sourceSpans ledger 1).length = 31 ∧
      (emittedSpans ledger).length = 70 := by
  decide

private theorem header_exact : headerExact ledger = true := by decide
private theorem fixed_runs_valid : fixedRunsValid ledger = true := by decide
private theorem retained_runs_valid : retainedRunsValid ledger = true := by decide
private theorem rewrite_batches_valid : rewriteBatchesValid ledger = true := by decide

/-- Exact fixed, retained, and rewrite-family census over 70 compact owner
records. -/
theorem family_census_exact :
    exactFixedCensus ledger = true ∧ exactRetainedCensus ledger = true ∧
      exactRewriteCensus ledger = true := by
  decide

private theorem even_source_separated :
    SeparatedWithin ledger.evenSourceRows (sourceSpans ledger 0) = true := by
  decide
private theorem odd_source_separated :
    SeparatedWithin ledger.oddSourceRows (sourceSpans ledger 1) = true := by
  decide
private theorem source_count_partition_exact :
    sourceCountPartitionExact ledger = true := by decide
private theorem even_rewrite_ids_exact :
    ContiguousFrom 0 2798 (sortSpans (rewriteIdSpans ledger 0)) = true := by
  decide
private theorem odd_rewrite_ids_exact :
    ContiguousFrom 7318 10118 (sortSpans (rewriteIdSpans ledger 1)) = true := by
  decide
private theorem emitted_cover_exact :
    ExactCover ledger.rows (emittedSpans ledger) = true := by decide

private def maximumList (values : List Nat) : Nat :=
  values.foldl Nat.max 0

def maximumCheckRun (raw : RawLedger) : Nat :=
  maximumList (
    (raw.fixedRuns.map fun run => run.length) ++
    (raw.retainedRuns.map fun run => run.length) ++
    (raw.rewriteBatches.flatMap fun batch =>
      [batch.count, batch.sourceWidth, batch.emittedWidth]))

/-- The largest compact interval or affine repetition count is 49,626. -/
theorem maximum_check_run_exact : maximumCheckRun ledger = 49626 := by
  decide

/-- The 109-line generated ledger satisfies every compact structural leaf.
The proof does not expand any source row, emitted row, rewrite identifier, or
witness value. -/
theorem ledger_valid : LedgerValid := by
  exact {
    header := header_exact
    fixedRuns := fixed_runs_valid
    retainedRuns := retained_runs_valid
    rewriteBatches := rewrite_batches_valid
    fixedCensus := family_census_exact.1
    retainedCensus := family_census_exact.2.1
    rewriteCensus := family_census_exact.2.2
    evenSourceSeparated := even_source_separated
    oddSourceSeparated := odd_source_separated
    sourceCountPartition := source_count_partition_exact
    evenRewriteIds := even_rewrite_ids_exact
    oddRewriteIds := odd_rewrite_ids_exact
    evenLinearRewriteIds := by decide
    oddLinearRewriteIds := by decide
    emittedCover := emitted_cover_exact
  }

/-- Every emitted selective-CCS row belongs to a compact owner interval. -/
theorem emitted_row_covered
    {row : Nat} (bounded : row < ledger.rows) :
    ∃ span ∈ emittedSpans ledger, span.contains row :=
  exactCover_covers emitted_cover_exact bounded

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
