import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-!
Contract: independent structural validation of the normalized production
PiRLC family-body row-owner ledger.

Assurance tier: artifact-checked for property
`FPRIME-PIRLC-FAMILY-BODY-ROW-LEDGER-COVER`.

Owns exact source-row, rewrite-identifier, and emitted-row cover. It also owns
the fixed-family census and the supported rewrite widths for both parity arms.

Does not own source-row semantics, port images, matrix actions, assignment
values, selector authority, decoder soundness, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

structure Ownership where
  evenSource : Array Bool
  oddSource : Array Bool
  rewriteIds : Array Bool
  emitted : Array Bool
  evenMarked : Nat
  oddMarked : Nat
  rewriteMarked : Nat
  emittedMarked : Nat

private def markSlot (slots : Array Bool) (index : Nat) : Option (Array Bool) :=
  match slots[index]? with
  | some false => some (slots.set! index true)
  | _ => none

private def markRange : Nat -> Nat -> Array Bool -> Option (Array Bool)
  | _, 0, slots => some slots
  | start, remaining + 1, slots => do
      let slots <- markSlot slots start
      markRange (start + 1) remaining slots

private def markSourceRange
    (arm start length : Nat)
    (state : Ownership) : Option Ownership :=
  if arm = 0 then do
    let slots <- markRange start length state.evenSource
    some { state with
      evenSource := slots
      evenMarked := state.evenMarked + length }
  else if arm = 1 then do
    let slots <- markRange start length state.oddSource
    some { state with
      oddSource := slots
      oddMarked := state.oddMarked + length }
  else
    none

private def markRewrite (rewrite : Nat) (state : Ownership) : Option Ownership := do
  let slots <- markSlot state.rewriteIds rewrite
  some { state with
    rewriteIds := slots
    rewriteMarked := state.rewriteMarked + 1 }

private def markEmittedRange
    (start length : Nat)
    (state : Ownership) : Option Ownership := do
  let slots <- markRange start length state.emitted
  some { state with
    emitted := slots
    emittedMarked := state.emittedMarked + length }

private def fixedArmValid (run : RawFixedRun) : Bool :=
  match run.family with
  | .armDomain => decide (run.arm = some 0 \/ run.arm = some 1)
  | _ => decide (run.arm = none)

private def checkFixedRuns : List RawFixedRun -> Ownership -> Option Ownership
  | [], state => some state
  | run :: runs, state =>
      if run.length = 0 || !fixedArmValid run then
        none
      else do
        let state <- markEmittedRange run.start run.length state
        checkFixedRuns runs state

private def checkRetainedRuns : List RawRetainedRun -> Ownership -> Option Ownership
  | [], state => some state
  | run :: runs, state =>
      if run.length = 0 || 1 < run.arm then
        none
      else do
        let state <- markSourceRange run.arm run.sourceStart run.length state
        let state <- markEmittedRange run.emittedStart run.length state
        checkRetainedRuns runs state

private def rewriteShapeValid (rows : Nat) (batch : RawRewriteBatch) : Bool :=
  let common := decide (0 < batch.count /\
    0 < batch.rewriteStride /\
    batch.arm < 2 /\
    (batch.count = 1 \/ 0 < batch.sourceStride) /\
    (batch.count = 1 \/ batch.emittedWidth = 0 \/ 0 < batch.emittedStride) /\
    batch.emittedStart <= rows)
  let family := match batch.kind with
    | .poseidon2 => decide (batch.sourceWidth = 600 /\
        (batch.emittedWidth = 86 \/
          (batch.emittedWidth = 90 /\ batch.count = 1)))
    | .shiftedTernaryCanonical =>
        decide (batch.sourceWidth = 124 /\ batch.emittedWidth = 21)
    | .linearDefinition =>
        decide (batch.sourceWidth = 1 /\ batch.emittedWidth = 0)
  common && family

private def checkRewriteBatchInstances
    (batch : RawRewriteBatch) : Nat -> Nat -> Ownership -> Option Ownership
  | _, 0, state => some state
  | index, remaining + 1, state => do
      let rewrite := batch.rewriteStart + batch.rewriteStride * index
      let sourceStart := batch.sourceStart + batch.sourceStride * index
      let emittedStart := batch.emittedStart + batch.emittedStride * index
      let state <- markRewrite rewrite state
      let state <- markSourceRange batch.arm sourceStart batch.sourceWidth state
      let state <- if batch.emittedWidth = 0 then
        some state
      else
        markEmittedRange emittedStart batch.emittedWidth state
      checkRewriteBatchInstances batch (index + 1) remaining state

private def checkRewriteBatches
    (rows : Nat) : List RawRewriteBatch -> Ownership -> Option Ownership
  | [], state => some state
  | batch :: batches, state =>
      if !rewriteShapeValid rows batch then
        none
      else do
        let state <- checkRewriteBatchInstances batch 0 batch.count state
        checkRewriteBatches rows batches state

def fixedRunCount (raw : RawLedger) (family : RawFixedFamily) : Nat :=
  (raw.fixedRuns.filter fun run => run.family == family).length

def fixedRowCount (raw : RawLedger) (family : RawFixedFamily) : Nat :=
  (raw.fixedRuns.filter (fun run => run.family == family) |>.map fun run => run.length).sum

def retainedRunCount (raw : RawLedger) (arm : Nat) : Nat :=
  (raw.retainedRuns.filter fun run => run.arm == arm).length

def retainedRowCount (raw : RawLedger) (arm : Nat) : Nat :=
  (raw.retainedRuns.filter (fun run => run.arm == arm) |>.map fun run => run.length).sum

def rewriteBatchCount (raw : RawLedger) (kind : RawRewriteKind) : Nat :=
  (raw.rewriteBatches.filter fun batch => batch.kind == kind).length

def rewriteInstanceCount
    (raw : RawLedger) (kind : RawRewriteKind) (arm : Option Nat := none) : Nat :=
  (raw.rewriteBatches.filter (fun batch =>
      batch.kind == kind && arm.all fun arm => batch.arm == arm) |>.map fun batch => batch.count).sum

def rewriteSourceRowCount
    (raw : RawLedger) (kind : RawRewriteKind) (arm : Option Nat := none) : Nat :=
  (raw.rewriteBatches.filter (fun batch =>
      batch.kind == kind && arm.all fun arm => batch.arm == arm) |>.map fun batch =>
        batch.count * batch.sourceWidth).sum

def rewriteEmittedRowCount (raw : RawLedger) (kind : RawRewriteKind) : Nat :=
  (raw.rewriteBatches.filter (fun batch => batch.kind == kind) |>.map fun batch =>
    batch.count * batch.emittedWidth).sum

private def exactFixedCensus (raw : RawLedger) : Bool :=
  decide (fixedRunCount raw .selectorDomain = 1 /\ fixedRowCount raw .selectorDomain = 2 /\
    fixedRunCount raw .sharedDomain = 1 /\ fixedRowCount raw .sharedDomain = 32826 /\
    fixedRunCount raw .armDomain = 2 /\ fixedRowCount raw .armDomain = 1280 /\
    fixedRunCount raw .oneHot = 1 /\ fixedRowCount raw .oneHot = 1 /\
    fixedRunCount raw .publicPadding = 1 /\ fixedRowCount raw .publicPadding = 7 /\
    fixedRunCount raw .privatePadding = 1 /\ fixedRowCount raw .privatePadding = 52 /\
    fixedRunCount raw .ringPadding = 1 /\ fixedRowCount raw .ringPadding = 33)

private def exactRetainedCensus (raw : RawLedger) : Bool :=
  decide (retainedRunCount raw 0 = 7 /\ retainedRowCount raw 0 = 46258 /\
    retainedRunCount raw 1 = 7 /\ retainedRowCount raw 1 = 46258)

private def exactRewriteCensus (raw : RawLedger) : Bool :=
  decide (rewriteBatchCount raw .poseidon2 = 14 /\
    rewriteInstanceCount raw .poseidon2 = 1376 /\
    rewriteInstanceCount raw .poseidon2 (some 0) = 687 /\
    rewriteInstanceCount raw .poseidon2 (some 1) = 689 /\
    rewriteSourceRowCount raw .poseidon2 (some 0) = 412200 /\
    rewriteSourceRowCount raw .poseidon2 (some 1) = 413400 /\
    rewriteEmittedRowCount raw .poseidon2 = 118352 /\
    rewriteBatchCount raw .shiftedTernaryCanonical = 2 /\
    rewriteInstanceCount raw .shiftedTernaryCanonical = 1620 /\
    rewriteInstanceCount raw .shiftedTernaryCanonical (some 0) = 810 /\
    rewriteInstanceCount raw .shiftedTernaryCanonical (some 1) = 810 /\
    rewriteEmittedRowCount raw .shiftedTernaryCanonical = 34020 /\
    rewriteBatchCount raw .linearDefinition = 10 /\
    rewriteInstanceCount raw .linearDefinition = 68 /\
    rewriteInstanceCount raw .linearDefinition (some 0) = 34 /\
    rewriteInstanceCount raw .linearDefinition (some 1) = 34 /\
    rewriteEmittedRowCount raw .linearDefinition = 0)

def exactShape (raw : RawLedger) : Bool :=
  decide (raw.schemaVersion = supportedSchemaVersion /\
    raw.rows = 279089 /\ raw.columns = 2484972 /\
    raw.evenSourceRows = 558932 /\ raw.oddSourceRows = 560132 /\
    raw.rewriteCount = 3064 /\ raw.fixedRuns.length = 8 /\
    raw.retainedRuns.length = 14 /\ raw.rewriteBatches.length = 26) &&
  exactFixedCensus raw && exactRetainedCensus raw && exactRewriteCensus raw

def validateLedger (raw : RawLedger) : Bool :=
  if !exactShape raw then
    false
  else
    let initial : Ownership :=
      { evenSource := Array.replicate raw.evenSourceRows false
        oddSource := Array.replicate raw.oddSourceRows false
        rewriteIds := Array.replicate raw.rewriteCount false
        emitted := Array.replicate raw.rows false
        evenMarked := 0
        oddMarked := 0
        rewriteMarked := 0
        emittedMarked := 0 }
    match checkFixedRuns raw.fixedRuns initial with
    | none => false
    | some afterFixed =>
        match checkRetainedRuns raw.retainedRuns afterFixed with
        | none => false
        | some afterRetained =>
            match checkRewriteBatches raw.rows raw.rewriteBatches afterRetained with
            | none => false
            | some complete => decide (complete.evenMarked = raw.evenSourceRows /\
                complete.oddMarked = raw.oddSourceRows /\
                complete.rewriteMarked = raw.rewriteCount /\
                complete.emittedMarked = raw.rows)

def LedgerValid : Prop := validateLedger ledger = true

private def maximumList (values : List Nat) : Nat :=
  values.foldl Nat.max 0

def maximumCheckRun (raw : RawLedger) : Nat :=
  maximumList (
    (raw.fixedRuns.map fun run => run.length) ++
    (raw.retainedRuns.map fun run => run.length) ++
    (raw.rewriteBatches.flatMap fun batch => [batch.count, batch.sourceWidth, batch.emittedWidth]))

/-- The artifact fixes both normalized source domains and the emitted domain. -/
theorem dimensions_exact :
    ledger.evenSourceRows = 558932 /\ ledger.oddSourceRows = 560132 /\
      ledger.rewriteCount = 3064 /\ ledger.rows = 279089 /\
      ledger.columns = 2484972 := by
  decide

/-- The compact artifact has the exact fixed, retained, and rewrite census. -/
theorem family_census_exact :
    exactFixedCensus ledger = true /\
      exactRetainedCensus ledger = true /\
      exactRewriteCensus ledger = true := by
  native_decide

/-- The largest recursive range check has 43,794 rows. -/
theorem maximum_check_run_exact : maximumCheckRun ledger = 43794 := by
  native_decide

/-- The ledger owns every source row, rewrite identifier, and emitted row
exactly once, and every expanded interval is in bounds. -/
theorem ledger_valid : LedgerValid := by
  unfold LedgerValid
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
