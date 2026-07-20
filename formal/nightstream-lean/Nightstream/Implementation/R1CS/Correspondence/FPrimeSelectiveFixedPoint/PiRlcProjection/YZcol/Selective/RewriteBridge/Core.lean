import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Artifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.LinearForm
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

/-!
Decoded coefficient data for the selectively emitted fixed-point `y_zcol`
projection rows.

Owns: fail-closed rewrite/retained provenance decoding, compiler-expanded
linear forms for every compact port, and the stable rewrite/retained
pair lists consumed by the checked coefficient shards.

Does not own: coefficient equality results, selected-row satisfaction,
rewrite recurrence soundness, source-schedule grouping, selector construction,
projection authority, security events, or permission to remove rows.

Emits constraints: no.

Assurance tier: typed executable decoding over the bounded artifact facade.

| Stage path | Mathematical obligation | Authority class | Artifact owner | Lean owner |
|---|---|---|---|---|
| `pi_rlc.y_zcol.selective.decode` | raw rewrite and retained records decode with exact bounds | checked | `Materialized.Checked` | `decodeSteps` |
| `pi_rlc.y_zcol.selective.derived_registry` | every decoded derived output has the exact witness-encoder zero base, predecessor, and factors | checked | `Materialized.Checked.derivedProductSums` | `decodedDerivedRecurrenceRegistryExact` |
| `pi_rlc.y_zcol.selective.forms` | source/compiler provenance expands into every symbolic port form | computed | `Materialized.Checked.sourceArtifact` | `rewritePortLinearForm` |
| `pi_rlc.y_zcol.selective.pairs` | decoded rows and decoded steps are paired in emitted order | direct dataflow | `Materialized.Artifact` | `rewritePairs` |
-/


namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

abbrev sourceArmRowCount : Nat :=
  Materialized.Checked.sourceArtifact.scope.sourceArmRowCount

abbrev finalRelationRowCount : Nat :=
  Materialized.Checked.finalRelationRows

structure DecodedDerivedSlot where
  compilerIndex : Nat
  start : Nat
  width : Nat
  startPositive : 0 < start
  balancedWidth : width = 41
  endBound : start + width ≤ finalRelationColumnCount

def decodeDerivedSlot (raw : RawDerivedProductSum) :
    Option DecodedDerivedSlot :=
  if startPositive : 0 < raw.start then
    if balancedWidth : raw.width = 41 then
      if endBound : raw.start + raw.width ≤ finalRelationColumnCount then
        some
          { compilerIndex := raw.compilerIndex
            start := raw.start
            width := raw.width
            startPositive
            balancedWidth
            endBound }
      else
        none
    else
      none
  else
    none

def decodeDerivedSlots : Option (List DecodedDerivedSlot) :=
  Materialized.Checked.derivedProductSums.mapM decodeDerivedSlot

set_option maxRecDepth 100000 in
theorem derivedSlotsDecode : decodeDerivedSlots.isSome := by
  native_decide

def decodedDerivedSlots : List DecodedDerivedSlot :=
  decodeDerivedSlots.get derivedSlotsDecode

def derivedSlot? (compilerIndex : Nat) : Option DecodedDerivedSlot :=
  decodedDerivedSlots.find? fun slot =>
    decide (slot.compilerIndex = compilerIndex)

set_option maxRecDepth 100000 in
theorem derivedCompilerIndicesExact :
    decodedDerivedSlots.map (fun slot => slot.compilerIndex) =
      Materialized.Checked.derivedProductSums.map
        RawDerivedProductSum.compilerIndex := by
  native_decide

set_option maxRecDepth 100000 in
theorem derivedCompilerIndicesNodup :
    (decodedDerivedSlots.map (fun slot => slot.compilerIndex)).Nodup := by
  native_decide

structure DecodedSourceRowBlock where
  start : Nat
  stop : Nat
  nonempty : start < stop
  endBound : stop ≤ sourceArmRowCount

def decodeSourceRowBlock (raw : RawSourceRowBlock) :
    Option DecodedSourceRowBlock :=
  if nonempty : raw.start < raw.stop then
    if endBound : raw.stop ≤ sourceArmRowCount then
      some { start := raw.start, stop := raw.stop, nonempty, endBound }
    else
      none
  else
    none

structure DecodedProductFactor where
  left : DecodedSourceLinearCombination
  right : DecodedSourceLinearCombination
  coefficient : Nat
  coefficientPositive : 0 < coefficient
  coefficientCanonical : coefficient < goldilocksP

def decodeProductFactor (raw : RawProductFactor) :
    Option DecodedProductFactor := do
  let left ← decodeSourceLinearCombination raw.left
  let right ← decodeSourceLinearCombination raw.right
  if coefficientPositive : 0 < raw.coefficient then
    if coefficientCanonical : raw.coefficient < goldilocksP then
      pure
        { left
          right
          coefficient := raw.coefficient
          coefficientPositive
          coefficientCanonical }
    else
      none
  else
    none

inductive DecodedRewriteOutput where
  | source (value : DecodedSourceLinearCombination)
  | derivedProductSum (slot : DecodedDerivedSlot)

def decodeRewriteOutput : RawRewriteOutput → Option DecodedRewriteOutput
  | .source value => do
      let decoded ← decodeSourceLinearCombination value
      pure (.source decoded)
  | .derivedProductSum compilerIndex => do
      let slot ← derivedSlot? compilerIndex
      pure (.derivedProductSum slot)

def decodePreviousSlot : Option Nat → Option (Option DecodedDerivedSlot)
  | none => some none
  | some compilerIndex => do
      let slot ← derivedSlot? compilerIndex
      pure (some slot)

structure DecodedRewriteStep where
  emittedRow : Nat
  emittedRowBound : emittedRow < finalRelationRowCount
  rewriteId : Nat
  kind : RawRewriteKind
  sourceRows : List DecodedSourceRowBlock
  sourceRowsNonempty : sourceRows ≠ []
  output : DecodedRewriteOutput
  base : DecodedSourceLinearCombination
  previous : Option DecodedDerivedSlot
  factors : List DecodedProductFactor
  factorsNonempty : factors ≠ []
  factorCapacity : factors.length ≤ 5

def decodeRewriteStep (raw : RawRewriteStep) : Option DecodedRewriteStep :=
  if emittedRowBound : raw.emittedRow < finalRelationRowCount then do
    let sourceRows ← raw.sourceRows.mapM decodeSourceRowBlock
    if sourceRowsNonempty : sourceRows ≠ [] then
      let output ← decodeRewriteOutput raw.output
      let base ← decodeSourceLinearCombination raw.base
      let previous ← decodePreviousSlot raw.previous
      let factors ← raw.factors.mapM decodeProductFactor
      if factorsNonempty : factors ≠ [] then
        if factorCapacity : factors.length ≤ 5 then
          pure
            { emittedRow := raw.emittedRow
              emittedRowBound
              rewriteId := raw.rewriteId
              kind := raw.kind
              sourceRows
              sourceRowsNonempty
              output
              base
              previous
              factors
              factorsNonempty
              factorCapacity }
        else
          none
      else
        none
    else
      none
  else
    none

structure DecodedRetainedStep where
  emittedRow : Nat
  emittedRowBound : emittedRow < finalRelationRowCount
  sourceRow : Nat
  sourceRowBound : sourceRow < sourceArmRowCount
  a : DecodedSourceLinearCombination
  b : DecodedSourceLinearCombination
  c : DecodedSourceLinearCombination

def decodeRetainedStep (raw : RawRetainedStep) :
    Option DecodedRetainedStep :=
  if emittedRowBound : raw.emittedRow < finalRelationRowCount then
    if sourceRowBound : raw.sourceRow < sourceArmRowCount then do
      let a ← decodeSourceLinearCombination raw.a
      let b ← decodeSourceLinearCombination raw.b
      let c ← decodeSourceLinearCombination raw.c
      pure
        { emittedRow := raw.emittedRow
          emittedRowBound
          sourceRow := raw.sourceRow
          sourceRowBound
          a
          b
          c }
    else
      none
  else
    none

structure DecodedSteps where
  rewrites : List DecodedRewriteStep
  retained : List DecodedRetainedStep

def decodeSteps : Option DecodedSteps := do
  let rewrites ← Materialized.Checked.rewriteSteps.mapM decodeRewriteStep
  let retained ← Materialized.Checked.retainedSteps.mapM decodeRetainedStep
  pure { rewrites, retained }

set_option maxRecDepth 100000 in
theorem stepsDecode : decodeSteps.isSome := by
  native_decide

def decodedSteps : DecodedSteps :=
  decodeSteps.get stepsDecode

/-- Exact generated rewrite order, preserved by fail-closed decoding. -/
def decodedRewriteSteps : List DecodedRewriteStep :=
  decodedSteps.rewrites

/-- Exact generated retained-row order, preserved by fail-closed decoding. -/
def decodedRetainedSteps : List DecodedRetainedStep :=
  decodedSteps.retained

/-- Normalized recurrence payload shared by the emitted rewrite stream and the
actual Rust derived-product witness encoding. -/
structure RawDerivedRecurrence where
  compilerIndex : Nat
  base : RawSourceLinearCombination
  previous : Option Nat
  factors : List RawProductFactor
deriving DecidableEq, Repr

def rawDerivedRecurrence
    (derived : RawDerivedProductSum) : RawDerivedRecurrence :=
  { compilerIndex := derived.compilerIndex
    base := { constant := 0, terms := [] }
    previous := derived.previous
    factors := derived.factors }

def rawRewriteDerivedRecurrence
    (step : RawRewriteStep) : Option RawDerivedRecurrence :=
  match step.output with
  | .source _ => none
  | .derivedProductSum compilerIndex =>
      some
        { compilerIndex
          base := step.base
          previous := step.previous
          factors := step.factors }

def decodedSourceTermRaw (term : DecodedSourceTerm) : RawSourceTerm :=
  { column := term.column, coefficient := term.coefficient }

def decodedSourceLinearCombinationRaw
    (linear : DecodedSourceLinearCombination) : RawSourceLinearCombination :=
  { constant := linear.constant
    terms := linear.terms.map decodedSourceTermRaw }

def decodedProductFactorRaw
    (factor : DecodedProductFactor) : RawProductFactor :=
  { left := decodedSourceLinearCombinationRaw factor.left
    right := decodedSourceLinearCombinationRaw factor.right
    coefficient := factor.coefficient }

def decodedDerivedRecurrencePayload (step : DecodedRewriteStep)
    (slot : DecodedDerivedSlot) : RawDerivedRecurrence :=
  { compilerIndex := slot.compilerIndex
    base := decodedSourceLinearCombinationRaw step.base
    previous := step.previous.map fun previous => previous.compilerIndex
    factors := step.factors.map decodedProductFactorRaw }

/-- Proof fields are erased while all recurrence data consumed by the witness
encoder is preserved. -/
def decodedDerivedRecurrence
    (step : DecodedRewriteStep) : Option RawDerivedRecurrence :=
  match step.output with
  | .source _ => none
  | .derivedProductSum slot =>
      some (decodedDerivedRecurrencePayload step slot)

set_option maxRecDepth 100000 in
/-- The raw recurrence attached to every generated derived output is exactly
the one exported from Rust's witness encoder. -/
theorem derivedRecurrenceRegistryExact :
    Materialized.Checked.rewriteSteps.filterMap
        rawRewriteDerivedRecurrence =
      Materialized.Checked.derivedProductSums.map
        rawDerivedRecurrence := by
  native_decide

set_option maxRecDepth 100000 in
/-- Fail-closed decoding preserves the exact ordered Rust witness recurrence
stream: compiler index, zero additive base, predecessor, ordered factors,
coefficients, and both source linear combinations. -/
theorem decodedDerivedRecurrenceRegistryExact :
    decodedRewriteSteps.filterMap decodedDerivedRecurrence =
      Materialized.Checked.derivedProductSums.map
        rawDerivedRecurrence := by
  native_decide

/-- Every decoded derived recurrence is an entry of that exact witness stream.
This is derived from the ordered equality rather than rechecking membership. -/
theorem decodedDerivedRecurrenceRegistered :
    ∀ step ∈ decodedRewriteSteps,
      match decodedDerivedRecurrence step with
      | none => True
      | some recurrence =>
          recurrence ∈ Materialized.Checked.derivedProductSums.map
            rawDerivedRecurrence := by
  intro step member
  cases recurrenceEq : decodedDerivedRecurrence step with
  | none => trivial
  | some recurrence =>
      have inDecoded : recurrence ∈
          decodedRewriteSteps.filterMap decodedDerivedRecurrence := by
        apply List.mem_filterMap.mpr
        exact ⟨step, member, recurrenceEq⟩
      rw [decodedDerivedRecurrenceRegistryExact] at inDecoded
      exact inDecoded

/-- The decoded semantic stream preserves the witness encoder's zero-base
boundary for every compiler-derived output. -/
private def decodedDerivedOutputBaseZeroCheck : Bool :=
  decodedRewriteSteps.all fun step =>
    match step.output with
    | .source _ => true
    | .derivedProductSum _ => decide (step.base.programTerms = [])

set_option maxRecDepth 100000 in
private theorem decodedDerivedOutputBaseZeroCheck_true :
    decodedDerivedOutputBaseZeroCheck = true := by
  native_decide

theorem decodedDerivedOutputBaseZero :
    ∀ step ∈ decodedRewriteSteps,
      match step.output with
      | .source _ => True
      | .derivedProductSum _ => step.base.programTerms = [] := by
  intro step member
  have checked :=
    (List.all_eq_true.mp decodedDerivedOutputBaseZeroCheck_true) step member
  cases outputEq : step.output with
  | source linear => trivial
  | derivedProductSum slot =>
      simpa [decodedDerivedOutputBaseZeroCheck, outputEq] using checked

set_option maxRecDepth 100000 in
theorem decodedRewriteEmittedRows :
    decodedRewriteSteps.map (fun step => step.emittedRow) =
      Materialized.Checked.rewriteSteps.map RawRewriteStep.emittedRow := by
  native_decide

set_option maxRecDepth 100000 in
theorem decodedRetainedEmittedRows :
    decodedRetainedSteps.map (fun step => step.emittedRow) =
      Materialized.Checked.retainedSteps.map RawRetainedStep.emittedRow := by
  native_decide

def sourceLinearForm (linear : DecodedSourceLinearCombination) : LinearForm :=
  substituteLinearTerms compilerLinearForms linear.programTerms

def derivedLinearForm (slot : DecodedDerivedSlot) : LinearForm :=
  natTermsLinearForm (slotExpansionTerms slot.start slot.width)

def outputLinearForm : DecodedRewriteOutput → LinearForm
  | .source value => sourceLinearForm value
  | .derivedProductSum slot => derivedLinearForm slot

def previousLinearForm : Option DecodedDerivedSlot → LinearForm
  | none => []
  | some slot => derivedLinearForm slot

def negateLinearForm (form : LinearForm) : LinearForm :=
  Materialized.LinearForm.scale (-1) form

def rewriteCLinearForm (step : DecodedRewriteStep) : LinearForm :=
  outputLinearForm step.output ++
    negateLinearForm (sourceLinearForm step.base) ++
    negateLinearForm (previousLinearForm step.previous)

def factorLeftLinearForm (factor : DecodedProductFactor) : LinearForm :=
  Materialized.LinearForm.scale
    (fieldResidue factor.coefficient) (sourceLinearForm factor.left)

def factorRightLinearForm (factor : DecodedProductFactor) : LinearForm :=
  sourceLinearForm factor.right

def factorLeftLinearFormAt (factors : List DecodedProductFactor)
    (index : Nat) : LinearForm :=
  match factors[index]? with
  | none => []
  | some factor => factorLeftLinearForm factor

def factorRightLinearFormAt (factors : List DecodedProductFactor)
    (index : Nat) : LinearForm :=
  match factors[index]? with
  | none => []
  | some factor => factorRightLinearForm factor

def steadySelectorLinearForm : LinearForm :=
  natTermsLinearForm [(Materialized.Checked.steadySelectorColumn, 1)]

/-- Exact thirteen expected compact ports for one executable rewrite step. -/
def rewritePortLinearForm (step : DecodedRewriteStep) : Fin 13 → LinearForm
  | ⟨0, _⟩ => factorLeftLinearFormAt step.factors 0
  | ⟨1, _⟩ => []
  | ⟨2, _⟩ => factorRightLinearFormAt step.factors 0
  | ⟨3, _⟩ => factorLeftLinearFormAt step.factors 1
  | ⟨4, _⟩ => rewriteCLinearForm step
  | ⟨5, _⟩ => factorRightLinearFormAt step.factors 1
  | ⟨6, _⟩ => factorLeftLinearFormAt step.factors 2
  | ⟨7, _⟩ => steadySelectorLinearForm
  | ⟨8, _⟩ => factorRightLinearFormAt step.factors 2
  | ⟨9, _⟩ => factorLeftLinearFormAt step.factors 3
  | ⟨10, _⟩ => factorRightLinearFormAt step.factors 3
  | ⟨11, _⟩ => factorLeftLinearFormAt step.factors 4
  | ⟨12, _⟩ => factorRightLinearFormAt step.factors 4

/-- Exact thirteen expected compact ports for one physically retained source
R1CS row. -/
def retainedPortLinearForm (step : DecodedRetainedStep) : Fin 13 → LinearForm
  | ⟨0, _⟩ => []
  | ⟨1, _⟩ => steadySelectorLinearForm
  | ⟨2, _⟩ => sourceLinearForm step.a
  | ⟨3, _⟩ => sourceLinearForm step.b
  | ⟨4, _⟩ => sourceLinearForm step.c
  | ⟨5, _⟩ => []
  | ⟨6, _⟩ => []
  | ⟨7, _⟩ => []
  | ⟨8, _⟩ => []
  | ⟨9, _⟩ => []
  | ⟨10, _⟩ => []
  | ⟨11, _⟩ => []
  | ⟨12, _⟩ => []

def RewriteCoefficientsMatch (row : DecodedRow)
    (step : DecodedRewriteStep) : Prop :=
  ∀ port,
    Materialized.LinearForm.Equivalent
      (Materialized.LinearForm.portTerms (row.port port))
      (rewritePortLinearForm step port)

instance (row : DecodedRow) (step : DecodedRewriteStep) :
    Decidable (RewriteCoefficientsMatch row step) := by
  unfold RewriteCoefficientsMatch
  infer_instance

def RetainedCoefficientsMatch (row : DecodedRow)
    (step : DecodedRetainedStep) : Prop :=
  ∀ port,
    Materialized.LinearForm.Equivalent
      (Materialized.LinearForm.portTerms (row.port port))
      (retainedPortLinearForm step port)

instance (row : DecodedRow) (step : DecodedRetainedStep) :
    Decidable (RetainedCoefficientsMatch row step) := by
  unfold RetainedCoefficientsMatch
  infer_instance

abbrev RewritePair := DecodedRow × DecodedRewriteStep

abbrev RetainedPair := DecodedRow × DecodedRetainedStep

def rewritePairs : List RewritePair :=
  Materialized.Artifact.rewriteRows.zip decodedRewriteSteps

def retainedPairs : List RetainedPair :=
  Materialized.Artifact.retainedRows.zip decodedRetainedSteps

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
