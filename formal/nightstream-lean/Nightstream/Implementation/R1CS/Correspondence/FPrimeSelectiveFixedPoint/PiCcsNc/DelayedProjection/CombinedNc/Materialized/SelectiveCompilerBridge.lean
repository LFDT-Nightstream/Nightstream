import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components

/-!
Generic semantic bridge for selectively emitted compiler rows.

Owns: the exact five-product rewrite point and ordinary retained-row point,
their evaluation under the independent thirteen-port polynomial, transport
from decoded emitted-row satisfaction to decoded rewrite/retained
obligations, and transport from a matched retained obligation to its decoded
physical source row.

Does not own: generated coefficient equality, emitted/source row pairing,
selector enforcement, source-program execution, coverage of rows eliminated
by a rewrite, transcript scheduling, parent or raw-child authority,
commitment binding, costs, or row removal.

Emits constraints: none.

Assurance tier: model-level.  In particular, `DecodedRewriteStep.sourceRows`
contains only source-row ranges.  A rewrite recurrence therefore does not, by
itself, imply satisfaction of every row in those ranges.  The generated
artifact leaf must separately connect each recurrence to the independently
decoded source program.  This file closes only the algebra common to every
such artifact and the directly retained physical rows.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selective_compiler_bridge` | State typed rewrite, elimination, retained-check, and row-satisfaction bridge obligations. | derived interface |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

private theorem modulus_eq : goldilocksP = goldilocksModulus := by
  rfl

/-! ## Decoded rewrite obligation -/

/-- Semantic value of a rewrite output.  Compiler-derived values are supplied
by the later decoded source/final-column bridge, never by a digest or label. -/
def rewriteOutputValue {sourceColumns : Nat}
    (assignment : Nat → Nat) (derivedValue : Nat → F) :
    DecodedRewriteOutput sourceColumns → F
  | .source value => linearCombinationValue value assignment
  | .derivedProductSum compilerIndex => derivedValue compilerIndex

/-- The absent predecessor is the additive zero; a present predecessor is
read from the same authoritative derived-value map as the output. -/
def rewritePreviousValue (derivedValue : Nat → F) : Option Nat → F
  | none => 0
  | some compilerIndex => derivedValue compilerIndex

def factorLeftValue {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factor : DecodedProductFactor sourceColumns) : F :=
  factor.coefficient * linearCombinationValue factor.left assignment

def factorRightValue {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factor : DecodedProductFactor sourceColumns) : F :=
  linearCombinationValue factor.right assignment

def factorLeftValueAt {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor sourceColumns)) (index : Nat) : F :=
  match factors[index]? with
  | none => 0
  | some factor => factorLeftValue assignment factor

def factorRightValueAt {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor sourceColumns)) (index : Nat) : F :=
  match factors[index]? with
  | none => 0
  | some factor => factorRightValue assignment factor

def factorValueAt {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor sourceColumns)) (index : Nat) : F :=
  match factors[index]? with
  | none => 0
  | some factor => productFactorValue factor assignment

/-- All five product positions representable by one selective evaluation row.
The generated decoder/certificate must separately establish that the decoded
factor list has length at most five. -/
def factorSum {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor sourceColumns)) : F :=
  factorValueAt assignment factors 0 +
    factorValueAt assignment factors 1 +
    factorValueAt assignment factors 2 +
    factorValueAt assignment factors 3 +
    factorValueAt assignment factors 4

def rewriteCValue {sourceColumns : Nat}
    (assignment : Nat → Nat) (derivedValue : Nat → F)
    (step : DecodedRewriteStep sourceColumns) : F :=
  rewriteOutputValue assignment derivedValue step.output +
    (-linearCombinationValue step.base assignment +
      -rewritePreviousValue derivedValue step.previous)

/-- Exact thirteen-port point represented by one compiler rewrite row. -/
def rewritePoint {sourceColumns : Nat}
    (assignment : Nat → Nat) (derivedValue : Nat → F)
    (step : DecodedRewriteStep sourceColumns) :
    Fin 13 → F
  | ⟨0, _⟩ => factorLeftValueAt assignment step.factors 0
  | ⟨1, _⟩ => 0
  | ⟨2, _⟩ => factorRightValueAt assignment step.factors 0
  | ⟨3, _⟩ => factorLeftValueAt assignment step.factors 1
  | ⟨4, _⟩ => rewriteCValue assignment derivedValue step
  | ⟨5, _⟩ => factorRightValueAt assignment step.factors 1
  | ⟨6, _⟩ => factorLeftValueAt assignment step.factors 2
  | ⟨7, _⟩ => 1
  | ⟨8, _⟩ => factorRightValueAt assignment step.factors 2
  | ⟨9, _⟩ => factorLeftValueAt assignment step.factors 3
  | ⟨10, _⟩ => factorRightValueAt assignment step.factors 3
  | ⟨11, _⟩ => factorLeftValueAt assignment step.factors 4
  | ⟨12, _⟩ => factorRightValueAt assignment step.factors 4

/-- Explicit decoded compiler recurrence.  `factorCapacity` is load-bearing:
without it, the five physical product ports need not cover the provenance. -/
def RewriteStepHolds {sourceColumns : Nat}
    (assignment : Nat → Nat) (derivedValue : Nat → F)
    (step : DecodedRewriteStep sourceColumns) : Prop :=
  step.factors.length ≤ 5 ∧
  rewriteOutputValue assignment derivedValue step.output =
    linearCombinationValue step.base assignment +
      rewritePreviousValue derivedValue step.previous +
      factorSum assignment step.factors

/-- Exact semantic pairing required from the later coefficient certificate.
The row index equality prevents a valid row from justifying the wrong
provenance record; the point equality is mathematical coefficient equality,
not a family-label or interval claim. -/
def RewriteProvenanceMatches {sourceColumns : Nat}
    (finalAssignment sourceAssignment : Nat → Nat)
    (derivedValue : Nat → F)
    (row : DecodedEmittedRow) (step : DecodedRewriteStep sourceColumns) : Prop :=
  row.emittedRow.val = step.emittedRow ∧
  emittedPoint row finalAssignment =
    rewritePoint sourceAssignment derivedValue step

private theorem factorPairAt_eq {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (factors : List (DecodedProductFactor sourceColumns)) (index : Nat) :
    factorLeftValueAt assignment factors index *
        factorRightValueAt assignment factors index =
      factorValueAt assignment factors index := by
  cases factorAt : factors[index]? with
  | none =>
      simp only [factorLeftValueAt, factorRightValueAt, factorValueAt,
        factorAt]
      exact Fin.zero_mul 0
  | some factor =>
      simp only [factorLeftValueAt, factorRightValueAt, factorValueAt,
        factorLeftValue, factorRightValue, productFactorValue, factorAt]

private theorem evaluate_rewritePoint {sourceColumns : Nat}
    (assignment : Nat → Nat) (derivedValue : Nat → F)
    (step : DecodedRewriteStep sourceColumns) :
    evaluate (rewritePoint assignment derivedValue step) =
      -(rewriteCValue assignment derivedValue step) +
        factorSum assignment step.factors := by
  have canonicalZero :
      canonicalResidual (rewritePoint assignment derivedValue step) = 0 :=
    canonicalResidual_zero_of_generalSelector_zero _ rfl
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    Role.index, rewritePoint, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Fin.one_mul, Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]
  change
    ((((-(rewriteCValue assignment derivedValue step) +
          factorLeftValueAt assignment step.factors 0 *
            factorRightValueAt assignment step.factors 0) +
        factorLeftValueAt assignment step.factors 1 *
          factorRightValueAt assignment step.factors 1) +
      factorLeftValueAt assignment step.factors 2 *
        factorRightValueAt assignment step.factors 2) +
      factorLeftValueAt assignment step.factors 3 *
        factorRightValueAt assignment step.factors 3) +
      factorLeftValueAt assignment step.factors 4 *
        factorRightValueAt assignment step.factors 4 = _
  rw [factorPairAt_eq assignment step.factors 0,
    factorPairAt_eq assignment step.factors 1,
    factorPairAt_eq assignment step.factors 2,
    factorPairAt_eq assignment step.factors 3,
    factorPairAt_eq assignment step.factors 4]
  simp only [factorSum, Lean.Grind.Fin.add_assoc]

/-- A satisfied emitted row with exact decoded coefficient provenance forces
the compiler recurrence.  Factor capacity remains explicit because it is an
artifact fact, not a consequence of the thirteen-port polynomial. -/
theorem emittedRowHolds_implies_rewriteStepHolds
    {sourceColumns : Nat} {finalAssignment sourceAssignment : Nat → Nat}
    {derivedValue : Nat → F} {row : DecodedEmittedRow}
    {step : DecodedRewriteStep sourceColumns}
    (factorCapacity : step.factors.length ≤ 5)
    (rowHolds : EmittedRowHolds row finalAssignment)
    (matching : RewriteProvenanceMatches finalAssignment sourceAssignment
      derivedValue row step) :
    RewriteStepHolds sourceAssignment derivedValue step := by
  have pointZero :
      evaluate (rewritePoint sourceAssignment derivedValue step) = 0 := by
    rw [← matching.2]
    exact rowHolds
  rw [evaluate_rewritePoint] at pointZero
  constructor
  · exact factorCapacity
  · unfold rewriteCValue at pointZero
    grind

/-! ## Directly retained source-row obligation -/

/-- Exact ordinary R1CS point represented by a retained selective row. -/
def retainedPoint {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (step : DecodedRetainedStep sourceColumns) :
    Fin 13 → F
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => linearCombinationValue step.a assignment
  | ⟨3, _⟩ => linearCombinationValue step.b assignment
  | ⟨4, _⟩ => linearCombinationValue step.c assignment
  | ⟨5, _⟩ => 0
  | ⟨6, _⟩ => 0
  | ⟨7, _⟩ => 0
  | ⟨8, _⟩ => 0
  | ⟨9, _⟩ => 0
  | ⟨10, _⟩ => 0
  | ⟨11, _⟩ => 0
  | ⟨12, _⟩ => 0

def RetainedStepHolds {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (step : DecodedRetainedStep sourceColumns) : Prop :=
  linearCombinationValue step.a assignment *
      linearCombinationValue step.b assignment =
    linearCombinationValue step.c assignment

def RetainedProvenanceMatches {sourceColumns : Nat}
    (finalAssignment sourceAssignment : Nat → Nat)
    (row : DecodedEmittedRow)
    (step : DecodedRetainedStep sourceColumns) : Prop :=
  row.emittedRow.val = step.emittedRow ∧
  emittedPoint row finalAssignment = retainedPoint sourceAssignment step

private theorem evaluate_retainedPoint {sourceColumns : Nat}
    (assignment : Nat → Nat)
    (step : DecodedRetainedStep sourceColumns) :
    evaluate (retainedPoint assignment step) =
      linearCombinationValue step.a assignment *
          linearCombinationValue step.b assignment +
        -linearCombinationValue step.c assignment := by
  have canonicalZero :
      canonicalResidual (retainedPoint assignment step) = 0 :=
    canonicalResidual_zero_of_classPorts_zero _ rfl rfl rfl rfl rfl
  rw [evaluate_eq_combinedResidual]
  unfold combinedResidual
  rw [canonicalZero]
  simp only [booleanResidual, productResidual,
    sboxResidual, centeredResidual, evaluationResidual,
    Role.index, retainedPoint, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Fin.one_mul, Fin.mul_one, Lean.Grind.AddCommGroup.neg_zero]

/-- A satisfied retained emitted row forces the exact decoded A/B/C equation.
-/
theorem emittedRowHolds_implies_retainedStepHolds
    {sourceColumns : Nat} {finalAssignment sourceAssignment : Nat → Nat}
    {row : DecodedEmittedRow} {step : DecodedRetainedStep sourceColumns}
    (rowHolds : EmittedRowHolds row finalAssignment)
    (matching : RetainedProvenanceMatches finalAssignment sourceAssignment
      row step) :
    RetainedStepHolds sourceAssignment step := by
  have pointZero : evaluate (retainedPoint sourceAssignment step) = 0 := by
    rw [← matching.2]
    exact rowHolds
  rw [evaluate_retainedPoint] at pointZero
  unfold RetainedStepHolds
  exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
    linearCombinationValue step.a sourceAssignment *
          linearCombinationValue step.b sourceAssignment -
        linearCombinationValue step.c sourceAssignment = 0 ↔
      linearCombinationValue step.a sourceAssignment *
          linearCombinationValue step.b sourceAssignment =
        linearCombinationValue step.c sourceAssignment).mp (by
          simpa [Fin.sub_eq_add_neg] using pointZero)

/-- Value of one side of a decoded physical source row. -/
def sourceSideValue {columns : Nat} (terms : List (DecodedTerm columns))
    (assignment : Nat → Nat) : F :=
  fieldResidue (lcEval assignment (termsAsNatTerms terms))

/-- Exact connection between retained provenance and a physical decoded
source row.  These are value equalities, not satisfaction assumptions; the
artifact coefficient bridge derives them from the decoded A/B/C streams and
the constant-one column. -/
def RetainedSourceMatches {sourceColumns : Nat}
    (assignment : Nat → Nat) (step : DecodedRetainedStep sourceColumns)
    (source : DecodedSourceRow) : Prop :=
  source.sourceRow.val = step.sourceRow ∧
  sourceSideValue source.a assignment =
      linearCombinationValue step.a assignment ∧
  sourceSideValue source.b assignment =
      linearCombinationValue step.b assignment ∧
  sourceSideValue source.c assignment =
    linearCombinationValue step.c assignment

/-- The directly retained compiler obligation is the corresponding physical
source-row equation once the independently decoded values are connected. -/
theorem retainedStepHolds_implies_sourceRowHolds
    {sourceColumns : Nat} {assignment : Nat → Nat}
    {step : DecodedRetainedStep sourceColumns} {source : DecodedSourceRow}
    (stepHolds : RetainedStepHolds assignment step)
    (matching : RetainedSourceMatches assignment step source) :
    SourceRowHolds source assignment := by
  have fieldEquality :
      sourceSideValue source.a assignment *
          sourceSideValue source.b assignment =
        sourceSideValue source.c assignment := by
    rw [matching.2.1, matching.2.2.1, matching.2.2.2]
    exact stepHolds
  have valueEquality := congrArg Fin.val fieldEquality
  unfold SourceRowHolds sourceRowToRow RowHolds
  simpa [sourceSideValue, fieldResidue, modulus_eq, Fin.val_mul, lcEval,
    Nat.mod_mod] using valueEquality

/-- Single-row composition for the physically retained case.  There is no
corresponding theorem for rewrite row ranges: their source equations must be
reconstructed by the later source-program artifact bridge. -/
theorem emittedRowHolds_implies_retainedSourceRowHolds
    {sourceColumns : Nat} {finalAssignment sourceAssignment : Nat → Nat}
    {emitted : DecodedEmittedRow}
    {provenance : DecodedRetainedStep sourceColumns}
    {source : DecodedSourceRow}
    (emittedHolds : EmittedRowHolds emitted finalAssignment)
    (provenanceMatches :
      RetainedProvenanceMatches finalAssignment sourceAssignment
        emitted provenance)
    (sourceMatches : RetainedSourceMatches sourceAssignment provenance source) :
    SourceRowHolds source sourceAssignment :=
  retainedStepHolds_implies_sourceRowHolds
    (emittedRowHolds_implies_retainedStepHolds
      emittedHolds provenanceMatches)
    sourceMatches

/-! ## List-level transport -/

structure RewriteLink (sourceColumns : Nat) where
  emitted : DecodedEmittedRow
  provenance : DecodedRewriteStep sourceColumns

structure RetainedSourceLink (sourceColumns : Nat) where
  emitted : DecodedEmittedRow
  provenance : DecodedRetainedStep sourceColumns
  source : DecodedSourceRow

def rewriteEmittedRows {sourceColumns : Nat}
    (links : List (RewriteLink sourceColumns)) : List DecodedEmittedRow :=
  links.map RewriteLink.emitted

def retainedEmittedRows {sourceColumns : Nat}
    (links : List (RetainedSourceLink sourceColumns)) : List DecodedEmittedRow :=
  links.map RetainedSourceLink.emitted

def retainedSourceRows {sourceColumns : Nat}
    (links : List (RetainedSourceLink sourceColumns)) : List DecodedSourceRow :=
  links.map RetainedSourceLink.source

/-- Satisfaction of an exact emitted-row list transports to every paired
rewrite recurrence.  Pair coverage and coefficient equality remain explicit
artifact obligations. -/
theorem rewriteLinksHold_of_emittedRowsSatisfy
    {sourceColumns : Nat} {finalAssignment sourceAssignment : Nat → Nat}
    {derivedValue : Nat → F} {links : List (RewriteLink sourceColumns)}
    (satisfies :
      EmittedRowsSatisfy (rewriteEmittedRows links) finalAssignment)
    (factorCapacity : ∀ link ∈ links, link.provenance.factors.length ≤ 5)
    (matching : ∀ link ∈ links,
      RewriteProvenanceMatches finalAssignment sourceAssignment derivedValue
        link.emitted link.provenance) :
    ∀ link ∈ links,
      RewriteStepHolds sourceAssignment derivedValue link.provenance := by
  intro link member
  apply emittedRowHolds_implies_rewriteStepHolds
    (factorCapacity link member)
  · apply satisfies link.emitted
    exact List.mem_map.mpr ⟨link, member, rfl⟩
  · exact matching link member

/-- Satisfaction of an exact emitted-row list transports through retained
provenance to every paired physical source row. -/
theorem retainedSourceRowsSatisfy_of_emittedRowsSatisfy
    {sourceColumns : Nat} {finalAssignment sourceAssignment : Nat → Nat}
    {links : List (RetainedSourceLink sourceColumns)}
    (satisfies :
      EmittedRowsSatisfy (retainedEmittedRows links) finalAssignment)
    (provenanceMatches : ∀ link ∈ links,
      RetainedProvenanceMatches finalAssignment sourceAssignment
        link.emitted link.provenance)
    (sourceMatches : ∀ link ∈ links,
      RetainedSourceMatches sourceAssignment link.provenance link.source) :
    SourceRowsSatisfy (retainedSourceRows links) sourceAssignment := by
  rw [sourceRowsSatisfy_iff]
  intro source sourceMember
  rcases List.mem_map.mp sourceMember with ⟨link, linkMember, rfl⟩
  apply retainedStepHolds_implies_sourceRowHolds
    (matching := sourceMatches link linkMember)
  apply emittedRowHolds_implies_retainedStepHolds
    (matching := provenanceMatches link linkMember)
  apply satisfies link.emitted
  exact List.mem_map.mpr ⟨link, linkMember, rfl⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge
