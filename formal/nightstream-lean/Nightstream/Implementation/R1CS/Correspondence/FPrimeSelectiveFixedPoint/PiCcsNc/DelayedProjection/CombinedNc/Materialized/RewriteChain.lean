import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge

/-!
Kernel folding of actual decoded combined-NC rewrite recurrences.

Owns: cancellation of compiler-derived product-sum predecessors across one
ordered chain ending in a source linear combination.

Does not own: generated chain partition or order, coefficient equality,
selected-row satisfaction, source-program semantics, transcript order,
parent or raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none.

This leaf introduces no parallel verifier model.  `SourceChain` merely records
the predecessor/output equalities of the existing `DecodedRewriteStep`
records so a bounded generated certificate can group them without expanding
field algebra repeatedly.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open SelectiveCompilerBridge

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_assoc⟩
local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_comm⟩

/-- Exact predecessor/output shape of one ordered decoded recurrence chain.
Every intermediate output is the predecessor named by the next step, and the
last output is an actual source linear combination. -/
inductive SourceChain {columns : Nat} :
    Option Nat → List (DecodedRewriteStep columns) →
      DecodedLinearCombination columns → Prop where
  | terminal {previous : Option Nat} {step : DecodedRewriteStep columns}
      {output : DecodedLinearCombination columns}
      (previousExact : step.previous = previous)
      (outputExact : step.output = .source output) :
      SourceChain previous [step] output
  | derived {previous : Option Nat} {step : DecodedRewriteStep columns}
      {compilerIndex : Nat} {rest : List (DecodedRewriteStep columns)}
      {output : DecodedLinearCombination columns}
      (previousExact : step.previous = previous)
      (outputExact : step.output = .derivedProductSum compilerIndex)
      (tail : SourceChain (some compilerIndex) rest output) :
      SourceChain previous (step :: rest) output

def contribution {columns : Nat} (assignment : Nat → Nat)
    (step : DecodedRewriteStep columns) : F :=
  linearCombinationValue step.base assignment +
    factorSum assignment step.factors

def contributionSum {columns : Nat} (assignment : Nat → Nat) :
    List (DecodedRewriteStep columns) → F
  | [] => 0
  | step :: rest => contribution assignment step +
      contributionSum assignment rest

/-- Checked recurrences telescope through every compiler-derived predecessor.
The conclusion mentions only the source output, the predecessor before the
chain, and the exact contributions of the grouped decoded steps. -/
theorem sourceValue_eq_previous_add_contributions
    {columns : Nat} {assignment : Nat → Nat} {derivedValue : Nat → F}
    {previous : Option Nat} {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    (chain : SourceChain previous steps output)
    (holds : ∀ step ∈ steps,
      RewriteStepHolds assignment derivedValue step) :
    linearCombinationValue output assignment =
      rewritePreviousValue derivedValue previous +
        contributionSum assignment steps := by
  induction chain with
  | @terminal previous step output previousExact outputExact =>
      have stepHolds := holds step (by simp)
      unfold RewriteStepHolds at stepHolds
      rw [previousExact, outputExact] at stepHolds
      simp only [rewriteOutputValue] at stepHolds
      rw [stepHolds.2]
      simp only [contributionSum, contribution, Lean.Grind.Fin.add_zero]
      ac_rfl
  | @derived previous step compilerIndex rest output previousExact outputExact
      tail inductionHypothesis =>
      have headHolds := holds step (by simp)
      have tailHolds : ∀ candidate ∈ rest,
          RewriteStepHolds assignment derivedValue candidate := by
        intro candidate member
        exact holds candidate (by simp [member])
      have tailValue := inductionHypothesis tailHolds
      unfold RewriteStepHolds at headHolds
      rw [previousExact, outputExact] at headHolds
      rw [tailValue]
      simp only [rewritePreviousValue, rewriteOutputValue, contributionSum,
        contribution] at headHolds ⊢
      rw [headHolds.2]
      ac_rfl

/-- Pointwise equality of the decoded base/product contributions lifts to the
exact ordered sum used by the recurrence fold. -/
theorem contributionSum_congr
    {columns : Nat} {leftAssignment rightAssignment : Nat → Nat}
    {steps : List (DecodedRewriteStep columns)}
    (equal : ∀ step ∈ steps,
      contribution leftAssignment step =
        contribution rightAssignment step) :
    contributionSum leftAssignment steps =
      contributionSum rightAssignment steps := by
  induction steps with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [contributionSum]
      rw [equal head (by simp)]
      congr 1
      exact inductionHypothesis fun step member =>
        equal step (by simp [member])

/-- Two assignments satisfying the same actual decoded recurrence chain agree
on its source output whenever they agree on the incoming accumulator and on
every concrete base/product contribution.  This is the comparison seam used
by the generated source-execution certificate; it assumes no row or protocol
acceptance proposition. -/
theorem sourceValue_eq_of_pointwise_contributions
    {columns : Nat} {leftAssignment rightAssignment : Nat → Nat}
    {leftDerived rightDerived : Nat → F}
    {previous : Option Nat} {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    (chain : SourceChain previous steps output)
    (leftHolds : ∀ step ∈ steps,
      RewriteStepHolds leftAssignment leftDerived step)
    (rightHolds : ∀ step ∈ steps,
      RewriteStepHolds rightAssignment rightDerived step)
    (previousEqual :
      rewritePreviousValue leftDerived previous =
        rewritePreviousValue rightDerived previous)
    (contributionsEqual : ∀ step ∈ steps,
      contribution leftAssignment step =
        contribution rightAssignment step) :
    linearCombinationValue output leftAssignment =
      linearCombinationValue output rightAssignment := by
  calc
    linearCombinationValue output leftAssignment =
        rewritePreviousValue leftDerived previous +
          contributionSum leftAssignment steps :=
      sourceValue_eq_previous_add_contributions chain leftHolds
    _ = rewritePreviousValue rightDerived previous +
          contributionSum rightAssignment steps := by
      rw [previousEqual,
        contributionSum_congr contributionsEqual]
    _ = linearCombinationValue output rightAssignment :=
      (sourceValue_eq_previous_add_contributions chain rightHolds).symm

/-- Closed chains start at additive zero, so only contribution agreement is
needed. -/
theorem sourceValue_eq_of_closedChain
    {columns : Nat} {leftAssignment rightAssignment : Nat → Nat}
    {leftDerived rightDerived : Nat → F}
    {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    (chain : SourceChain none steps output)
    (leftHolds : ∀ step ∈ steps,
      RewriteStepHolds leftAssignment leftDerived step)
    (rightHolds : ∀ step ∈ steps,
      RewriteStepHolds rightAssignment rightDerived step)
    (contributionsEqual : ∀ step ∈ steps,
      contribution leftAssignment step =
        contribution rightAssignment step) :
    linearCombinationValue output leftAssignment =
      linearCombinationValue output rightAssignment := by
  apply sourceValue_eq_of_pointwise_contributions chain leftHolds rightHolds
  · rfl
  · exact contributionsEqual

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain
