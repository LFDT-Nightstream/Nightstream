import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.Core

/-!
Closed-chain agreement for one eliminated combined-NC source block.

Owns: symbolic aggregation of the concrete contributions in one decoded
rewrite chain, exact coefficient comparison with an independently executed
source-definition block, and transport of that comparison to equality with a
compiler assignment satisfying the same closed chain.

Does not own: generated block or chain certificates, dependency scheduling,
selected-row satisfaction, source-row satisfaction, retained checks,
transcript order, parent or child authority, commitment binding, costs, or row
removal.

The source side is derived directly from its deterministic definitions. It is
not required to satisfy the compiler rewrite steps. This keeps the comparison
usable in the production refinement without hiding the missing result in a
reconstructed-side recurrence premise.

Emits constraints: none.

Assurance tier: model-level.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.rewrite_source.chain_agreement` | Prove closed-chain agreement between source definitions and rewrite outputs. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.Core

/-! ## Symbolic contribution fold -/

/-- Symbolic form of one rewrite contribution with no predecessor term. -/
def stepContributionExpression? {columns : Nat}
    (state : SymbolicState) (step : DecodedRewriteStep columns) :
    Option Symbolic.Form :=
  recurrenceExpression? state [] step

/-- Ordered symbolic sum of every base/product contribution in a chain. -/
def contributionsExpression? {columns : Nat} (state : SymbolicState) :
    List (DecodedRewriteStep columns) → Option Symbolic.Form
  | [] => some []
  | step :: rest => do
      let head ← stepContributionExpression? state step
      let tail ← contributionsExpression? state rest
      pure (head ++ tail)

private theorem eval_stepContributionExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state)
    (step : DecodedRewriteStep columns)
    (capacity : step.factors.length ≤ 5)
    {expression : Symbolic.Form}
    (decoded : stepContributionExpression? state step = some expression) :
    Symbolic.eval (fieldAssignment assignment) expression =
      contribution assignment step := by
  have evaluated := eval_recurrenceExpression represents [] step capacity decoded
  simpa [stepContributionExpression?, contribution, Symbolic.eval,
    Lean.Grind.Fin.add_zero] using evaluated

theorem eval_contributionsExpression
    {columns : Nat} {assignment : Nat → Nat} {state : SymbolicState}
    (represents : StateRepresents assignment state) :
    ∀ {steps : List (DecodedRewriteStep columns)}
      {expression : Symbolic.Form},
      (∀ step ∈ steps, step.factors.length ≤ 5) →
      contributionsExpression? state steps = some expression →
      Symbolic.eval (fieldAssignment assignment) expression =
        contributionSum assignment steps := by
  intro steps
  induction steps with
  | nil =>
      intro expression _ decoded
      simp [contributionsExpression?] at decoded
      subst expression
      rfl
  | cons step rest inductionHypothesis =>
      intro expression capacity decoded
      cases headEq : stepContributionExpression? state step with
      | none =>
          simp [contributionsExpression?, headEq] at decoded
      | some head =>
          cases tailEq : contributionsExpression? state rest with
          | none =>
              simp [contributionsExpression?, headEq, tailEq] at decoded
          | some tail =>
              have expressionEq : expression = head ++ tail := by
                simpa [contributionsExpression?, headEq, tailEq] using decoded.symm
              subst expression
              rw [Symbolic.eval_append,
                eval_stepContributionExpression represents step
                  (capacity step (by simp)) headEq,
                inductionHypothesis
                  (fun candidate member => capacity candidate (by simp [member]))
                  tailEq]
              rfl

/-! ## Exact block match and assignment comparison -/

/-- Executable coefficient contract for one closed decoded rewrite chain and
one candidate ordered source-definition block. The block is symbolically
executed from variables; the normalized source terminal must equal the
ordered sum of the chain's concrete base/product contributions.

Membership in the generated source program and association with the decoded
chain are deliberately separate artifact-level certificates. -/
def ExactChainMatch {columns : Nat}
    (definitions : List Program.Definition)
    (steps : List (DecodedRewriteStep columns))
    (output : DecodedLinearCombination columns) : Prop :=
  match runDefinitions? variableState definitions with
  | none => False
  | some state =>
      match contributionsExpression? state steps with
      | none => False
      | some contributions =>
          (∀ step ∈ steps, step.factors.length ≤ 5) ∧
            Symbolic.Equivalent
              (decodedLinearExpression state output) contributions

instance {columns : Nat} (definitions : List Program.Definition)
    (steps : List (DecodedRewriteStep columns))
    (output : DecodedLinearCombination columns) :
    Decidable (ExactChainMatch definitions steps output) := by
  unfold ExactChainMatch
  split <;> try infer_instance
  split <;> infer_instance

/-- Exact coefficient matching plus deterministic source execution derives
the closed-chain contribution equation on the source assignment. No rewrite
row truth is assumed for that assignment. -/
theorem exactChainMatch_implies_sourceValue_eq_contributions
    {columns : Nat} {definitions : List Program.Definition}
    {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    {assignment : Nat → Nat}
    (matching : ExactChainMatch definitions steps output)
    (definitionsHold : ∀ definition ∈ definitions,
      definition.Holds assignment) :
    linearCombinationValue output assignment =
      contributionSum assignment steps := by
  cases stateEq : runDefinitions? variableState definitions with
  | none =>
      simp [ExactChainMatch, stateEq] at matching
  | some state =>
      have represents : StateRepresents assignment state :=
        runDefinitions_represents_of_holds
          (variableState_represents assignment) definitionsHold stateEq
      cases contributionsEq : contributionsExpression? state steps with
      | none =>
          simp [ExactChainMatch, stateEq, contributionsEq] at matching
      | some contributions =>
          have matched :
              (∀ step ∈ steps, step.factors.length ≤ 5) ∧
                Symbolic.Equivalent
                  (decodedLinearExpression state output) contributions := by
            simpa [ExactChainMatch, stateEq, contributionsEq] using matching
          calc
            linearCombinationValue output assignment =
                Symbolic.eval (fieldAssignment assignment)
                  (decodedLinearExpression state output) :=
              (eval_decodedLinearExpression represents output).symm
            _ = Symbolic.eval (fieldAssignment assignment) contributions :=
              Symbolic.eval_eq_of_equivalent matched.2 _
            _ = contributionSum assignment steps :=
              eval_contributionsExpression represents matched.1
                contributionsEq

/-- One exact closed source block agrees with the compiler view whenever the
compiler satisfies the decoded recurrence chain and the caller supplies
equality of every contribution. A future strict dependency schedule must
derive those equalities from previously established coordinate agreement;
without that schedule this theorem is intentionally conditional and is not a
non-circular production-refinement result. -/
theorem exactChainMatch_implies_sourceValue_eq_compiler_of_contributionsEqual
    {columns : Nat} {definitions : List Program.Definition}
    {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    {sourceAssignment compilerAssignment : Nat → Nat}
    {compilerDerivedValue : Nat → F}
    (matching : ExactChainMatch definitions steps output)
    (definitionsHold : ∀ definition ∈ definitions,
      definition.Holds sourceAssignment)
    (chain : SourceChain none steps output)
    (compilerHolds : ∀ step ∈ steps,
      RewriteStepHolds compilerAssignment compilerDerivedValue step)
    (contributionsEqual : ∀ step ∈ steps,
      contribution sourceAssignment step =
        contribution compilerAssignment step) :
    linearCombinationValue output sourceAssignment =
      linearCombinationValue output compilerAssignment := by
  calc
    linearCombinationValue output sourceAssignment =
        contributionSum sourceAssignment steps :=
      exactChainMatch_implies_sourceValue_eq_contributions
        matching definitionsHold
    _ = contributionSum compilerAssignment steps :=
      contributionSum_congr contributionsEqual
    _ = linearCombinationValue output compilerAssignment := by
      have compilerValue :=
        sourceValue_eq_previous_add_contributions chain compilerHolds
      simpa [rewritePreviousValue] using compilerValue.symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement
