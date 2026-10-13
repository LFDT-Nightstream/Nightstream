import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Real.Basic

/-!
Owns the knowledge-soundness contract: one field per question that an auditor
asks of a knowledge-soundness theorem, after Ironwood's `KnowledgeContract`
(zcash/ironwood `86e3c7026db8`, `book/src/formal-verification/knowledge-contract.md`).

1. What is a run? `Run`, with its finite law `weight`.
2. When does the verifier accept? `accepts`.
3. What does extraction return? `extract`, a total function; `none` is failure.
4. What does a returned witness certify? `witness_statement`.
5. What is the failure event? Accepted, yet `extract` returned `none`.
6. What is the error? `error`; `knowledge_sound` bounds the failure event by it.

The record proves nothing new: an instance names the definitions that a
theorem is stated in and applies that theorem. It does not imply
completeness: an acceptance predicate that holds nowhere satisfies every
field.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec

open scoped BigOperators

attribute [local instance low] Classical.propDecidable

/-- A knowledge-soundness contract. -/
structure KnowledgeContract where
  /-- 1. A run: everything that one experiment draws. -/
  Run : Type
  [runFinite : Fintype Run]
  /-- The probability of each run. -/
  weight : Run → ℝ
  weight_nonnegative : ∀ run, 0 ≤ weight run
  weight_sum : ∑ run, weight run = 1
  /-- The statement that the run selects. -/
  Statement : Type
  statement : Run → Statement
  /-- 2. The verifier accepts the run's proof. -/
  accepts : Run → Prop
  /-- 3. The extractor's output; `none` is failure. -/
  Witness : Type
  extract : Run → Option Witness
  /-- 4. A returned witness satisfies the relation for the run's statement. -/
  Holds : Statement → Witness → Prop
  witness_statement : ∀ run witness, extract run = some witness → Holds (statement run) witness
  /-- 6. The knowledge error. -/
  error : ℝ
  /-- 5. The run is accepted, yet extraction returns nothing, with probability
  at most `error`. -/
  knowledge_sound :
    ∑ run, weight run * (if accepts run ∧ extract run = none then 1 else 0) ≤ error

end NightstreamFPrime.Spec
