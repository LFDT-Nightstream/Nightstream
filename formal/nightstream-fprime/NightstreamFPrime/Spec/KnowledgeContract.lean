import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Real.Basic

/-!
Owns the knowledge-soundness contract: one field per question that an auditor
asks of a knowledge-soundness theorem, after Ironwood's `KnowledgeContract`
(zcash/ironwood `86e3c7026db8`, `book/src/formal-verification/knowledge-contract.md`).

It does not imply completeness: an acceptance predicate that holds nowhere
satisfies every field.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec

open scoped BigOperators

attribute [local instance low] Classical.propDecidable

structure KnowledgeContract where
  Run : Type
  [runFinite : Fintype Run]
  weight : Run → ℝ
  weight_nonnegative : ∀ run, 0 ≤ weight run
  weight_sum : ∑ run, weight run = 1
  Statement : Type
  statement : Run → Statement
  accepts : Run → Prop
  Witness : Type
  extract : Run → Option Witness
  Holds : Statement → Witness → Prop
  witness_statement : ∀ run witness, extract run = some witness → Holds (statement run) witness
  error : ℝ
  knowledge_sound :
    ∑ run, weight run * (if accepts run ∧ extract run = none then 1 else 0) ≤ error

end NightstreamFPrime.Spec
