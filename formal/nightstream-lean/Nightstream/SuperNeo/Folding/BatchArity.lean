import Nightstream.SuperNeo.Relations

/-!
Production arities for one SuperNeo multi-fold batch.

The first recursive step has no running accumulator, while later recursive and
terminal folds carry the full `k`-wide running product.  Keeping that choice in
the type prevents the bootstrap artifact from being modeled as if it consumed
`k` synthetic CE statements.
-/

namespace Nightstream.SuperNeo.Folding

/-- Whether this fold starts without a running accumulator or consumes all `k`
running CE statements.  Production has no intermediate running cardinality. -/
inductive RunningMode where
  | bootstrap
  | active
deriving Repr, DecidableEq

namespace RunningMode

/-- Number of running CE inputs selected by a production fold mode. -/
def count (mode : RunningMode) (params : GlobalParams) : Nat :=
  match mode with
  | .bootstrap => 0
  | .active => params.k

theorem count_le (mode : RunningMode) (params : GlobalParams) :
    mode.count params ≤ params.k := by
  cases mode <;> simp [count]

@[simp] theorem bootstrap_count (params : GlobalParams) :
    RunningMode.bootstrap.count params = 0 := rfl

@[simp] theorem active_count (params : GlobalParams) :
    RunningMode.active.count params = params.k := rfl

end RunningMode

/-- Verifier-owned shape of one production multi-fold batch. -/
structure BatchArity (params : GlobalParams) where
  freshCount : Nat
  /-- Rust rejects an empty fresh batch. -/
  freshPositive : 0 < freshCount
  /-- Definition 14 is instantiated below the deployment maximum `K`. -/
  freshBound : freshCount ≤ params.maxFresh
  mode : RunningMode

namespace BatchArity

/-- Total number of PiCCS outputs and PiRLC inputs in this batch. -/
def total {params : GlobalParams} (arity : BatchArity params) : Nat :=
  arity.freshCount + arity.mode.count params

theorem totalPositive {params : GlobalParams} (arity : BatchArity params) :
    0 < arity.total := by
  exact Nat.lt_of_lt_of_le arity.freshPositive (Nat.le_add_right arity.freshCount _)

/-- Every production batch is covered by Definition 14's maximum-arity bound. -/
theorem total_le {params : GlobalParams} (arity : BatchArity params) :
    arity.total ≤ params.maxFresh + params.k := by
  exact Nat.add_le_add arity.freshBound (arity.mode.count_le params)

/-- First-step arity: positive fresh inputs and no synthetic running claims. -/
def bootstrap
    (params : GlobalParams)
    (freshCount : Nat)
    (freshPositive : 0 < freshCount)
    (freshBound : freshCount ≤ params.maxFresh) : BatchArity params where
  freshCount := freshCount
  freshPositive := freshPositive
  freshBound := freshBound
  mode := .bootstrap

/-- Steady-state/paper arity: positive fresh inputs plus the full `k` product. -/
def active
    (params : GlobalParams)
    (freshCount : Nat)
    (freshPositive : 0 < freshCount)
    (freshBound : freshCount ≤ params.maxFresh) : BatchArity params where
  freshCount := freshCount
  freshPositive := freshPositive
  freshBound := freshBound
  mode := .active

@[simp] theorem bootstrap_total
    (params : GlobalParams)
    (freshCount : Nat)
    (freshPositive : 0 < freshCount)
    (freshBound : freshCount ≤ params.maxFresh) :
    (bootstrap params freshCount freshPositive freshBound).total = freshCount := by
  simp [total, bootstrap]

@[simp] theorem active_total
    (params : GlobalParams)
    (freshCount : Nat)
    (freshPositive : 0 < freshCount)
    (freshBound : freshCount ≤ params.maxFresh) :
    (active params freshCount freshPositive freshBound).total = freshCount + params.k := by
  simp [total, active]

end BatchArity

end Nightstream.SuperNeo.Folding
