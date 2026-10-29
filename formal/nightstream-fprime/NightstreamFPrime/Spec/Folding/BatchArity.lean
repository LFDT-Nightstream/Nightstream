import NightstreamFPrime.Spec.Relation

namespace NightstreamFPrime.Spec.Folding

inductive RunningMode where
  | bootstrap
  | active
deriving Repr, DecidableEq

namespace RunningMode

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

structure BatchArity (params : GlobalParams) where
  freshCount : Nat
  freshPositive : 0 < freshCount
  /-- SuperNeo v1.2 Definition 22. -/
  freshBound : freshCount ≤ params.maxFresh
  mode : RunningMode

namespace BatchArity

def total {params : GlobalParams} (arity : BatchArity params) : Nat :=
  arity.freshCount + arity.mode.count params

theorem totalPositive {params : GlobalParams} (arity : BatchArity params) :
    0 < arity.total := by
  exact Nat.lt_of_lt_of_le arity.freshPositive (Nat.le_add_right arity.freshCount _)

theorem total_le {params : GlobalParams} (arity : BatchArity params) :
    arity.total ≤ params.maxFresh + params.k := by
  exact Nat.add_le_add arity.freshBound (arity.mode.count_le params)

def bootstrap
    (params : GlobalParams)
    (freshCount : Nat)
    (freshPositive : 0 < freshCount)
    (freshBound : freshCount ≤ params.maxFresh) : BatchArity params where
  freshCount := freshCount
  freshPositive := freshPositive
  freshBound := freshBound
  mode := .bootstrap

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

end NightstreamFPrime.Spec.Folding
