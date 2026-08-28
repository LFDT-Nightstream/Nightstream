import NightstreamFPrime.Layout.ProductionRelation.PoseidonStepTrace

/-!
Owns the compact fixed-schedule ledger for one production Poseidon2 template.
The template has eight caller inputs, 592 local source columns, 31 schedule
steps, and 334 direct selective rows.

This module is constant-size. Package invocation count is not evaluated here.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonScheduleTrace

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

abbrev Step := Permutation.Step
abbrev EState := Permutation.EState

def inputCount : Nat := 8
def localColumnCount : Nat := 592
def sourceColumnCount : Nat := inputCount + localColumnCount

def canonicalState : EState := fun lane => Expr.var lane.val

/-- One step and the exact source-column state entering it. -/
structure Record where
  start : Nat
  step : Step
  state : EState

/-- Structural scan of the fixed step schedule. -/
def recordsFrom : Nat → EState → List Step → List Record
  | _, _, [] => []
  | start, state, step :: rest =>
      { start := start, step := step, state := state } ::
        recordsFrom (start + Permutation.stepSize step)
          (Permutation.stepOutput start step) rest

def records : List Record :=
  recordsFrom inputCount canonicalState Permutation.schedule

def finalStart : Nat → List Step → Nat
  | start, [] => start
  | start, step :: rest =>
      finalStart (start + Permutation.stepSize step) rest

@[simp] theorem sourceColumnCount_eq : sourceColumnCount = 600 := by
  rfl

@[simp] theorem records_length : records.length = 31 := by
  rfl

@[simp] theorem finalStart_eq :
    finalStart inputCount Permutation.schedule = sourceColumnCount := by
  rfl

/-- Every invocation replaces the 592 source recipe rows by exactly 334
direct selective rows. -/
@[simp] theorem directRowCount_eq :
    (records.map fun record =>
      PoseidonStepTrace.directRowCount record.step).sum = 334 := by
  rfl

@[simp] theorem removedRowCount_eq : localColumnCount - 334 = 258 := by
  rfl

/-- Exact composition of one scanned schedule. Only the retained boundary
equation of each record is used. -/
theorem recordsFrom_imply_runF (env : Env) (start : Nat) (state : EState)
    (steps : List Step)
    (holds : ∀ record ∈ recordsFrom start state steps,
      Layer.evalState env
          (Permutation.stepOutput record.start record.step) =
        Permutation.applyF record.step
          (Layer.evalState env record.state)) :
    Layer.evalState env (Permutation.compile start state steps).output =
      Permutation.runF steps (Layer.evalState env state) := by
  induction steps generalizing start state with
  | nil => rfl
  | cons step rest induction =>
      have head := holds
        { start := start, step := step, state := state }
        (by simp [recordsFrom])
      have tail := induction
        (start := start + Permutation.stepSize step)
        (state := Permutation.stepOutput start step)
        (fun record member => holds record (by simp [recordsFrom, member]))
      simpa [Permutation.compile, Permutation.runF, head] using tail

/-- The retained boundary equations of all 31 canonical records imply the
exact reference Poseidon2 permutation. -/
theorem records_imply_permute (env : Env)
    (holds : ∀ record ∈ records,
      Layer.evalState env
          (Permutation.stepOutput record.start record.step) =
        Permutation.applyF record.step
          (Layer.evalState env record.state)) :
    List.ofFn (Layer.evalState env
        (Permutation.scheduleOutput inputCount)) =
      Spec.Poseidon2.permute
        (List.ofFn (Layer.evalState env canonicalState)) := by
  have run := recordsFrom_imply_runF env inputCount canonicalState
    Permutation.schedule holds
  calc
    List.ofFn (Layer.evalState env
        (Permutation.scheduleOutput inputCount)) =
        List.ofFn (Layer.evalState env
          (Permutation.compile inputCount canonicalState
            Permutation.schedule).output) := by
      rw [Permutation.scheduleOutput_eq_compile]
    _ = List.ofFn (Permutation.runF Permutation.schedule
          (Layer.evalState env canonicalState)) := congrArg List.ofFn run
    _ = Permutation.runReference Permutation.schedule
          (List.ofFn (Layer.evalState env canonicalState)) :=
      Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute
          (List.ofFn (Layer.evalState env canonicalState)) :=
      Permutation.runReference_schedule _

end NightstreamFPrime.Layout.ProductionRelation.PoseidonScheduleTrace
