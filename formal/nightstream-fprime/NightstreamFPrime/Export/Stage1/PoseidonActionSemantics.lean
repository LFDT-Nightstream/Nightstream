import NightstreamFPrime.Export.Stage1.InvocationLastOutput
import NightstreamFPrime.Export.Stage1.PoseidonActionSchedule

/-!
Owns the structural value-semantics bridge from an indexed Poseidon2 action
schedule to the authoritative Duplex trace. The proof is generic in the
action list and does not materialize a production schedule.

This module does not select a concrete phase or package assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PoseidonActionSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Spec

abbrev State := Spec.Poseidon2.State

def previousState {count : Nat} (initial : State)
    (output : Fin count → State) (current : Fin count) : State :=
  if first : current.val = 0 then
    initial
  else
    output ⟨current.val - 1, by
      have currentBound := current.isLt
      omega⟩

def runKind (env : Env) (state : State) : PoseidonActionSchedule.Kind → State
  | .absorb block =>
      Spec.Poseidon2.absorbBlock state (Hash.evalList env block)

def runKinds (env : Env) : State → List PoseidonActionSchedule.Kind → State
  | state, [] => state
  | state, kind :: kinds => runKinds env (runKind env state kind) kinds

theorem runKinds_absorbBlocks (env : Env) (state : State)
    (blocks : List (List Expr)) :
    runKinds env state (blocks.map PoseidonActionSchedule.Kind.absorb) =
      (blocks.map (Hash.evalList env)).foldl
        Spec.Poseidon2.absorbBlock state := by
  induction blocks generalizing state with
  | nil => rfl
  | cons block blocks inductionHypothesis =>
      simp only [List.map_cons, runKinds, runKind, List.foldl_cons]
      exact inductionHypothesis _

/-- Each read sees the value state after the invocations that precede it in
the same action list. -/
def ReadsAt (env : Env) (stateAt : Nat → State) :
    Nat → List Formal.Action → Prop
  | _, [] => True
  | index, .absorb input :: actions =>
      ReadsAt env stateAt (index + (Hash.inputChunks input).length) actions
  | index, .readK pair expected :: actions =>
      expected.eval env = Read.referenceSample (stateAt index) pair ∧
        ReadsAt env stateAt index actions

/-- The value state after `index` invocations of an indexed schedule. -/
def stateAfter {count : Nat} (initial : State) (output : Fin count → State)
    (index : Nat) : State :=
  if index = 0 then
    initial
  else if bound : index - 1 < count then
    output ⟨index - 1, bound⟩
  else
    initial

/-- Default kind for a total list lookup. Every lookup below is bounded. -/
private def noKind : PoseidonActionSchedule.Kind := .absorb []

private theorem runKinds_of_steps (env : Env) (stateAt : Nat → State) :
    ∀ (kinds : List PoseidonActionSchedule.Kind) (offset : Nat),
      (∀ index, index < kinds.length →
        stateAt (offset + index + 1) =
          runKind env (stateAt (offset + index)) (kinds.getD index noKind)) →
      runKinds env (stateAt offset) kinds = stateAt (offset + kinds.length) := by
  intro kinds
  induction kinds with
  | nil => intro offset _; rfl
  | cons kind kinds inductionHypothesis =>
      intro offset steps
      have head := steps 0 (by simp)
      simp only [Nat.add_zero, List.getD_cons_zero] at head
      simp only [runKinds]
      rw [← head]
      have tail := inductionHypothesis (offset + 1) (by
        intro index bound
        have step := steps (index + 1) (by simpa using bound)
        simp only [List.getD_cons_succ] at step
        simpa [Nat.add_assoc, Nat.add_comm 1 index] using step)
      rw [tail, List.length_cons]
      congr 1
      omega

private theorem traceHolds_of_steps (env : Env) (stateAt : Nat → State) :
    ∀ (actions : List Formal.Action) (offset : Nat),
      (∀ index, index < (PoseidonActionSchedule.kinds actions).length →
        stateAt (offset + index + 1) =
          runKind env (stateAt (offset + index))
            ((PoseidonActionSchedule.kinds actions).getD index noKind)) →
      ReadsAt env stateAt offset actions →
      Formal.TraceHolds (stateAt offset)
        (actions.map (Formal.Action.eval env))
        (stateAt (offset + (PoseidonActionSchedule.kinds actions).length)) := by
  intro actions
  induction actions with
  | nil => intro offset _ _; rfl
  | cons action actions inductionHypothesis =>
      intro offset steps reads
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          have kindsEq : PoseidonActionSchedule.kinds (.absorb input :: actions) =
              blocks.map PoseidonActionSchedule.Kind.absorb ++
                PoseidonActionSchedule.kinds actions := rfl
          have absorbed : runKinds env (stateAt offset)
              (blocks.map PoseidonActionSchedule.Kind.absorb) =
                stateAt (offset + blocks.length) := by
            have run := runKinds_of_steps env stateAt
              (blocks.map PoseidonActionSchedule.Kind.absorb) offset (by
                intro index bound
                have step := steps index (by
                  rw [kindsEq, List.length_append]
                  omega)
                rw [kindsEq, List.getD_append _ _ _ _ bound] at step
                exact step)
            simpa using run
          have reference : Absorb.reference (stateAt offset)
              (Hash.evalList env input) = stateAt (offset + blocks.length) := by
            rw [← absorbed, runKinds_absorbBlocks]
            unfold Absorb.reference
            rw [Hash.inputChunks_eval]
          have tail := inductionHypothesis (offset + blocks.length) (by
              intro index bound
              have step := steps (blocks.length + index) (by
                rw [kindsEq, List.length_append, List.length_map]
                omega)
              rw [kindsEq, List.getD_append_right _ _ _ _ (by simp)] at step
              simpa [Nat.add_assoc] using step)
            reads
          simp only [List.map_cons, Formal.Action.eval, Formal.TraceHolds]
          rw [reference, kindsEq, List.length_append, List.length_map,
            ← Nat.add_assoc]
          exact tail
      | readK pair expected =>
          have tail := inductionHypothesis offset steps reads.2
          simp only [List.map_cons, Formal.Action.eval, Formal.TraceHolds]
          exact ⟨reads.1, tail⟩

structure IndexedSemantics (env : Env) {count : Nat} (initial : State)
    (kindAt : Fin count → PoseidonActionSchedule.Kind)
    (output : Fin count → State) : Prop where
  step : ∀ current,
    output current = runKind env
      (previousState initial output current) (kindAt current)

@[simp] theorem previousState_zero (initial : State)
    {count : Nat} (output : Fin (count + 1) → State) :
    previousState initial output 0 = initial := by
  simp [previousState]

theorem previousState_tail (initial : State) {count : Nat}
    (output : Fin (count + 1) → State) (current : Fin count) :
    previousState (output 0) (fun index => output index.succ) current =
      previousState initial output current.succ := by
  have left : previousState (output 0)
      (fun index => output index.succ) current = output current.castSucc := by
    unfold previousState
    by_cases first : current.val = 0
    · rw [dif_pos first]
      apply congrArg output
      apply Fin.ext
      simpa using first.symm
    · rw [dif_neg first]
      apply congrArg output
      apply Fin.ext
      simp only [Fin.val_succ, Fin.val_castSucc]
      omega
  have right : previousState initial output current.succ =
      output current.castSucc := by
    unfold previousState
    rw [dif_neg (by simp)]
    apply congrArg output
    apply Fin.ext
    simp
  exact left.trans right.symm

def IndexedSemantics.tail {env : Env} {count : Nat} {initial : State}
    {kindAt : Fin (count + 1) → PoseidonActionSchedule.Kind}
    {output : Fin (count + 1) → State}
    (semantics : IndexedSemantics env initial kindAt output) :
    IndexedSemantics env (output 0) (fun index => kindAt index.succ)
      (fun index => output index.succ) where
  step current := by
    rw [previousState_tail initial output current]
    exact semantics.step current.succ

def sliceIndex {total : Nat} (offset count : Nat)
    (fits : offset + count ≤ total) (current : Fin count) : Fin total :=
  ⟨offset + current.val, by
    have currentBound := current.isLt
    omega⟩

def sliceOutput {total : Nat} (output : Fin total → State)
    (offset count : Nat) (fits : offset + count ≤ total) : Fin count → State :=
  fun current => output (sliceIndex offset count fits current)

def sliceInitial {total : Nat} (initial : State) (output : Fin total → State)
    (offset : Nat) (offsetBound : offset < total) : State :=
  if first : offset = 0 then
    initial
  else
    output ⟨offset - 1, by omega⟩

theorem previousState_slice {total : Nat} (initial : State)
    (output : Fin total → State) (offset count : Nat)
    (fits : offset + count ≤ total) (offsetBound : offset < total)
    (current : Fin count) :
    previousState (sliceInitial initial output offset offsetBound)
        (sliceOutput output offset count fits) current =
      previousState initial output (sliceIndex offset count fits current) := by
  unfold previousState
  by_cases currentFirst : current.val = 0
  · rw [dif_pos currentFirst]
    by_cases offsetFirst : offset = 0
    · rw [sliceInitial, dif_pos offsetFirst]
      rw [dif_pos (by simp [sliceIndex, currentFirst, offsetFirst])]
    · rw [sliceInitial, dif_neg offsetFirst]
      rw [dif_neg (by simp [sliceIndex, currentFirst]; omega)]
      apply congrArg output
      apply Fin.ext
      simp only [sliceIndex]
      omega
  · rw [dif_neg currentFirst]
    rw [dif_neg (by simp [sliceIndex]; omega)]
    apply congrArg output
    apply Fin.ext
    simp only [sliceIndex]
    omega

def IndexedSemantics.slice {env : Env} {total : Nat} {initial : State}
    {kindAt : Fin total → PoseidonActionSchedule.Kind}
    {output : Fin total → State}
    (semantics : IndexedSemantics env initial kindAt output)
    (offset count : Nat) (fits : offset + count ≤ total)
    (offsetBound : offset < total) :
    IndexedSemantics env (sliceInitial initial output offset offsetBound)
      (fun current => kindAt (sliceIndex offset count fits current))
      (sliceOutput output offset count fits) where
  step current := by
    rw [previousState_slice initial output offset count fits offsetBound current]
    exact semantics.step (sliceIndex offset count fits current)

/-- Held invocations and reads of one indexed schedule imply the exact
Duplex trace of its action list. -/
theorem indexed_traceHolds (count : Nat) (env : Env) (initial : State)
    (kindAt : Fin count → PoseidonActionSchedule.Kind)
    (output : Fin count → State) (actions : List Formal.Action)
    (materializes : List.ofFn kindAt = PoseidonActionSchedule.kinds actions)
    (semantics : IndexedSemantics env initial kindAt output)
    (reads : ReadsAt env (stateAfter initial output) 0 actions) :
    Formal.TraceHolds initial (actions.map (Formal.Action.eval env))
      (stateAfter initial output count) := by
  have lengthEq : (PoseidonActionSchedule.kinds actions).length = count := by
    rw [← materializes, List.length_ofFn]
  have trace := traceHolds_of_steps env (stateAfter initial output) actions 0
    (by
      intro index bound
      have indexBound : index < count := by omega
      have step := semantics.step ⟨index, indexBound⟩
      have kindEq : (PoseidonActionSchedule.kinds actions).getD index noKind =
          kindAt ⟨index, indexBound⟩ := by
        rw [← materializes, List.getD_eq_getElem _ _ (by simpa using indexBound),
          List.getElem_ofFn]
      rw [kindEq]
      have previousEq : previousState initial output ⟨index, indexBound⟩ =
          stateAfter initial output (0 + index) := by
        unfold previousState stateAfter
        by_cases first : index = 0
        · simp [first]
        · rw [dif_neg first, Nat.zero_add, if_neg first,
            dif_pos (by omega)]
      have outputEq : output ⟨index, indexBound⟩ =
          stateAfter initial output (0 + index + 1) := by
        unfold stateAfter
        rw [if_neg (by omega), dif_pos (by omega)]
        apply congrArg output
        apply Fin.ext
        simp
      rw [← previousEq, ← outputEq]
      exact step)
    reads
  rw [lengthEq, Nat.zero_add] at trace
  simpa [stateAfter] using trace

/-- Reads are sound when every symbolic state of the compiled action list
evaluates to the indexed value state at the same position. -/
theorem readsAt_of_outputs (env : Env) (stateAt : Nat → State) :
    ∀ (actions : List Formal.Action) (start offset : Nat)
      (symbolic : Layer.EState),
      List.ofFn (Layer.evalState env symbolic) = stateAt offset →
      (∀ index, index < Invocations.invocationCount actions →
        List.ofFn (Layer.evalState env
          (Invocations.permutationOutput (start + index * 1096))) =
            stateAt (offset + index + 1)) →
      Formal.expectedSamples actions =
        (Formal.compile start symbolic actions).samples →
      ReadsAt env stateAt offset actions := by
  intro actions
  induction actions with
  | nil => intro _ _ _ _ _ _; trivial
  | cons action actions inductionHypothesis =>
      intro start offset symbolic initialEq outputsEq samples
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let absorbed := Hash.compileAbsorptions start symbolic blocks
          have countEq : Invocations.invocationCount (.absorb input :: actions) =
              blocks.length + Invocations.invocationCount actions := by
            simp [Invocations.invocationCount, Invocations.Action.invocationCount,
              blocks]
          have nextEq : List.ofFn (Layer.evalState env absorbed.output) =
              stateAt (offset + blocks.length) := by
            by_cases empty : blocks = []
            · have outputEq : absorbed.output = symbolic := by
                simp only [absorbed, empty]
                rfl
              rw [outputEq, initialEq, empty]
              rfl
            · have last := InvocationLastOutput.compileBlocks_state_last 0 0 start
                symbolic blocks empty
              rw [Invocations.compileBlocks_state_eq] at last
              have positive : 0 < blocks.length := List.length_pos_of_ne_nil empty
              rw [show absorbed.output = _ from last]
              have output := outputsEq (blocks.length - 1) (by omega)
              rw [output]
              congr 1
              omega
          have tailSamples : Formal.expectedSamples actions =
              (Formal.compile (start + absorbed.recipes.length) absorbed.output
                actions).samples := by
            simpa [Formal.expectedSamples, Formal.compile, absorbed, blocks]
              using samples
          rw [Hash.compileAbsorptions_recipes_length] at tailSamples
          change ReadsAt env stateAt (offset + blocks.length) actions
          exact inductionHypothesis (start + blocks.length * 1096)
            (offset + blocks.length) absorbed.output nextEq (by
              intro index bound
              have output := outputsEq (blocks.length + index) (by omega)
              have startEq : start + (blocks.length + index) * 1096 =
                  start + blocks.length * 1096 + index * 1096 := by
                rw [Nat.add_mul, Nat.add_assoc]
              rw [startEq] at output
              rw [output]
              congr 1
              omega) tailSamples
      | readK pair expected =>
          have parts : expected = Read.sample symbolic pair ∧
              Formal.expectedSamples actions =
                (Formal.compile start symbolic actions).samples := by
            simpa [Formal.expectedSamples, Formal.compile]
              using List.cons.inj samples
          refine ⟨?_, inductionHypothesis start offset symbolic initialEq
            (by
              intro index bound
              exact outputsEq index (by
                simpa [Invocations.invocationCount,
                  Invocations.Action.invocationCount] using bound))
            parts.2⟩
          rw [parts.1, Read.sample_eval, initialEq]

end NightstreamFPrime.Export.Stage1.PoseidonActionSemantics
