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
  | .squeezeFirst _ | .squeezeSecond => Spec.Poseidon2.permute state

def ExpectedAt (env : Env) (state : State) :
    PoseidonActionSchedule.Kind → Prop
  | .squeezeFirst expected =>
      expected.eval env = Squeeze.referenceSample state
  | .absorb _ | .squeezeSecond => True

def runKinds (env : Env) : State → List PoseidonActionSchedule.Kind → State
  | state, [] => state
  | state, kind :: kinds => runKinds env (runKind env state kind) kinds

def KindsHold (env : Env) : State → List PoseidonActionSchedule.Kind → Prop
  | _state, [] => True
  | state, kind :: kinds =>
      ExpectedAt env state kind ∧
        KindsHold env (runKind env state kind) kinds

@[simp] theorem runKinds_append (env : Env) (state : State)
    (left right : List PoseidonActionSchedule.Kind) :
    runKinds env state (left ++ right) =
      runKinds env (runKinds env state left) right := by
  induction left generalizing state with
  | nil => rfl
  | cons kind kinds inductionHypothesis =>
      simp only [List.cons_append, runKinds]
      exact inductionHypothesis _

theorem kindsHold_append_iff (env : Env) (state : State)
    (left right : List PoseidonActionSchedule.Kind) :
    KindsHold env state (left ++ right) ↔
      KindsHold env state left ∧
        KindsHold env (runKinds env state left) right := by
  induction left generalizing state with
  | nil => simp [KindsHold, runKinds]
  | cons kind kinds inductionHypothesis =>
      simp only [List.cons_append, KindsHold, runKinds]
      rw [inductionHypothesis]
      tauto

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

theorem actionKinds_traceHolds (env : Env) (state : State)
    (action : Formal.Action)
    (holds : KindsHold env state
      (PoseidonActionSchedule.actionKinds action)) :
    Formal.TraceHolds state [action.eval env]
      (runKinds env state (PoseidonActionSchedule.actionKinds action)) := by
  cases action with
  | absorb input =>
      simp only [Formal.Action.eval, Formal.TraceHolds]
      unfold PoseidonActionSchedule.actionKinds
      rw [runKinds_absorbBlocks]
      unfold Absorb.reference
      rw [Hash.inputChunks_eval]
  | squeezeK expected =>
      simpa [PoseidonActionSchedule.actionKinds, KindsHold, runKinds, runKind,
        ExpectedAt, Formal.Action.eval, Formal.TraceHolds,
        Squeeze.referenceState] using holds

theorem traceHolds_append {initial middle final : State}
    {left right : List Formal.ValueAction}
    (leftHolds : Formal.TraceHolds initial left middle)
    (rightHolds : Formal.TraceHolds middle right final) :
    Formal.TraceHolds initial (left ++ right) final := by
  induction left generalizing initial with
  | nil =>
      simp only [Formal.TraceHolds] at leftHolds
      subst middle
      exact rightHolds
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          simp only [Formal.TraceHolds] at leftHolds ⊢
          exact inductionHypothesis leftHolds
      | squeezeK expected =>
          simp only [Formal.TraceHolds] at leftHolds ⊢
          exact ⟨leftHolds.1,
            inductionHypothesis leftHolds.2⟩

theorem kinds_traceHolds (env : Env) (state : State)
    (actions : List Formal.Action)
    (holds : KindsHold env state
      (PoseidonActionSchedule.kinds actions)) :
    Formal.TraceHolds state (actions.map (Formal.Action.eval env))
      (runKinds env state (PoseidonActionSchedule.kinds actions)) := by
  induction actions generalizing state with
  | nil => rfl
  | cons action actions inductionHypothesis =>
      have split := (kindsHold_append_iff env state
        (PoseidonActionSchedule.actionKinds action)
        (PoseidonActionSchedule.kinds actions)).mp (by
          simpa [PoseidonActionSchedule.kinds] using holds)
      have head := actionKinds_traceHolds env state action split.1
      have tail := inductionHypothesis
        (runKinds env state (PoseidonActionSchedule.actionKinds action))
        split.2
      have combined := traceHolds_append head tail
      simpa [PoseidonActionSchedule.kinds, runKinds_append] using combined

structure IndexedSemantics (env : Env) {count : Nat} (initial : State)
    (kindAt : Fin count → PoseidonActionSchedule.Kind)
    (output : Fin count → State) : Prop where
  step : ∀ current,
    output current = runKind env
      (previousState initial output current) (kindAt current)
  expected : ∀ current expected,
    kindAt current = .squeezeFirst expected →
      expected.eval env =
        Squeeze.referenceSample (previousState initial output current)

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
  expected current expected found := by
    rw [previousState_tail initial output current]
    exact semantics.expected current.succ expected found

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
  expected current expected found := by
    rw [previousState_slice initial output offset count fits offsetBound current]
    exact semantics.expected (sliceIndex offset count fits current) expected found

theorem indexed_holds_and_final (count : Nat) (env : Env) (initial : State)
    (kindAt : Fin count → PoseidonActionSchedule.Kind)
    (output : Fin count → State)
    (semantics : IndexedSemantics env initial kindAt output) :
    KindsHold env initial (List.ofFn kindAt) ∧
      runKinds env initial (List.ofFn kindAt) =
        match count with
        | 0 => initial
        | count + 1 => output ⟨count, Nat.lt_succ_self count⟩ := by
  induction count generalizing initial with
  | zero => exact ⟨by trivial, rfl⟩
  | succ count inductionHypothesis =>
      let tailKind : Fin count → PoseidonActionSchedule.Kind :=
        fun index => kindAt index.succ
      let tailOutput : Fin count → State := fun index => output index.succ
      have tailSemantics : IndexedSemantics env (output 0) tailKind tailOutput :=
        semantics.tail
      have tailResult := inductionHypothesis (output 0) tailKind tailOutput
        tailSemantics
      have headStep : output 0 = runKind env initial (kindAt 0) := by
        simpa using semantics.step 0
      have headExpected : ExpectedAt env initial (kindAt 0) := by
        cases found : kindAt 0 with
        | absorb block => trivial
        | squeezeFirst expected =>
            exact semantics.expected 0 expected found
        | squeezeSecond => trivial
      constructor
      · rw [List.ofFn_succ]
        change ExpectedAt env initial (kindAt 0) ∧
          KindsHold env (runKind env initial (kindAt 0))
            (List.ofFn tailKind)
        rw [← headStep]
        exact ⟨headExpected, tailResult.1⟩
      · rw [List.ofFn_succ]
        change runKinds env (runKind env initial (kindAt 0))
          (List.ofFn tailKind) = _
        rw [← headStep, tailResult.2]
        cases count with
        | zero => rfl
        | succ count => rfl

theorem indexed_traceHolds (count : Nat) (env : Env) (initial : State)
    (kindAt : Fin count → PoseidonActionSchedule.Kind)
    (output : Fin count → State) (actions : List Formal.Action)
    (materializes : List.ofFn kindAt = PoseidonActionSchedule.kinds actions)
    (semantics : IndexedSemantics env initial kindAt output) :
    Formal.TraceHolds initial (actions.map (Formal.Action.eval env))
      (match count with
       | 0 => initial
       | count + 1 => output ⟨count, Nat.lt_succ_self count⟩) := by
  have indexed := indexed_holds_and_final count env initial kindAt output
    semantics
  have holds : KindsHold env initial
      (PoseidonActionSchedule.kinds actions) := by
    rw [← materializes]
    exact indexed.1
  have trace := kinds_traceHolds env initial actions holds
  rw [← materializes, indexed.2] at trace
  cases count <;> simpa using trace

end NightstreamFPrime.Export.Stage1.PoseidonActionSemantics
