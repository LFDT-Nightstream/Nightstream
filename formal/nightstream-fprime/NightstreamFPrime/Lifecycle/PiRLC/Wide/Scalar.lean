import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption
import NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript

/-! Candidate scalar lifecycle: the existing domain entry, one checked
wide reduction of its four rate lanes, and one digest advance. The child
interfaces are opaque. This module does not select the production sampler. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.Scalar

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Gadgets.Poseidon2

abbrev referenceEnter := Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter
abbrev referenceNext := Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.next
abbrev referenceBlock := Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.block
abbrev EState := Layer.EState
abbrev Interface := v1_1.TranscriptAbsorption.Interface

def rangeOffset (offset : Nat) : Nat := offset + 592
def advanceOffset (offset : Nat) : Nat := rangeOffset offset + WideReduction.Program.privateCount
def privateCount : Nat := 592 + WideReduction.Program.privateCount + 592
def logicalRowCount : Nat := 592 + WideReduction.rowCount + 592

def entry (interface : Interface) (coordinate : Nat) : FormalCircuit :=
  (v1_1.TranscriptAbsorption.circuit interface coordinate).withConstantFootprint 592 592
    (v1_1.TranscriptAbsorption.localLength_eq interface coordinate)
    (v1_1.TranscriptAbsorption.flatConstraints_length interface coordinate)

def enteredState (interface : Interface) (coordinate : Nat) (offset : Nat) : EState :=
  v1_1.TranscriptAbsorption.output interface coordinate offset

def rateLane (lane : Fin 4) : Fin 8 := ⟨lane.val, lt_trans lane.isLt (by decide)⟩

def rangeInterface (interface : Interface) (coordinate : Nat) (offset : Nat) : WideReduction.Interface where
  source := fun lane _ => enteredState interface coordinate offset (rateLane lane)

def advanceInterface (interface : Interface) (coordinate : Nat) (offset : Nat) :
    Permutation.Owned.Interface where
  initialState := fun _ => enteredState interface coordinate offset

def rangeCircuit (interface : Interface) (coordinate : Nat) (offset : Nat) : FormalCircuit :=
  WideReduction.Program.circuit (rangeInterface interface coordinate offset)

def advance (interface : Interface) (coordinate : Nat) (offset : Nat) : FormalCircuit :=
  Permutation.Owned.circuit (advanceInterface interface coordinate offset)

def entryOp (interface : Interface) (coordinate : Nat) (offset : Nat) : Op :=
  Sequence.childOp "pirlc.wide.enter_scalar" (entry interface coordinate) offset

def rangeOp (interface : Interface) (coordinate : Nat) (offset : Nat) : Op :=
  Sequence.childOp "pirlc.wide.reduce" (rangeCircuit interface coordinate offset) (rangeOffset offset)

def advanceOp (interface : Interface) (coordinate : Nat) (offset : Nat) : Op :=
  Sequence.childOp "pirlc.wide.advance" (advance interface coordinate offset) (advanceOffset offset)

def operations (interface : Interface) (coordinate : Nat) (offset : Nat) : List Op :=
  [entryOp interface coordinate offset, rangeOp interface coordinate offset,
    advanceOp interface coordinate offset]

def outputState (interface : Interface) (coordinate : Nat) (offset : Nat) : EState :=
  Permutation.Owned.output (advanceInterface interface coordinate offset) (advanceOffset offset)

def outputChallenge (offset : Nat) (position : Fin ringDegree) : Expr :=
  WideReduction.Program.outputChallenge (rangeOffset offset) position

def Assumptions (interface : Interface) (offset : Nat) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

def evalState (env : Env) (state : EState) : Poseidon2.State := List.ofFn (Layer.evalState env state)

structure SpecHolds (interface : Interface) (coordinate : Nat) (offset : Nat) (env : Env) : Prop where
  state : evalState env (outputState interface coordinate offset) =
    referenceNext (evalState env (interface.initialState offset)) coordinate
  digits : ∀ position : Fin ringDegree,
    WideReduction.digitValue env (WideReduction.Program.coreOffset (rangeOffset offset)) position.val =
      (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.sample
        (referenceBlock (referenceEnter (evalState env (interface.initialState offset)) coordinate)) position).val

theorem counts : privateCount = 3205 ∧ logicalRowCount = 1865 := ⟨rfl, rfl⟩

theorem localLength_eq (interface : Interface) (coordinate : Nat) (offset : Nat) :
    localLength (operations interface coordinate offset) = privateCount := by
  change (entryOp interface coordinate offset).localLength +
    ((rangeOp interface coordinate offset).localLength +
      ((advanceOp interface coordinate offset).localLength + 0)) = _
  simp only [entryOp, rangeOp, advanceOp, Sequence.childOp, Op.localLength,
    FormalCircuit.asSubcircuit_localLength]
  rw [entry, FormalCircuit.withConstantFootprint_main, v1_1.TranscriptAbsorption.localLength_eq]
  change 592 + (localLength (WideReduction.Program.operations _ _) +
    (localLength (Permutation.Owned.operations _ _) + 0)) = privateCount
  rw [WideReduction.Program.localLength_eq, Permutation.Owned.localLength_eq]
  unfold privateCount
  omega

theorem rowCount_eq (interface : Interface) (coordinate : Nat) (offset : Nat) :
    (flatConstraints (operations interface coordinate offset)).length = logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  simp only [operations, Circuit.rowCount, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    entryOp, rangeOp, advanceOp, Sequence.childOp, Op.rowCount,
    FormalCircuit.asSubcircuit_rowCount]
  rw [entry, FormalCircuit.withConstantFootprint_rowCount]
  change 592 + (WideReduction.rowCount + (592 + 0)) = logicalRowCount
  unfold logicalRowCount
  omega

theorem enteredState_below (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ lane, (enteredState interface coordinate offset lane).VarsBelow (rangeOffset offset) := by
  have scope := v1_1.TranscriptAbsorption.output_varsBelow interface coordinate offset (fun _ => 0) inputs
  rw [v1_1.TranscriptAbsorption.localLength_eq] at scope
  exact scope

theorem range_inputs (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    WideReduction.Assumptions (rangeInterface interface coordinate offset) (rangeOffset offset) :=
  fun lane => enteredState_below interface coordinate offset inputs (rateLane lane)

theorem advance_inputs (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) (env : Env) :
    Permutation.Owned.Assumptions (advanceInterface interface coordinate offset) (advanceOffset offset) env := by
  intro lane
  exact Expr.VarsBelow.mono _ (enteredState_below interface coordinate offset inputs lane)
    (by unfold advanceOffset; omega)

theorem outputState_below (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ lane, (outputState interface coordinate offset lane).VarsBelow (offset + privateCount) := by
  have scope := Permutation.Owned.output_varsBelow (advanceInterface interface coordinate offset)
    (advanceOffset offset) (advance_inputs interface coordinate offset inputs (fun _ => 0))
  simpa only [advanceOffset, rangeOffset, privateCount, Nat.add_assoc] using scope

theorem outputChallenge_below (offset : Nat) (position : Fin ringDegree) :
    (outputChallenge offset position).VarsBelow (offset + privateCount) :=
  Expr.VarsBelow.mono _ (WideReduction.Program.outputChallenge_varsBelow (rangeOffset offset) position)
    (by unfold rangeOffset privateCount; omega)

theorem enter_eq (state : Poseidon2.State) (coordinate : Nat) :
    Lifecycle.Transcript.PiRlcSampler.enterScalar state coordinate = referenceEnter state coordinate := by
  simp [Lifecycle.Transcript.PiRlcSampler.enterScalar, Lifecycle.Transcript.absorb,
    referenceEnter, Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter,
    Poseidon2.rate, Lifecycle.natWord]

theorem soundness (interface : Interface) (coordinate : Nat) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (rows : holds env (operations interface coordinate offset)) :
    SpecHolds interface coordinate offset env := by
  have entryRows := rows (entryOp interface coordinate offset) (by simp [operations])
  have rangeRows := rows (rangeOp interface coordinate offset) (by simp [operations])
  have advanceRows := rows (advanceOp interface coordinate offset) (by simp [operations])
  have entered : v1_1.TranscriptAbsorption.SpecHolds interface coordinate offset env := entryRows inputs
  have decoded := rangeRows (range_inputs interface coordinate offset inputs)
  have advanced : Permutation.Owned.SpecHolds (advanceInterface interface coordinate offset)
      (advanceOffset offset) env := advanceRows (advance_inputs interface coordinate offset inputs env)
  have entryMeaning : referenceEnter (evalState env (interface.initialState offset)) coordinate =
      evalState env (enteredState interface coordinate offset) := by
    rw [← enter_eq]
    exact entered
  have drawMeaning : WideReduction.drawOf
      (WideReduction.Program.coreInterface (rangeInterface interface coordinate offset) (rangeOffset offset))
      env (WideReduction.Program.coreOffset (rangeOffset offset)) =
        referenceBlock (referenceEnter (evalState env (interface.initialState offset)) coordinate) := by
    rw [entryMeaning]
    funext lane
    change (enteredState interface coordinate offset (rateLane lane)).eval env =
      (List.ofFn (Layer.evalState env (enteredState interface coordinate offset))).getD lane.val 0
    rw [List.getD_eq_get _ _ ⟨lane.val, by have laneBound : lane.val < 4 := lane.isLt; simp only [List.length_ofFn]; omega⟩]
    simp only [List.get_ofFn, Layer.evalState]
    rfl
  refine ⟨?_, ?_⟩
  · change evalState env (outputState interface coordinate offset) =
      Poseidon2.permute (referenceEnter (evalState env (interface.initialState offset)) coordinate)
    rw [entryMeaning]
    exact advanced
  · intro position
    have digit := decoded position
    rw [drawMeaning] at digit
    exact digit

theorem entry_scope (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops (entry interface coordinate).main offset),
      expression.VarsBelow (offset + localLength (Circuit.ops (entry interface coordinate).main offset)) := by
  simp only [entry, FormalCircuit.withConstantFootprint_main]
  exact v1_1.TranscriptAbsorption.flatConstraints_varsBelow interface coordinate offset (fun _ => 0) inputs

theorem range_scope (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset)),
      expression.VarsBelow (rangeOffset offset +
        localLength (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset))) := by
  change ∀ expression ∈ flatConstraints (WideReduction.Program.operations _ _),
    expression.VarsBelow (_ + localLength (WideReduction.Program.operations _ _))
  rw [WideReduction.Program.localLength_eq]
  exact WideReduction.Program.flatConstraints_varsBelow _ _ (range_inputs interface coordinate offset inputs)

theorem advance_scope (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (advance interface coordinate offset).main (advanceOffset offset)),
      expression.VarsBelow (advanceOffset offset +
        localLength (Circuit.ops (advance interface coordinate offset).main (advanceOffset offset))) := by
  change ∀ expression ∈ flatConstraints (Permutation.Owned.operations _ _),
    expression.VarsBelow (_ + localLength (Permutation.Owned.operations _ _))
  rw [Permutation.Owned.localLength_eq]
  exact Permutation.Owned.flatConstraints_varsBelow _ _
    (advance_inputs interface coordinate offset inputs (fun _ => 0))

theorem complete (interface : Interface) (coordinate : Nat) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∃ completed,
      AgreesOutside env completed offset (localLength (operations interface coordinate offset)) ∧
      holdsFlat completed (operations interface coordinate offset) := by
  obtain ⟨entered, entryAgreement, entryRows⟩ :=
    v1_1.TranscriptAbsorption.complete interface coordinate env offset inputs
  obtain ⟨first, firstOps, firstEnd, _, _⟩ := Sequence.appendBuiltAt
    (Sequence.empty env offset) "pirlc.wide.enter_scalar" (entry interface coordinate) offset (by rfl)
    (entry_scope interface coordinate offset inputs) entered entryAgreement entryRows
  have firstStart : offset + localLength first.operations = rangeOffset offset := by
    simpa only [entry, FormalCircuit.withConstantFootprint_main,
      v1_1.TranscriptAbsorption.localLength_eq, rangeOffset] using firstEnd
  let decoded := WideReduction.Program.completeEnv (rangeInterface interface coordinate offset)
    first.current (rangeOffset offset)
  have rangeCorrect := WideReduction.Program.completeEnv_correct (rangeInterface interface coordinate offset)
    first.current (rangeOffset offset) (range_inputs interface coordinate offset inputs)
  have rangeAgreement : AgreesOutside first.current decoded (rangeOffset offset)
      (localLength (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset))) := by
    change AgreesOutside first.current decoded _ (localLength (WideReduction.Program.operations _ _))
    rw [WideReduction.Program.localLength_eq]
    exact rangeCorrect.1
  obtain ⟨second, secondOps, secondEnd, _, _⟩ := Sequence.appendBuiltAt first "pirlc.wide.reduce"
    (rangeCircuit interface coordinate offset) (rangeOffset offset) firstStart
    (range_scope interface coordinate offset inputs) decoded rangeAgreement rangeCorrect.2
  have secondStart : offset + localLength second.operations = advanceOffset offset := by
    change offset + localLength second.operations = rangeOffset offset + WideReduction.Program.privateCount
    have count := WideReduction.Program.localLength_eq (rangeInterface interface coordinate offset) (rangeOffset offset)
    change localLength (Circuit.ops (rangeCircuit interface coordinate offset).main (rangeOffset offset)) =
      WideReduction.Program.privateCount at count
    rwa [count] at secondEnd
  obtain ⟨advanced, advanceAgreement, advanceRows⟩ :=
    Permutation.Owned.complete (advanceInterface interface coordinate offset) second.current (advanceOffset offset)
      (advance_inputs interface coordinate offset inputs second.current)
  obtain ⟨third, thirdOps, _, _, _⟩ := Sequence.appendBuiltAt second "pirlc.wide.advance"
    (advance interface coordinate offset) (advanceOffset offset) secondStart
    (advance_scope interface coordinate offset inputs) advanced advanceAgreement advanceRows
  have allOps : third.operations = operations interface coordinate offset := by
    rw [thirdOps, secondOps, firstOps]
    rfl
  refine ⟨third.current, ?_, ?_⟩
  · have agreement := third.agrees
    rwa [allOps] at agreement
  · rw [← allOps]
    exact third.rows

theorem scope (interface : Interface) (coordinate : Nat) (offset : Nat)
    (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (operations interface coordinate offset),
      expression.VarsBelow (offset + privateCount) := by
  intro expression member
  simp only [operations, flatConstraints, List.flatMap_cons, List.flatMap_nil, List.mem_append,
    List.not_mem_nil, or_false] at member
  rcases member with first | second | third
  · have bounded := entry_scope interface coordinate offset inputs expression first
    apply Expr.VarsBelow.mono _ bounded
    rw [entry, FormalCircuit.withConstantFootprint_main, v1_1.TranscriptAbsorption.localLength_eq]
    unfold privateCount
    omega
  · have bounded := range_scope interface coordinate offset inputs expression second
    apply Expr.VarsBelow.mono _ bounded
    change rangeOffset offset + localLength (WideReduction.Program.operations _ _) ≤ _
    rw [WideReduction.Program.localLength_eq]
    unfold rangeOffset privateCount
    omega
  · have bounded := advance_scope interface coordinate offset inputs expression third
    apply Expr.VarsBelow.mono _ bounded
    change advanceOffset offset + localLength (Permutation.Owned.operations _ _) ≤ _
    rw [Permutation.Owned.localLength_eq]
    unfold advanceOffset rangeOffset privateCount
    omega

def circuit (interface : Interface) (coordinate : Nat) : FormalCircuit where
  main := fun offset => ((), offset + privateCount, operations interface coordinate offset)
  assumptions := fun offset _ => Assumptions interface offset
  spec := SpecHolds interface coordinate
  privateCount := fun _ => privateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface coordinate
  rowCount_eq := rowCount_eq interface coordinate
  soundness := soundness interface coordinate
  completeness := fun env offset inputs _ => complete interface coordinate env offset inputs

end NightstreamFPrime.Lifecycle.PiRLC.Wide.Scalar
