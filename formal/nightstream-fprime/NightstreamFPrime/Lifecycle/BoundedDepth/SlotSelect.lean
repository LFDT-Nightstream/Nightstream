import NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
import NightstreamFPrime.Lifecycle.BoundedDepth.Step
import NightstreamFPrime.Spec.FieldPrime

/-! The slot-select gadget of the light step (CW-2.5d, d4b). It allocates no variables: the step
layout supplies the iteration word with its inverse and recursive flag, the prior-circuit
selector, the state words and the slot words. The flag rows are those of
`Stage1.RunningTransition` (`r = iteration · inverse`, `iteration · (1 - r) = 0`); then a boolean
selector `s`, the prior pc `r · (pc - s - 1) = 0`, the base state `(1 - r) · (z0 - zi) = 0`, and
five word selections: the active slot from the prior slots, each slot's inner words (the new
claim for the selected slot, the prior slot otherwise), and each next slot (the default slot at
iteration 0, the inner words otherwise). That is 5 × 2,055 + 8 rows. `rows_implies_select` gives
`SelectSpec` and the active slot from the rows and the words' values. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.SlotSelect

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  iteration : Nat → Expr
  inverse : Nat → Expr
  recursive : Nat → Expr
  priorPc : Nat → Expr
  selector : Nat → Expr
  initialState : Nat → Fin 4 → Expr
  currentState : Nat → Fin 4 → Expr
  prior : Nat → Fin 2 → SlotExpr logicalWidth publicFits
  active : Nat → SlotExpr logicalWidth publicFits
  new : Nat → SlotExpr logicalWidth publicFits
  inner : Nat → Fin 2 → SlotExpr logicalWidth publicFits
  next : Nat → Fin 2 → SlotExpr logicalWidth publicFits

/-- The words of the default slot at depth 0 (HyperNova Definition 12). -/
def defaultWords : List Expr :=
  (serializeSlot ((defaultSlot, 0) : SlotData logicalWidth publicFits)).map Expr.const

variable (interface : Interface logicalWidth publicFits) (offset : Nat)

/-- `1` when `slot` is the selected slot: the selector for slot 1, one minus it for slot 0. -/
def isSelected (slot : Fin 2) : Expr :=
  if slot.val = 0 then 1 - interface.selector offset else interface.selector offset

def flagRows : List Expr :=
  [interface.recursive offset - interface.iteration offset * interface.inverse offset,
    interface.iteration offset * (1 - interface.recursive offset),
    interface.selector offset * (interface.selector offset - 1),
    interface.recursive offset * (interface.priorPc offset - interface.selector offset - 1)]

def baseStateRows : List Expr :=
  List.ofFn fun index : Fin 4 => (1 - interface.recursive offset) *
    (interface.initialState offset index - interface.currentState offset index)

def wordRows : List Expr :=
  selectRows (interface.selector offset) (interface.prior offset 0).words
      (interface.prior offset 1).words (interface.active offset).words ++
    selectRows (isSelected interface offset 0) (interface.prior offset 0).words
      (interface.new offset).words (interface.inner offset 0).words ++
    selectRows (isSelected interface offset 1) (interface.prior offset 1).words
      (interface.new offset).words (interface.inner offset 1).words ++
    selectRows (interface.recursive offset) (defaultWords (publicFits := publicFits))
      (interface.inner offset 0).words (interface.next offset 0).words ++
    selectRows (interface.recursive offset) (defaultWords (publicFits := publicFits))
      (interface.inner offset 1).words (interface.next offset 1).words

def rows : List Expr :=
  flagRows interface offset ++ baseStateRows interface offset ++ wordRows interface offset

def circuit : FormalCircuit where
  main := fun start => ((), start, (rows interface start).map Op.assertZero)
  spec := fun start env => ConstraintsHold env (rows interface start)
  soundness := by
    intro env _ _ holdsOps row member
    exact holdsOps (Op.assertZero row) (List.mem_map_of_mem member)
  completeness := by
    intro env _ _ spec
    refine ⟨env, fun _ _ => rfl, ?_⟩
    intro row member
    simp only [Circuit.ops, flatConstraints, List.flatMap_map, List.mem_flatMap,
      Op.flatConstraints, List.mem_singleton] at member
    obtain ⟨original, inside, rfl⟩ := member
    exact spec _ inside

/-- The selected slot: the selector's value as a slot index. -/
def selected (env : Env) : Fin 2 :=
  if (interface.selector offset).eval env = 1 then 1 else 0

/-- In the base field, a zero product has a zero factor. -/
theorem mul_eq_zero_of {left right : F} (zero : left * right = 0) :
    left = 0 ∨ right = 0 := by
  let : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩
  have key : ∀ first second : ZMod fieldModulus, first * second = 0 →
      first = 0 ∨ second = 0 := fun _ _ => mul_eq_zero.mp
  exact key left right zero

theorem one_ne_zero_F : (1 : F) ≠ 0 := by decide

theorem eval_one (env : Env) : (1 : Expr).eval env = 1 := rfl

/-- The flag rows: the recursive flag is 0 at iteration 0 and 1 otherwise, the selector is a bit,
and after the base case the prior pc is the selector plus one. -/
theorem flag_facts (env : Env) (holds : ConstraintsHold env (flagRows interface offset)) :
    ((interface.iteration offset).eval env = 0 → (interface.recursive offset).eval env = 0) ∧
      ((interface.iteration offset).eval env ≠ 0 →
        (interface.recursive offset).eval env = 1) ∧
      ((interface.selector offset).eval env = 0 ∨ (interface.selector offset).eval env = 1) ∧
      ((interface.recursive offset).eval env = 1 →
        (interface.priorPc offset).eval env = (interface.selector offset).eval env + 1) := by
  have recipe := holds _ (List.mem_cons_self ..)
  have binding := holds _ (List.mem_cons_of_mem _ (List.mem_cons_self ..))
  have bit := holds _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_self ..)))
  have pc := holds _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
    (List.mem_cons_of_mem _ (List.mem_cons_self ..))))
  simp only [Expr.eval_sub, Expr.eval_hmul] at recipe binding bit pc
  refine ⟨fun zero => ?_, fun nonzero => ?_, ?_, fun one => ?_⟩
  · rw [zero, zero_mul, sub_zero] at recipe
    exact recipe
  · rcases mul_eq_zero_of binding with zero | zero
    · exact absurd zero nonzero
    · exact (sub_eq_zero.mp zero).symm
  · rcases mul_eq_zero_of bit with zero | zero
    · exact Or.inl zero
    · exact Or.inr (sub_eq_zero.mp zero)
  · rw [one, one_mul, sub_sub, sub_eq_zero] at pc
    exact pc

private theorem defaultWords_eval (env : Env) :
    Hash.evalList env (defaultWords (publicFits := publicFits)) =
      serializeSlot ((defaultSlot, 0) : SlotData logicalWidth publicFits) := by
  simp [defaultWords, Hash.evalList, List.map_map, Function.comp_def]

private theorem defaultWords_length :
    (defaultWords (publicFits := publicFits)).length = 2055 := by
  rw [defaultWords, List.length_map, serializeSlot_length_eq]

/-- Below the modulus, a count's word is 0 exactly when the count is 0. -/
theorem natWord_eq_zero {count : Nat} (bound : count < fieldModulus) :
    natWord count = 0 ↔ count = 0 := by
  constructor
  · intro zero
    exact natWord_injective bound (by decide) (zero.trans rfl)
  · rintro rfl
    rfl

/-- The slot-select rows, with the words' values, give `SelectSpec` for the selected slot, with
the new slot's claim and depth, and show that the active slot is the selected prior slot. -/
theorem rows_implies_select (env : Env) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits)
    (holds : ConstraintsHold env (rows interface offset))
    (iterationWord : (interface.iteration offset).eval env = natWord input.iteration)
    (iterationBound : input.iteration < fieldModulus)
    (pcWord : (interface.priorPc offset).eval env = natWord input.priorPc)
    (pcBound : input.priorPc < fieldModulus)
    (initialWords :
      List.ofFn (fun index => (interface.initialState offset index).eval env) = input.z0)
    (currentWords :
      List.ofFn (fun index => (interface.currentState offset index).eval env) = input.zi)
    (priorWords : ∀ slot,
      Hash.evalList env (interface.prior offset slot).words = serializeSlot (input.running slot))
    (priorDepth : ∀ slot, (input.running slot).2 < fieldModulus)
    (nextWords : ∀ slot, Hash.evalList env (interface.next offset slot).words =
      serializeSlot (output.runningNext slot))
    (nextDepth : ∀ slot, (output.runningNext slot).2 < fieldModulus) :
    SelectSpec input output (selected interface offset env)
        ((interface.new offset).value env).1 ((interface.new offset).value env).2 ∧
      (interface.active offset).value env = input.running (selected interface offset env) := by
  obtain ⟨front, wordHolds⟩ := (constraintsHold_append env _ _).mp holds
  obtain ⟨flagHolds, baseHolds⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨zeroFlag, oneFlag, selectorBit, pcLink⟩ := flag_facts interface offset env flagHolds
  obtain ⟨first, nextOneRows⟩ := (constraintsHold_append env _ _).mp wordHolds
  obtain ⟨second, nextZeroRows⟩ := (constraintsHold_append env _ _).mp first
  obtain ⟨third, innerOneRows⟩ := (constraintsHold_append env _ _).mp second
  obtain ⟨activeRows, innerZeroRows⟩ := (constraintsHold_append env _ _).mp third
  have length := fun slot : SlotExpr logicalWidth publicFits => slot.words_length
  have active := selectRows_eval env _ _ _ _ (by rw [length, length]) (by rw [length, length])
    activeRows
  have innerZero := selectRows_eval env _ _ _ _ (by rw [length, length])
    (by rw [length, length]) innerZeroRows
  have innerOne := selectRows_eval env _ _ _ _ (by rw [length, length])
    (by rw [length, length]) innerOneRows
  have nextZero := selectRows_eval env _ _ _ _ (by rw [defaultWords_length, length])
    (by rw [length, length]) nextZeroRows
  have nextOne := selectRows_eval env _ _ _ _ (by rw [defaultWords_length, length])
    (by rw [length, length]) nextOneRows
  have slotOf := fun (data : SlotData logicalWidth publicFits) (depth : data.2 < fieldModulus)
      (slot : SlotExpr logicalWidth publicFits)
      (words : Hash.evalList env slot.words = serializeSlot data) =>
    slot.value_eq_of_words env data depth words
  have activeValue : (interface.active offset).value env =
      input.running (selected interface offset env) := by
    rcases selectorBit with zero | one
    · rw [show selected interface offset env = 0 by unfold selected; rw [zero]; decide]
      exact slotOf _ (priorDepth 0) _ ((active.1 zero).trans (priorWords 0))
    · rw [show selected interface offset env = 1 by unfold selected; rw [one]; decide]
      exact slotOf _ (priorDepth 1) _ ((active.2 one).trans (priorWords 1))
  refine ⟨⟨fun zero => ?_, fun nonzero => ?_⟩, activeValue⟩
  · have flagZero := zeroFlag (by rw [iterationWord, (natWord_eq_zero iterationBound).mpr zero])
    have recursiveZero : (interface.recursive offset).eval env = 0 := flagZero
    refine ⟨?_, ?_⟩
    · rw [← initialWords, ← currentWords]
      congr 1
      funext index
      have row := baseHolds _ ((List.mem_ofFn ..).mpr ⟨index, rfl⟩)
      simp only [Expr.eval_sub, Expr.eval_hmul, recursiveZero, sub_zero] at row
      rcases mul_eq_zero_of row with one | same
      · exact absurd one one_ne_zero_F
      · exact sub_eq_zero.mp same
    · funext slot
      have words : Hash.evalList env (interface.next offset slot).words =
          serializeSlot ((defaultSlot, 0) : SlotData logicalWidth publicFits) := by
        match slot with
        | 0 => exact (nextZero.1 recursiveZero).trans (defaultWords_eval env)
        | 1 => exact (nextOne.1 recursiveZero).trans (defaultWords_eval env)
      exact serializeSlot_injective (nextDepth slot) (show (0 : Nat) < fieldModulus by decide)
        ((nextWords slot).symm.trans words)
  · have recursiveOne : (interface.recursive offset).eval env = 1 :=
      oneFlag (by
        rw [iterationWord]
        exact fun zero => nonzero ((natWord_eq_zero iterationBound).mp zero))
    have pc := pcLink recursiveOne
    rw [pcWord] at pc
    have newValue := fun (slot : Fin 2)
        (words : Hash.evalList env (interface.next offset slot).words =
          Hash.evalList env (interface.new offset).words) =>
      serializeSlot_injective (nextDepth slot) ((interface.new offset).depth.eval env).isLt
        ((nextWords slot).symm.trans (words.trans ((interface.new offset).words_eval env)))
    have priorValue := fun (slot : Fin 2)
        (words : Hash.evalList env (interface.next offset slot).words =
          Hash.evalList env (interface.prior offset slot).words) =>
      serializeSlot_injective (nextDepth slot) (priorDepth slot)
        ((nextWords slot).symm.trans (words.trans (priorWords slot)))
    rcases selectorBit with zero | one
    · rw [show selected interface offset env = 0 by unfold selected; rw [zero]; decide]
      have selectedZero : (isSelected interface offset 0).eval env = 1 := by
        simp [isSelected, zero, eval_one]
      have selectedOne : (isSelected interface offset 1).eval env = 0 := by
        simp [isSelected, zero]
      have nextValue := newValue 0 ((nextZero.2 recursiveOne).trans (innerZero.2 selectedZero))
      refine ⟨?_, congrArg Prod.fst nextValue, congrArg Prod.snd nextValue,
        fun slot other => ?_⟩
      · rw [zero, zero_add] at pc
        exact natWord_injective pcBound (by decide) (pc.trans rfl)
      · match slot, other with
        | 1, _ => exact priorValue 1 ((nextOne.2 recursiveOne).trans (innerOne.1 selectedOne))
    · rw [show selected interface offset env = 1 by unfold selected; rw [one]; decide]
      have selectedZero : (isSelected interface offset 0).eval env = 0 := by
        simp [isSelected, one, eval_one]
      have selectedOne : (isSelected interface offset 1).eval env = 1 := by
        simp [isSelected, one]
      have nextValue := newValue 1 ((nextOne.2 recursiveOne).trans (innerOne.2 selectedOne))
      refine ⟨?_, congrArg Prod.fst nextValue, congrArg Prod.snd nextValue,
        fun slot other => ?_⟩
      · rw [one] at pc
        exact natWord_injective pcBound (by decide) (pc.trans rfl)
      · match slot, other with
        | 0, _ => exact priorValue 0 ((nextZero.2 recursiveOne).trans (innerZero.1 selectedZero))
        | 1, other => exact absurd rfl other

/-- The converse of `rows_implies_select`: words with the honest flags and the honest slot
selections satisfy every slot-select row. The recursive flag is 0 at iteration 0 and 1 with an
inverse otherwise; the active slot is the selected prior slot; the selected slot's inner words are
the new claim and the other slot's are its prior words; the next slots are the default slot at
iteration 0 and the inner words otherwise. -/
theorem rows_of_select (env : Env)
    (flags : ((interface.iteration offset).eval env = 0 ∧
        (interface.recursive offset).eval env = 0) ∨
      ((interface.iteration offset).eval env * (interface.inverse offset).eval env = 1 ∧
        (interface.recursive offset).eval env = 1))
    (selectorBit : (interface.selector offset).eval env = 0 ∨
      (interface.selector offset).eval env = 1)
    (pcLink : (interface.recursive offset).eval env = 1 →
      (interface.priorPc offset).eval env = (interface.selector offset).eval env + 1)
    (baseState : (interface.recursive offset).eval env = 0 → ∀ index,
      (interface.initialState offset index).eval env =
        (interface.currentState offset index).eval env)
    (activeWords : Hash.evalList env (interface.active offset).words =
      Hash.evalList env (interface.prior offset (selected interface offset env)).words)
    (innerSelected : Hash.evalList env (interface.inner offset (selected interface offset env)).words =
      Hash.evalList env (interface.new offset).words)
    (innerOther : ∀ slot, slot ≠ selected interface offset env →
      Hash.evalList env (interface.inner offset slot).words =
        Hash.evalList env (interface.prior offset slot).words)
    (nextBase : (interface.recursive offset).eval env = 0 → ∀ slot,
      Hash.evalList env (interface.next offset slot).words =
        Hash.evalList env (defaultWords (publicFits := publicFits)))
    (nextRecursive : (interface.recursive offset).eval env = 1 → ∀ slot,
      Hash.evalList env (interface.next offset slot).words =
        Hash.evalList env (interface.inner offset slot).words) :
    ConstraintsHold env (rows interface offset) := by
  have length := fun slot : SlotExpr logicalWidth publicFits => slot.words_length
  have recursiveBit : (interface.recursive offset).eval env = 0 ∨
      (interface.recursive offset).eval env = 1 := by
    rcases flags with ⟨_, zero⟩ | ⟨_, one⟩
    · exact Or.inl zero
    · exact Or.inr one
  have selectedZero : (interface.selector offset).eval env = 0 →
      selected interface offset env = 0 := fun zero => by
    unfold selected
    rw [zero]
    decide
  have selectedOne : (interface.selector offset).eval env = 1 →
      selected interface offset env = 1 := fun one => by
    unfold selected
    rw [one]
    decide
  refine (constraintsHold_append env _ _).mpr
    ⟨(constraintsHold_append env _ _).mpr ⟨?_, ?_⟩, ?_⟩
  · intro row member
    simp only [flagRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl <;>
      simp only [Expr.eval_sub, Expr.eval_hmul, eval_one]
    · rcases flags with ⟨zero, recursiveZero⟩ | ⟨product, recursiveOne⟩
      · rw [zero, recursiveZero, zero_mul, sub_zero]
      · rw [product, recursiveOne, sub_self]
    · rcases flags with ⟨zero, _⟩ | ⟨_, recursiveOne⟩
      · rw [zero, zero_mul]
      · rw [recursiveOne, sub_self, mul_zero]
    · rcases selectorBit with zero | one
      · rw [zero, zero_mul]
      · rw [one, sub_self, mul_zero]
    · rcases recursiveBit with zero | one
      · rw [zero, zero_mul]
      · rw [one, pcLink one, one_mul, add_sub_cancel_left, sub_self]
  · intro row member
    obtain ⟨index, rfl⟩ := (List.mem_ofFn ..).mp member
    simp only [Expr.eval_hmul, Expr.eval_sub, eval_one]
    rcases recursiveBit with zero | one
    · rw [zero, baseState zero index, sub_self, mul_zero]
    · rw [one, sub_self, zero_mul]
  · have select := fun (condition : Expr) (left right out : List Expr)
        (leftLength : left.length = out.length) (rightLength : right.length = out.length) =>
      selectRows_holds env condition left right out leftLength rightLength
    refine (constraintsHold_append env _ _).mpr ⟨(constraintsHold_append env _ _).mpr
      ⟨(constraintsHold_append env _ _).mpr ⟨(constraintsHold_append env _ _).mpr
        ⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩
    · apply select _ _ _ _ (by rw [length, length]) (by rw [length, length])
      rcases selectorBit with zero | one
      · refine Or.inl ⟨zero, ?_⟩
        rw [activeWords, selectedZero zero]
      · refine Or.inr ⟨one, ?_⟩
        rw [activeWords, selectedOne one]
    · apply select _ _ _ _ (by rw [length, length]) (by rw [length, length])
      rcases selectorBit with zero | one
      · refine Or.inr ⟨by simp [isSelected, zero, eval_one], ?_⟩
        have words := innerSelected
        rw [selectedZero zero] at words
        exact words
      · refine Or.inl ⟨by simp [isSelected, one, eval_one], ?_⟩
        exact innerOther 0 (by rw [selectedOne one]; decide)
    · apply select _ _ _ _ (by rw [length, length]) (by rw [length, length])
      rcases selectorBit with zero | one
      · refine Or.inl ⟨by simp [isSelected, zero], ?_⟩
        exact innerOther 1 (by rw [selectedZero zero]; decide)
      · refine Or.inr ⟨by simp [isSelected, one], ?_⟩
        have words := innerSelected
        rw [selectedOne one] at words
        exact words
    · apply select _ _ _ _ (by rw [defaultWords_length, length]) (by rw [length, length])
      rcases recursiveBit with zero | one
      · exact Or.inl ⟨zero, nextBase zero 0⟩
      · exact Or.inr ⟨one, nextRecursive one 0⟩
    · apply select _ _ _ _ (by rw [defaultWords_length, length]) (by rw [length, length])
      rcases recursiveBit with zero | one
      · exact Or.inl ⟨zero, nextBase zero 1⟩
      · exact Or.inr ⟨one, nextRecursive one 1⟩

/-- The slot-select rows hold again on values that keep every word the rows read: the counters,
the selector, the states and every slot's words. -/
theorem rows_congr (left right : Env)
    (iterationEq : (interface.iteration offset).eval right = (interface.iteration offset).eval left)
    (inverseEq : (interface.inverse offset).eval right = (interface.inverse offset).eval left)
    (recursiveEq : (interface.recursive offset).eval right = (interface.recursive offset).eval left)
    (pcEq : (interface.priorPc offset).eval right = (interface.priorPc offset).eval left)
    (selectorEq : (interface.selector offset).eval right = (interface.selector offset).eval left)
    (initialEq : ∀ index,
      (interface.initialState offset index).eval right =
        (interface.initialState offset index).eval left)
    (currentEq : ∀ index,
      (interface.currentState offset index).eval right =
        (interface.currentState offset index).eval left)
    (priorEq : ∀ slot, Hash.evalList right (interface.prior offset slot).words =
      Hash.evalList left (interface.prior offset slot).words)
    (activeEq : Hash.evalList right (interface.active offset).words =
      Hash.evalList left (interface.active offset).words)
    (newEq : Hash.evalList right (interface.new offset).words =
      Hash.evalList left (interface.new offset).words)
    (innerEq : ∀ slot, Hash.evalList right (interface.inner offset slot).words =
      Hash.evalList left (interface.inner offset slot).words)
    (nextEq : ∀ slot, Hash.evalList right (interface.next offset slot).words =
      Hash.evalList left (interface.next offset slot).words)
    (holds : ConstraintsHold left (rows interface offset)) :
    ConstraintsHold right (rows interface offset) := by
  obtain ⟨front, wordHolds⟩ := (constraintsHold_append left _ _).mp holds
  obtain ⟨flagHolds, baseHolds⟩ := (constraintsHold_append left _ _).mp front
  have defaultEq : Hash.evalList right (defaultWords (publicFits := publicFits)) =
      Hash.evalList left (defaultWords (publicFits := publicFits)) := by
    rw [defaultWords_eval, defaultWords_eval]
  have isSelectedEq : ∀ slot, (isSelected interface offset slot).eval right =
      (isSelected interface offset slot).eval left := by
    intro slot
    unfold isSelected
    split_ifs
    · simp only [Expr.eval_sub, eval_one, selectorEq]
    · exact selectorEq
  refine (constraintsHold_append right _ _).mpr
    ⟨(constraintsHold_append right _ _).mpr ⟨?_, ?_⟩, ?_⟩
  · intro row member
    have holdsRow := flagHolds row member
    simp only [flagRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl <;>
      simp only [Expr.eval_sub, Expr.eval_hmul, eval_one, iterationEq, inverseEq, recursiveEq,
        pcEq, selectorEq] at holdsRow ⊢ <;>
      exact holdsRow
  · intro row member
    have holdsRow := baseHolds row member
    obtain ⟨index, rfl⟩ := (List.mem_ofFn ..).mp member
    simp only [Expr.eval_sub, Expr.eval_hmul, eval_one, recursiveEq, initialEq index,
      currentEq index] at holdsRow ⊢
    exact holdsRow
  · have select := fun (condition : Expr) (conditionEq : condition.eval right = condition.eval left)
        (first second out : List Expr) => selectRows_congr left right condition conditionEq first
        second out
    obtain ⟨first, nextOneRows⟩ := (constraintsHold_append left _ _).mp wordHolds
    obtain ⟨second, nextZeroRows⟩ := (constraintsHold_append left _ _).mp first
    obtain ⟨third, innerOneRows⟩ := (constraintsHold_append left _ _).mp second
    obtain ⟨activeRows, innerZeroRows⟩ := (constraintsHold_append left _ _).mp third
    refine (constraintsHold_append right _ _).mpr ⟨(constraintsHold_append right _ _).mpr
      ⟨(constraintsHold_append right _ _).mpr ⟨(constraintsHold_append right _ _).mpr
        ⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩
    · exact select _ selectorEq _ _ _ (priorEq 0) (priorEq 1) activeEq activeRows
    · exact select _ (isSelectedEq 0) _ _ _ (priorEq 0) newEq (innerEq 0) innerZeroRows
    · exact select _ (isSelectedEq 1) _ _ _ (priorEq 1) newEq (innerEq 1) innerOneRows
    · exact select _ recursiveEq _ _ _ defaultEq (innerEq 0) (nextEq 0) nextZeroRows
    · exact select _ recursiveEq _ _ _ defaultEq (innerEq 1) (nextEq 1) nextOneRows

end NightstreamFPrime.Lifecycle.BoundedDepth.SlotSelect
