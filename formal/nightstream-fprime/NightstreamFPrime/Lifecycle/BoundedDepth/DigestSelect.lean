import NightstreamFPrime.Lifecycle.BoundedDepth.SlotSelect
import NightstreamFPrime.Lifecycle.BoundedDepth.StateHashGadget

/-! The digest select of the light and refresh step circuits. A step changes only the selected
slot, so its next state needs the new slot's digest and the other slot's prior digest, not the
next slots' words (HyperNova §6.3, the optimization after Construction 2). The gadget allocates no
variables: the step layout supplies the counters, the selector, the states, the prior and active
slots and the slot digests. Its rows are the slot select's flag and base-state rows, the active
slot's selection from the prior slots, and two digest selections per slot: the inner digest (the
new slot's digest for the selected slot, the prior digest otherwise) and the next digest (the
default slot's digest at iteration 0, the inner digest otherwise). That is 2,055 + 16 + 8 rows.
`rows_implies_select` gives `SelectSpec` for the next slots `nextSlots`, the active slot and the
digest of each next slot. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.DigestSelect

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  iteration : Nat → Expr
  inverse : Nat → Expr
  recursive : Nat → Expr
  priorPc : Nat → Expr
  selector : Nat → Expr
  initialState : Nat → Fin 4 → Expr
  currentState : Nat → Fin 4 → Expr
  prior : Nat → Fin 2 → SlotExpr logicalWidth publicFits
  active : Nat → SlotExpr logicalWidth publicFits
  priorDigest : Nat → Fin 2 → Fin 4 → Expr
  newDigest : Nat → Fin 4 → Expr
  innerDigest : Nat → Fin 2 → Fin 4 → Expr
  nextDigest : Nat → Fin 2 → Fin 4 → Expr

/-- The default slot at depth 0 (HyperNova Definition 12). -/
def defaultData : SlotData logicalWidth publicFits := (defaultSlot, 0)

/-- The next slots of a step that folds the selected slot into `next`: the default slots at
iteration 0, otherwise `next` in the selected slot and the prior slot in the other
(`SelectSpec`). -/
def nextSlots (iteration : Nat) (running : Fin 2 → SlotData logicalWidth publicFits)
    (selected : Fin 2) (next : SlotData logicalWidth publicFits) :
    Fin 2 → SlotData logicalWidth publicFits :=
  fun slot => if iteration = 0 then defaultData else if slot = selected then next else running slot

/-- The next slots' depths are field values when the prior slots' and the new slot's are. -/
theorem nextSlots_depth (iteration : Nat) (running : Fin 2 → SlotData logicalWidth publicFits)
    (selected : Fin 2) (next : SlotData logicalWidth publicFits)
    (runningBound : ∀ slot, (running slot).2 < goldilocksModulus)
    (nextBound : next.2 < goldilocksModulus) (slot : Fin 2) :
    (nextSlots iteration running selected next slot).2 < goldilocksModulus := by
  unfold nextSlots
  split_ifs
  · show (0 : Nat) < goldilocksModulus
    decide
  · exact nextBound
  · exact runningBound slot

/-- The default slot's digest as constant words. -/
def defaultDigest : List Expr :=
  (slotDigest (defaultData (logicalWidth := logicalWidth) (publicFits := publicFits))).map
    Expr.const

variable (interface : Interface logicalWidth publicFits) (offset : Nat)

/-- The slot-select interface of this gadget's counters, selector and states, whose flag and
base-state rows this gadget reuses. Those rows do not read its new, inner and next slots, which
are the active slot here. -/
def control : SlotSelect.Interface logicalWidth publicFits where
  iteration := interface.iteration
  inverse := interface.inverse
  recursive := interface.recursive
  priorPc := interface.priorPc
  selector := interface.selector
  initialState := interface.initialState
  currentState := interface.currentState
  prior := interface.prior
  active := interface.active
  new := interface.active
  inner := fun offset _ => interface.active offset
  next := fun offset _ => interface.active offset

def activeRows : List Expr :=
  selectRows (interface.selector offset) (interface.prior offset 0).words
    (interface.prior offset 1).words (interface.active offset).words

def digestRows (slot : Fin 2) : List Expr :=
  selectRows (SlotSelect.isSelected (control interface) offset slot)
      (List.ofFn (interface.priorDigest offset slot)) (List.ofFn (interface.newDigest offset))
      (List.ofFn (interface.innerDigest offset slot)) ++
    selectRows (interface.recursive offset) (defaultDigest (publicFits := publicFits))
      (List.ofFn (interface.innerDigest offset slot)) (List.ofFn (interface.nextDigest offset slot))

def rows : List Expr :=
  SlotSelect.flagRows (control interface) offset ++
    SlotSelect.baseStateRows (control interface) offset ++ activeRows interface offset ++
    digestRows interface offset 0 ++ digestRows interface offset 1

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
def selected (env : Env) : Fin 2 := SlotSelect.selected (control interface) offset env

theorem ofFn_eval (digest : Fin 4 → Expr) (env : Env) :
    Hash.evalList env (List.ofFn digest) = digestValue digest env := by
  simp [Hash.evalList, digestValue]

theorem defaultDigest_eval (env : Env) :
    Hash.evalList env (defaultDigest (logicalWidth := logicalWidth) (publicFits := publicFits)) =
      slotDigest (defaultData (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
  simp [defaultDigest, Hash.evalList, List.map_map, Function.comp_def]

theorem defaultDigest_length :
    (defaultDigest (logicalWidth := logicalWidth) (publicFits := publicFits)).length = 4 := by
  rw [defaultDigest, List.length_map, slotDigest, Poseidon2.hash_length]
  rfl

theorem selector_facts (env : Env)
    (selectorBit : (interface.selector offset).eval env = 0 ∨
      (interface.selector offset).eval env = 1) (slot : Fin 2) :
    (slot = selected interface offset env →
        (SlotSelect.isSelected (control interface) offset slot).eval env = 1) ∧
      (slot ≠ selected interface offset env →
        (SlotSelect.isSelected (control interface) offset slot).eval env = 0) := by
  rcases selectorBit with zero | one
  · have selectedZero : selected interface offset env = 0 := by
      unfold selected SlotSelect.selected
      simp only [control, zero]
      decide
    rw [selectedZero]
    match slot with
    | 0 => exact ⟨fun _ => by simp [SlotSelect.isSelected, control, zero, SlotSelect.eval_one],
        fun other => absurd rfl other⟩
    | 1 => exact ⟨fun same => absurd same (by decide),
        fun _ => by simp [SlotSelect.isSelected, control, zero]⟩
  · have selectedOne : selected interface offset env = 1 := by
      unfold selected SlotSelect.selected
      simp only [control, one]
      decide
    rw [selectedOne]
    match slot with
    | 0 => exact ⟨fun same => absurd same (by decide),
        fun _ => by simp [SlotSelect.isSelected, control, one, SlotSelect.eval_one]⟩
    | 1 => exact ⟨fun _ => by simp [SlotSelect.isSelected, control, one],
        fun other => absurd rfl other⟩

/-- A slot's digest rows: the inner digest is the prior digest or the new digest, and the next
digest is the default slot's digest or the inner digest. -/
theorem digestRows_eval (env : Env) (slot : Fin 2)
    (holds : ConstraintsHold env (digestRows interface offset slot)) :
    ((SlotSelect.isSelected (control interface) offset slot).eval env = 0 →
        digestValue (interface.innerDigest offset slot) env =
          digestValue (interface.priorDigest offset slot) env) ∧
      ((SlotSelect.isSelected (control interface) offset slot).eval env = 1 →
        digestValue (interface.innerDigest offset slot) env =
          digestValue (interface.newDigest offset) env) ∧
      ((interface.recursive offset).eval env = 0 →
        digestValue (interface.nextDigest offset slot) env =
          slotDigest (defaultData (logicalWidth := logicalWidth) (publicFits := publicFits))) ∧
      ((interface.recursive offset).eval env = 1 →
        digestValue (interface.nextDigest offset slot) env =
          digestValue (interface.innerDigest offset slot) env) := by
  obtain ⟨innerHolds, nextHolds⟩ := (constraintsHold_append env _ _).mp holds
  have inner := selectRows_eval env _ _ _ _ (by simp) (by simp) innerHolds
  have next := selectRows_eval env _ _ _ _ (by rw [defaultDigest_length]; simp) (by simp)
    nextHolds
  simp only [ofFn_eval, defaultDigest_eval] at inner next
  exact ⟨inner.1, inner.2, next.1, next.2⟩

theorem selected_eq (env : Env) :
    selected interface offset env = if (interface.selector offset).eval env = 1 then 1 else 0 :=
  rfl

private theorem selected_of_bit (env : Env) :
    ((interface.selector offset).eval env = 0 → selected interface offset env = 0) ∧
      ((interface.selector offset).eval env = 1 → selected interface offset env = 1) :=
  ⟨fun zero => by rw [selected_eq, zero]; decide, fun one => by rw [selected_eq, one]; decide⟩

/-- The flag and base-state rows give the counters' meaning: the selector is a bit; at iteration 0
the recursive flag is 0 and `z0 = zi`; after it the recursive flag is 1 and the prior pc names the
selected slot. -/
theorem control_facts (env : Env) (input : StepInput logicalWidth publicFits)
    (flagHolds : ConstraintsHold env (SlotSelect.flagRows (control interface) offset))
    (baseHolds : ConstraintsHold env (SlotSelect.baseStateRows (control interface) offset))
    (iterationWord : (interface.iteration offset).eval env = natWord input.iteration)
    (iterationBound : input.iteration < goldilocksModulus)
    (pcWord : (interface.priorPc offset).eval env = natWord input.priorPc)
    (pcBound : input.priorPc < goldilocksModulus)
    (initialWords :
      List.ofFn (fun index => (interface.initialState offset index).eval env) = input.z0)
    (currentWords :
      List.ofFn (fun index => (interface.currentState offset index).eval env) = input.zi) :
    ((interface.selector offset).eval env = 0 ∨ (interface.selector offset).eval env = 1) ∧
      (input.iteration = 0 → (interface.recursive offset).eval env = 0 ∧ input.z0 = input.zi) ∧
      (input.iteration ≠ 0 → (interface.recursive offset).eval env = 1 ∧
        input.priorPc = (selected interface offset env).val + 1) := by
  obtain ⟨zeroFlag, oneFlag, selectorBit, pcLink⟩ :
      ((interface.iteration offset).eval env = 0 → (interface.recursive offset).eval env = 0) ∧
        ((interface.iteration offset).eval env ≠ 0 →
          (interface.recursive offset).eval env = 1) ∧
        ((interface.selector offset).eval env = 0 ∨ (interface.selector offset).eval env = 1) ∧
        ((interface.recursive offset).eval env = 1 →
          (interface.priorPc offset).eval env = (interface.selector offset).eval env + 1) :=
    SlotSelect.flag_facts (control interface) offset env flagHolds
  have recursiveZero : input.iteration = 0 → (interface.recursive offset).eval env = 0 :=
    fun zero => zeroFlag (by
      rw [iterationWord, (SlotSelect.natWord_eq_zero iterationBound).mpr zero])
  have recursiveOne : input.iteration ≠ 0 → (interface.recursive offset).eval env = 1 :=
    fun nonzero => oneFlag (by
      rw [iterationWord]
      exact fun word => nonzero ((SlotSelect.natWord_eq_zero iterationBound).mp word))
  obtain ⟨selectedZero, selectedOne⟩ := selected_of_bit interface offset env
  refine ⟨selectorBit, fun zero => ⟨recursiveZero zero, ?_⟩,
    fun nonzero => ⟨recursiveOne nonzero, ?_⟩⟩
  · rw [← initialWords, ← currentWords]
    congr 1
    funext index
    have row := baseHolds _ ((List.mem_ofFn ..).mpr ⟨index, rfl⟩)
    simp only [control, Expr.eval_sub, Expr.eval_hmul, recursiveZero zero, sub_zero] at row
    rcases SlotSelect.mul_eq_zero_of row with one | same
    · exact absurd one SlotSelect.one_ne_zero_F
    · exact sub_eq_zero.mp same
  · have pc := pcLink (recursiveOne nonzero)
    rw [pcWord] at pc
    rcases selectorBit with zero | one
    · rw [selectedZero zero]
      rw [zero, zero_add] at pc
      exact natWord_injective pcBound (by decide) (pc.trans rfl)
    · rw [selectedOne one]
      rw [one] at pc
      exact natWord_injective pcBound (by decide) (pc.trans rfl)

/-- The control facts, the digest rows and, after the base case, prior digests that are the prior
slots' digests give `SelectSpec` for `nextSlots` and the digest of each next slot. -/
theorem select_of_digestRows (env : Env) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) (new : SlotData logicalWidth publicFits)
    (selectorBit : (interface.selector offset).eval env = 0 ∨
      (interface.selector offset).eval env = 1)
    (base : input.iteration = 0 →
      (interface.recursive offset).eval env = 0 ∧ input.z0 = input.zi)
    (recursive : input.iteration ≠ 0 → (interface.recursive offset).eval env = 1 ∧
      input.priorPc = (selected interface offset env).val + 1)
    (digestHolds : ∀ slot, ConstraintsHold env (digestRows interface offset slot))
    (priorDigests : input.iteration ≠ 0 → ∀ slot,
      digestValue (interface.priorDigest offset slot) env = slotDigest (input.running slot))
    (newDigest : digestValue (interface.newDigest offset) env = slotDigest new)
    (next : output.runningNext =
      nextSlots input.iteration input.running (selected interface offset env) new) :
    SelectSpec input output (selected interface offset env) new.1 new.2 ∧
      ∀ slot, digestValue (interface.nextDigest offset slot) env =
        slotDigest (output.runningNext slot) := by
  refine ⟨⟨fun zero => ⟨(base zero).2, by
        rw [next]
        funext slot
        simp [nextSlots, zero, defaultData]⟩,
      fun nonzero => ⟨(recursive nonzero).2, by rw [next]; simp [nextSlots, nonzero],
        by rw [next]; simp [nextSlots, nonzero],
        fun slot other => by rw [next]; simp [nextSlots, nonzero, other]⟩⟩, fun slot => ?_⟩
  obtain ⟨innerPrior, innerNew, nextDefault, nextInner⟩ :=
    digestRows_eval interface offset env slot (digestHolds slot)
  obtain ⟨isSelectedOne, isSelectedZero⟩ := selector_facts interface offset env selectorBit slot
  rw [next]
  by_cases zero : input.iteration = 0
  · rw [show nextSlots input.iteration input.running (selected interface offset env) new slot =
        defaultData by simp [nextSlots, zero]]
    exact nextDefault (base zero).1
  · rw [nextInner (recursive zero).1]
    by_cases same : slot = selected interface offset env
    · rw [show nextSlots input.iteration input.running (selected interface offset env) new slot =
          new by simp [nextSlots, zero, same], innerNew (isSelectedOne same), newDigest]
    · rw [show nextSlots input.iteration input.running (selected interface offset env) new slot =
          input.running slot by simp [nextSlots, zero, same], innerPrior (isSelectedZero same),
        priorDigests zero]

/-- The rows, with the words' values and the prior and new digests, give `SelectSpec` for the next
slots `nextSlots`, the active slot, and each next digest as the digest of its next slot. -/
theorem rows_implies_select (env : Env) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) (new : SlotData logicalWidth publicFits)
    (holds : ConstraintsHold env (rows interface offset))
    (iterationWord : (interface.iteration offset).eval env = natWord input.iteration)
    (iterationBound : input.iteration < goldilocksModulus)
    (pcWord : (interface.priorPc offset).eval env = natWord input.priorPc)
    (pcBound : input.priorPc < goldilocksModulus)
    (initialWords :
      List.ofFn (fun index => (interface.initialState offset index).eval env) = input.z0)
    (currentWords :
      List.ofFn (fun index => (interface.currentState offset index).eval env) = input.zi)
    (priorWords : ∀ slot,
      Hash.evalList env (interface.prior offset slot).words = serializeSlot (input.running slot))
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (priorDigests : ∀ slot,
      digestValue (interface.priorDigest offset slot) env = slotDigest (input.running slot))
    (newDigest : digestValue (interface.newDigest offset) env = slotDigest new)
    (next : output.runningNext =
      nextSlots input.iteration input.running (selected interface offset env) new) :
    SelectSpec input output (selected interface offset env) new.1 new.2 ∧
      (interface.active offset).value env = input.running (selected interface offset env) ∧
      ∀ slot, digestValue (interface.nextDigest offset slot) env =
        slotDigest (output.runningNext slot) := by
  obtain ⟨front, digestOneHolds⟩ := (constraintsHold_append env _ _).mp holds
  obtain ⟨front, digestZeroHolds⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨front, activeHolds⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨flagHolds, baseHolds⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨selectorBit, base, recursive⟩ := control_facts interface offset env input flagHolds
    baseHolds iterationWord iterationBound pcWord pcBound initialWords currentWords
  obtain ⟨selectedZero, selectedOne⟩ := selected_of_bit interface offset env
  have length := fun slot : SlotExpr logicalWidth publicFits => slot.words_length
  have active := selectRows_eval env _ _ _ _ (by rw [length, length]) (by rw [length, length])
    activeHolds
  have activeValue : (interface.active offset).value env =
      input.running (selected interface offset env) := by
    rcases selectorBit with zero | one
    · rw [selectedZero zero]
      exact (interface.active offset).value_eq_of_words env _ (priorDepth 0)
        ((active.1 zero).trans (priorWords 0))
    · rw [selectedOne one]
      exact (interface.active offset).value_eq_of_words env _ (priorDepth 1)
        ((active.2 one).trans (priorWords 1))
  obtain ⟨select, nextDigests⟩ := select_of_digestRows interface offset env input output new
    selectorBit base recursive (fun slot => by
      match slot with
      | 0 => exact digestZeroHolds
      | 1 => exact digestOneHolds) (fun _ => priorDigests) newDigest next
  exact ⟨select, activeValue, nextDigests⟩

/-- The flag and base-state rows hold on honest counters. -/
theorem controlRows_holds (env : Env)
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
        (interface.currentState offset index).eval env) :
    ConstraintsHold env (SlotSelect.flagRows (control interface) offset ++
      SlotSelect.baseStateRows (control interface) offset) := by
  have recursiveBit : (interface.recursive offset).eval env = 0 ∨
      (interface.recursive offset).eval env = 1 := by
    rcases flags with ⟨_, zero⟩ | ⟨_, one⟩
    · exact Or.inl zero
    · exact Or.inr one
  refine (constraintsHold_append env _ _).mpr ⟨?_, ?_⟩
  · intro row member
    simp only [SlotSelect.flagRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl <;>
      simp only [control, Expr.eval_sub, Expr.eval_hmul, SlotSelect.eval_one]
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
    simp only [control, Expr.eval_hmul, Expr.eval_sub, SlotSelect.eval_one]
    rcases recursiveBit with zero | one
    · rw [zero, baseState zero index, sub_self, mul_zero]
    · rw [one, sub_self, zero_mul]

/-- A slot's digest rows hold on honest digest selections. -/
theorem digestRows_holds (env : Env)
    (selectorBit : (interface.selector offset).eval env = 0 ∨
      (interface.selector offset).eval env = 1)
    (recursiveBit : (interface.recursive offset).eval env = 0 ∨
      (interface.recursive offset).eval env = 1)
    (innerSelected :
      digestValue (interface.innerDigest offset (selected interface offset env)) env =
        digestValue (interface.newDigest offset) env)
    (innerOther : ∀ slot, slot ≠ selected interface offset env →
      digestValue (interface.innerDigest offset slot) env =
        digestValue (interface.priorDigest offset slot) env)
    (nextBase : (interface.recursive offset).eval env = 0 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        slotDigest (defaultData (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (nextRecursive : (interface.recursive offset).eval env = 1 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        digestValue (interface.innerDigest offset slot) env) (slot : Fin 2) :
    ConstraintsHold env (digestRows interface offset slot) := by
  obtain ⟨isSelectedOne, isSelectedZero⟩ :=
    selector_facts interface offset env selectorBit slot
  refine (constraintsHold_append env _ _).mpr ⟨?_, ?_⟩
  · apply selectRows_holds env _ _ _ _ (by simp) (by simp)
    by_cases same : slot = selected interface offset env
    · refine Or.inr ⟨isSelectedOne same, ?_⟩
      rw [ofFn_eval, ofFn_eval, same]
      exact innerSelected
    · refine Or.inl ⟨isSelectedZero same, ?_⟩
      rw [ofFn_eval, ofFn_eval]
      exact innerOther slot same
  · apply selectRows_holds env _ _ _ _ (by rw [defaultDigest_length]; simp) (by simp)
    rcases recursiveBit with zero | one
    · refine Or.inl ⟨zero, ?_⟩
      rw [ofFn_eval, defaultDigest_eval]
      exact nextBase zero slot
    · refine Or.inr ⟨one, ?_⟩
      rw [ofFn_eval, ofFn_eval]
      exact nextRecursive one slot

/-- The converse of `rows_implies_select`: words with the honest flags, the honest active slot and
the honest digest selections satisfy every row. -/
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
    (innerSelected :
      digestValue (interface.innerDigest offset (selected interface offset env)) env =
        digestValue (interface.newDigest offset) env)
    (innerOther : ∀ slot, slot ≠ selected interface offset env →
      digestValue (interface.innerDigest offset slot) env =
        digestValue (interface.priorDigest offset slot) env)
    (nextBase : (interface.recursive offset).eval env = 0 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        slotDigest (defaultData (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (nextRecursive : (interface.recursive offset).eval env = 1 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        digestValue (interface.innerDigest offset slot) env) :
    ConstraintsHold env (rows interface offset) := by
  have length := fun slot : SlotExpr logicalWidth publicFits => slot.words_length
  have recursiveBit : (interface.recursive offset).eval env = 0 ∨
      (interface.recursive offset).eval env = 1 := by
    rcases flags with ⟨_, zero⟩ | ⟨_, one⟩
    · exact Or.inl zero
    · exact Or.inr one
  have digestHolds := digestRows_holds interface offset env selectorBit recursiveBit innerSelected
    innerOther nextBase nextRecursive
  refine (constraintsHold_append env _ _).mpr ⟨(constraintsHold_append env _ _).mpr
    ⟨(constraintsHold_append env _ _).mpr
      ⟨controlRows_holds interface offset env flags selectorBit pcLink baseState, ?_⟩,
      digestHolds 0⟩, digestHolds 1⟩
  apply selectRows_holds env _ _ _ _ (by rw [length, length]) (by rw [length, length])
  rcases selectorBit with zero | one
  · refine Or.inl ⟨zero, ?_⟩
    rw [activeWords, selected_eq, zero]
    rfl
  · refine Or.inr ⟨one, ?_⟩
    rw [activeWords, selected_eq, one]
    rfl

/-- The flag and base-state rows hold again on values that keep every word they read. -/
theorem controlRows_congr (left right : Env)
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
    (holds : ConstraintsHold left (SlotSelect.flagRows (control interface) offset ++
      SlotSelect.baseStateRows (control interface) offset)) :
    ConstraintsHold right (SlotSelect.flagRows (control interface) offset ++
      SlotSelect.baseStateRows (control interface) offset) := by
  obtain ⟨flagHolds, baseHolds⟩ := (constraintsHold_append left _ _).mp holds
  refine (constraintsHold_append right _ _).mpr ⟨?_, ?_⟩
  · intro row member
    have holdsRow := flagHolds row member
    simp only [SlotSelect.flagRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl <;>
      simp only [control, Expr.eval_sub, Expr.eval_hmul, SlotSelect.eval_one, iterationEq,
        inverseEq, recursiveEq, pcEq, selectorEq] at holdsRow ⊢ <;>
      exact holdsRow
  · intro row member
    have holdsRow := baseHolds row member
    obtain ⟨index, rfl⟩ := (List.mem_ofFn ..).mp member
    simp only [control, Expr.eval_sub, Expr.eval_hmul, SlotSelect.eval_one, recursiveEq,
      initialEq index, currentEq index] at holdsRow ⊢
    exact holdsRow

theorem isSelected_congr (left right : Env)
    (selectorEq : (interface.selector offset).eval right = (interface.selector offset).eval left)
    (slot : Fin 2) :
    (SlotSelect.isSelected (control interface) offset slot).eval right =
      (SlotSelect.isSelected (control interface) offset slot).eval left := by
  unfold SlotSelect.isSelected
  split_ifs
  · simp only [control, Expr.eval_sub, SlotSelect.eval_one, selectorEq]
  · exact selectorEq

theorem ofFn_congr (left right : Env) (digest : Fin 4 → Expr)
    (same : digestValue digest right = digestValue digest left) :
    Hash.evalList right (List.ofFn digest) = Hash.evalList left (List.ofFn digest) := by
  rw [ofFn_eval, ofFn_eval, same]

/-- A slot's digest rows hold again on values that keep every word they read. -/
theorem digestRows_congr (left right : Env)
    (selectorEq : (interface.selector offset).eval right = (interface.selector offset).eval left)
    (recursiveEq : (interface.recursive offset).eval right = (interface.recursive offset).eval left)
    (slot : Fin 2)
    (priorDigestEq : digestValue (interface.priorDigest offset slot) right =
      digestValue (interface.priorDigest offset slot) left)
    (newDigestEq : digestValue (interface.newDigest offset) right =
      digestValue (interface.newDigest offset) left)
    (innerDigestEq : digestValue (interface.innerDigest offset slot) right =
      digestValue (interface.innerDigest offset slot) left)
    (nextDigestEq : digestValue (interface.nextDigest offset slot) right =
      digestValue (interface.nextDigest offset slot) left)
    (holds : ConstraintsHold left (digestRows interface offset slot)) :
    ConstraintsHold right (digestRows interface offset slot) := by
  have defaultEq : Hash.evalList right (defaultDigest (publicFits := publicFits)) =
      Hash.evalList left (defaultDigest (publicFits := publicFits)) := by
    rw [defaultDigest_eval, defaultDigest_eval]
  obtain ⟨innerHolds, nextHolds⟩ := (constraintsHold_append left _ _).mp holds
  exact (constraintsHold_append right _ _).mpr
    ⟨selectRows_congr left right _ (isSelected_congr interface offset left right selectorEq slot)
        _ _ _ (ofFn_congr left right _ priorDigestEq) (ofFn_congr left right _ newDigestEq)
        (ofFn_congr left right _ innerDigestEq) innerHolds,
      selectRows_congr left right _ recursiveEq _ _ _ defaultEq
        (ofFn_congr left right _ innerDigestEq) (ofFn_congr left right _ nextDigestEq) nextHolds⟩

/-- The rows hold again on values that keep every word they read. -/
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
    (priorDigestEq : ∀ slot, digestValue (interface.priorDigest offset slot) right =
      digestValue (interface.priorDigest offset slot) left)
    (newDigestEq : digestValue (interface.newDigest offset) right =
      digestValue (interface.newDigest offset) left)
    (innerDigestEq : ∀ slot, digestValue (interface.innerDigest offset slot) right =
      digestValue (interface.innerDigest offset slot) left)
    (nextDigestEq : ∀ slot, digestValue (interface.nextDigest offset slot) right =
      digestValue (interface.nextDigest offset slot) left)
    (holds : ConstraintsHold left (rows interface offset)) :
    ConstraintsHold right (rows interface offset) := by
  obtain ⟨front, digestOneHolds⟩ := (constraintsHold_append left _ _).mp holds
  obtain ⟨front, digestZeroHolds⟩ := (constraintsHold_append left _ _).mp front
  obtain ⟨controlHolds, activeHolds⟩ := (constraintsHold_append left _ _).mp front
  have digestHolds := fun slot => digestRows_congr interface offset left right selectorEq
    recursiveEq slot (priorDigestEq slot) newDigestEq (innerDigestEq slot) (nextDigestEq slot)
  refine (constraintsHold_append right _ _).mpr ⟨(constraintsHold_append right _ _).mpr
    ⟨(constraintsHold_append right _ _).mpr
      ⟨controlRows_congr interface offset left right iterationEq inverseEq recursiveEq pcEq
        selectorEq initialEq currentEq controlHolds, ?_⟩,
      digestHolds 0 digestZeroHolds⟩, digestHolds 1 digestOneHolds⟩
  exact selectRows_congr left right _ selectorEq _ _ _ (priorEq 0) (priorEq 1) activeEq
    activeHolds

end NightstreamFPrime.Lifecycle.BoundedDepth.DigestSelect
