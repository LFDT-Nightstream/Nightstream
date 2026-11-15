import NightstreamFPrime.Lifecycle.BoundedDepth.DigestSelect

/-! The light step circuit's digest select. The light circuit hashes only the active slot's words,
into the active digest, and reads the other prior slot only as its digest word (HyperNova §6.3,
the optimization after Construction 2). The rows are `DigestSelect`'s flag, base-state and digest
rows without its word select, and four prior-digest rows per slot: the selected slot's prior
digest is the active digest. That is 8 + 2 × (4 + 8) = 32 rows.

`rows_implies_select` gives `SelectSpec`, the prior digests after the base case and the digest
of each next slot, when the other slot's prior digest word is the digest of the other prior slot.
`next_other` gives, from the rows alone, the other slot's next digest after the base case: its
prior digest word. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightSelect

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- `DigestSelect`'s words and the active slot's digest. -/
structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) extends
    DigestSelect.Interface logicalWidth publicFits where
  activeDigest : Nat → Fin 4 → Expr

variable (interface : Interface logicalWidth publicFits) (offset : Nat)

/-- The selected slot's prior digest is the active digest; the other slot's rows hold on any
words. -/
def priorRows (slot : Fin 2) : List Expr :=
  selectRows (SlotSelect.isSelected (DigestSelect.control interface.toInterface) offset slot)
    (List.ofFn (interface.priorDigest offset slot)) (List.ofFn (interface.activeDigest offset))
    (List.ofFn (interface.priorDigest offset slot))

def rows : List Expr :=
  SlotSelect.flagRows (DigestSelect.control interface.toInterface) offset ++
    SlotSelect.baseStateRows (DigestSelect.control interface.toInterface) offset ++
    priorRows interface offset 0 ++ priorRows interface offset 1 ++
    DigestSelect.digestRows interface.toInterface offset 0 ++
    DigestSelect.digestRows interface.toInterface offset 1

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
abbrev selected (env : Env) : Fin 2 := DigestSelect.selected interface.toInterface offset env

/-- A selected slot's prior-digest rows make its prior digest the active digest. -/
theorem priorRows_eval (env : Env) (slot : Fin 2)
    (holds : ConstraintsHold env (priorRows interface offset slot))
    (isSelected :
      (SlotSelect.isSelected (DigestSelect.control interface.toInterface) offset slot).eval env =
        1) :
    digestValue (interface.priorDigest offset slot) env =
      digestValue (interface.activeDigest offset) env := by
  have select := selectRows_eval env _ _ _ _ (by simp) (by simp) holds
  simp only [DigestSelect.ofFn_eval] at select
  exact select.2 isSelected

private theorem split (env : Env) (holds : ConstraintsHold env (rows interface offset)) :
    ConstraintsHold env (SlotSelect.flagRows (DigestSelect.control interface.toInterface) offset) ∧
      ConstraintsHold env
        (SlotSelect.baseStateRows (DigestSelect.control interface.toInterface) offset) ∧
      (∀ slot, ConstraintsHold env (priorRows interface offset slot)) ∧
      ∀ slot, ConstraintsHold env (DigestSelect.digestRows interface.toInterface offset slot) := by
  obtain ⟨front, digestOne⟩ := (constraintsHold_append env _ _).mp holds
  obtain ⟨front, digestZero⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨front, priorOne⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨front, priorZero⟩ := (constraintsHold_append env _ _).mp front
  obtain ⟨flag, base⟩ := (constraintsHold_append env _ _).mp front
  refine ⟨flag, base, fun slot => ?_, fun slot => ?_⟩
  · match slot with
    | 0 => exact priorZero
    | 1 => exact priorOne
  · match slot with
    | 0 => exact digestZero
    | 1 => exact digestOne

/-- The rows, with the words' values, the active digest of the selected prior slot and, after the
base case, the other slot's prior digest word as the other prior slot's digest, give `SelectSpec`
for `nextSlots`, every prior digest after the base case and the digest of each next slot. -/
theorem rows_implies_select (env : Env) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits) (new : SlotData logicalWidth publicFits)
    (holds : ConstraintsHold env (rows interface offset))
    (iterationWord : (interface.iteration offset).eval env = natWord input.iteration)
    (iterationBound : input.iteration < fieldModulus)
    (pcWord : (interface.priorPc offset).eval env = natWord input.priorPc)
    (pcBound : input.priorPc < fieldModulus)
    (initialWords :
      List.ofFn (fun index => (interface.initialState offset index).eval env) = input.z0)
    (currentWords :
      List.ofFn (fun index => (interface.currentState offset index).eval env) = input.zi)
    (activeDigest : digestValue (interface.activeDigest offset) env =
      slotDigest (input.running (selected interface offset env)))
    (otherDigests : input.iteration ≠ 0 → ∀ slot, slot ≠ selected interface offset env →
      digestValue (interface.priorDigest offset slot) env = slotDigest (input.running slot))
    (newDigest : digestValue (interface.newDigest offset) env = slotDigest new)
    (next : output.runningNext =
      DigestSelect.nextSlots input.iteration input.running (selected interface offset env) new) :
    SelectSpec input output (selected interface offset env) new.1 new.2 ∧
      (input.iteration ≠ 0 → ∀ slot,
        digestValue (interface.priorDigest offset slot) env = slotDigest (input.running slot)) ∧
      ∀ slot, digestValue (interface.nextDigest offset slot) env =
        slotDigest (output.runningNext slot) := by
  obtain ⟨flagHolds, baseHolds, priorHolds, digestHolds⟩ := split interface offset env holds
  obtain ⟨selectorBit, base, recursive⟩ := DigestSelect.control_facts interface.toInterface offset
    env input flagHolds baseHolds iterationWord iterationBound pcWord pcBound initialWords
    currentWords
  have priorDigests : input.iteration ≠ 0 → ∀ slot,
      digestValue (interface.priorDigest offset slot) env = slotDigest (input.running slot) := by
    intro nonzero slot
    by_cases same : slot = selected interface offset env
    · rw [priorRows_eval interface offset env slot (priorHolds slot)
        ((DigestSelect.selector_facts interface.toInterface offset env selectorBit slot).1 same),
        activeDigest, ← same]
    · exact otherDigests nonzero slot same
  obtain ⟨select, nextDigests⟩ := DigestSelect.select_of_digestRows interface.toInterface offset
    env input output new selectorBit base recursive digestHolds priorDigests newDigest next
  exact ⟨select, priorDigests, nextDigests⟩

/-- After the base case, the rows alone make the other slot's next digest its prior digest
word. -/
theorem next_other (env : Env) (holds : ConstraintsHold env (rows interface offset))
    (nonzero : (interface.iteration offset).eval env ≠ 0) (slot : Fin 2)
    (other : slot ≠ selected interface offset env) :
    digestValue (interface.nextDigest offset slot) env =
      digestValue (interface.priorDigest offset slot) env := by
  obtain ⟨flagHolds, _, _, digestHolds⟩ := split interface offset env holds
  obtain ⟨_, oneFlag, selectorBit, _⟩ :
      ((interface.iteration offset).eval env = 0 → (interface.recursive offset).eval env = 0) ∧
        ((interface.iteration offset).eval env ≠ 0 →
          (interface.recursive offset).eval env = 1) ∧
        ((interface.selector offset).eval env = 0 ∨ (interface.selector offset).eval env = 1) ∧
        ((interface.recursive offset).eval env = 1 →
          (interface.priorPc offset).eval env = (interface.selector offset).eval env + 1) :=
    SlotSelect.flag_facts (DigestSelect.control interface.toInterface) offset env flagHolds
  obtain ⟨innerPrior, _, _, nextInner⟩ :=
    DigestSelect.digestRows_eval interface.toInterface offset env slot (digestHolds slot)
  rw [nextInner (oneFlag nonzero), innerPrior
    ((DigestSelect.selector_facts interface.toInterface offset env selectorBit slot).2 other)]

/-- The converse of `rows_implies_select`: words with the honest flags, the active digest as the
selected slot's prior digest and the honest digest selections satisfy every row. -/
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
    (priorSelected :
      digestValue (interface.priorDigest offset (selected interface offset env)) env =
        digestValue (interface.activeDigest offset) env)
    (innerSelected :
      digestValue (interface.innerDigest offset (selected interface offset env)) env =
        digestValue (interface.newDigest offset) env)
    (innerOther : ∀ slot, slot ≠ selected interface offset env →
      digestValue (interface.innerDigest offset slot) env =
        digestValue (interface.priorDigest offset slot) env)
    (nextBase : (interface.recursive offset).eval env = 0 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        slotDigest (DigestSelect.defaultData (logicalWidth := logicalWidth)
          (publicFits := publicFits)))
    (nextRecursive : (interface.recursive offset).eval env = 1 → ∀ slot,
      digestValue (interface.nextDigest offset slot) env =
        digestValue (interface.innerDigest offset slot) env) :
    ConstraintsHold env (rows interface offset) := by
  have recursiveBit : (interface.recursive offset).eval env = 0 ∨
      (interface.recursive offset).eval env = 1 := by
    rcases flags with ⟨_, zero⟩ | ⟨_, one⟩
    · exact Or.inl zero
    · exact Or.inr one
  have digestHolds := DigestSelect.digestRows_holds interface.toInterface offset env selectorBit
    recursiveBit innerSelected innerOther nextBase nextRecursive
  have priorHolds : ∀ slot, ConstraintsHold env (priorRows interface offset slot) := by
    intro slot
    obtain ⟨isSelectedOne, isSelectedZero⟩ :=
      DigestSelect.selector_facts interface.toInterface offset env selectorBit slot
    apply selectRows_holds env _ _ _ _ (by simp) (by simp)
    by_cases same : slot = selected interface offset env
    · refine Or.inr ⟨isSelectedOne same, ?_⟩
      rw [DigestSelect.ofFn_eval, DigestSelect.ofFn_eval, same]
      exact priorSelected
    · exact Or.inl ⟨isSelectedZero same, rfl⟩
  exact (constraintsHold_append env _ _).mpr ⟨(constraintsHold_append env _ _).mpr
    ⟨(constraintsHold_append env _ _).mpr ⟨(constraintsHold_append env _ _).mpr
      ⟨DigestSelect.controlRows_holds interface.toInterface offset env flags selectorBit pcLink
        baseState, priorHolds 0⟩, priorHolds 1⟩, digestHolds 0⟩, digestHolds 1⟩

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
    (activeDigestEq : digestValue (interface.activeDigest offset) right =
      digestValue (interface.activeDigest offset) left)
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
  obtain ⟨front, digestOne⟩ := (constraintsHold_append left _ _).mp holds
  obtain ⟨front, digestZero⟩ := (constraintsHold_append left _ _).mp front
  obtain ⟨front, priorOne⟩ := (constraintsHold_append left _ _).mp front
  obtain ⟨control, priorZero⟩ := (constraintsHold_append left _ _).mp front
  have digest := fun slot => DigestSelect.digestRows_congr interface.toInterface offset left right
    selectorEq recursiveEq slot (priorDigestEq slot) newDigestEq (innerDigestEq slot)
    (nextDigestEq slot)
  have prior := fun slot => selectRows_congr left right _
    (DigestSelect.isSelected_congr interface.toInterface offset left right selectorEq slot) _ _ _
    (DigestSelect.ofFn_congr left right _ (priorDigestEq slot))
    (DigestSelect.ofFn_congr left right _ activeDigestEq)
    (DigestSelect.ofFn_congr left right _ (priorDigestEq slot))
  exact (constraintsHold_append right _ _).mpr ⟨(constraintsHold_append right _ _).mpr
    ⟨(constraintsHold_append right _ _).mpr ⟨(constraintsHold_append right _ _).mpr
      ⟨DigestSelect.controlRows_congr interface.toInterface offset left right iterationEq
        inverseEq recursiveEq pcEq selectorEq initialEq currentEq control, prior 0 priorZero⟩,
      prior 1 priorOne⟩, digest 0 digestZero⟩, digest 1 digestOne⟩

end NightstreamFPrime.Lifecycle.BoundedDepth.LightSelect
