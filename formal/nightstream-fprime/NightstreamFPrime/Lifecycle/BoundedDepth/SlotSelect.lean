import NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
import NightstreamFPrime.Lifecycle.BoundedDepth.Step
import NightstreamFPrime.Spec.FieldPrime

/-! The control rows that the digest select (`DigestSelect`) and the light select (`LightSelect`)
share. They allocate no variables: the step layout supplies the iteration word with its inverse
and recursive flag, the prior-circuit selector, the state words and the slot words. The flag rows
are those of `Stage1.RunningTransition` (`r = iteration · inverse`, `iteration · (1 - r) = 0`);
then a boolean selector `s` and the prior pc `r · (pc - s - 1) = 0` (`flagRows`, `flag_facts`),
and the base state `(1 - r) · (z0 - zi) = 0` (`baseStateRows`). -/

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

/-- Below the modulus, a count's word is 0 exactly when the count is 0. -/
theorem natWord_eq_zero {count : Nat} (bound : count < fieldModulus) :
    natWord count = 0 ↔ count = 0 := by
  constructor
  · intro zero
    exact natWord_injective bound (by decide) (zero.trans rfl)
  · rintro rfl
    rfl

end NightstreamFPrime.Lifecycle.BoundedDepth.SlotSelect
