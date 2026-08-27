import NightstreamFPrime.Layout.PiDEC.v1_1.RingKRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition

/-!
Owns the separate Pad `Eval_K` PiDEC physical footprint: one 54-coefficient
ring, two extension-field cells per coefficient, and no matrix-zero encoding.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.EvalKRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.circuit
abbrev ringInterface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.ringInterface
abbrev coefficient :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.coefficient
abbrev blockCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.blockCount
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.localLength_eq

end Logical

structure InputsLinear (interface : Logical.Interface) (offset : Nat) : Prop where
  parent : ∀ coefficient,
    PiDEC.v1_1.RingKRecomposition.ValueLinear
      (interface.parent offset coefficient)
  child : ∀ child coefficient,
    PiDEC.v1_1.RingKRecomposition.ValueLinear
      (interface.child offset child coefficient)

def ringInputs (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    ∀ offset,
      PiDEC.v1_1.RingKRecomposition.InputsLinear
        (Logical.ringInterface interface) offset := by
  intro offset
  refine ⟨?_, ?_⟩
  · intro _ lane
    exact inputs offset |>.parent (Logical.coefficient lane)
  · intro child _ lane
    exact inputs offset |>.child child (Logical.coefficient lane)

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 108
  freshColumnCount_eq := by
    intro offset
    exact PiDEC.v1_1.RingKRecomposition.freshColumnCount_eq
      (Logical.ringInterface interface) (ringInputs interface inputs) offset
  physicalRowCount_eq := by
    intro offset
    calc
      R1CS.totalRowCount
          (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) =
          NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.coordinateCount
            Logical.blockCount :=
        PiDEC.v1_1.RingKRecomposition.physicalRowCount_eq
          (Logical.ringInterface interface) (ringInputs interface inputs) offset
      _ = 108 :=
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.coordinateCount_eq

theorem freshColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 108 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 := by
  rw [Logical.localLength_eq, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiDEC.v1_1.EvalKRecomposition
