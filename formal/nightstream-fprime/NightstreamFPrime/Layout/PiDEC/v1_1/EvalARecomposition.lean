import NightstreamFPrime.Layout.PiDEC.v1_1.RingKRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition

/-!
Owns the physical footprint for all 14 separate PiDEC `Eval_A` matrix
families. Matrix order, coefficient order, and `c0, c1` cell order are the
logical circuit's order; this module adds no compression or copy row.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.EvalARecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.circuit
abbrev ringInterface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.ringInterface
abbrev blockCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.blockCount
abbrev coefficient :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition.coefficient
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.localLength_eq

end Logical

structure InputsLinear (interface : Logical.Interface) (offset : Nat) : Prop where
  parent : ∀ matrix coefficient,
    PiDEC.v1_1.RingKRecomposition.ValueLinear
      (interface.parent offset matrix coefficient)
  child : ∀ child matrix coefficient,
    PiDEC.v1_1.RingKRecomposition.ValueLinear
      (interface.child offset child matrix coefficient)

def ringInputs (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    ∀ offset,
      PiDEC.v1_1.RingKRecomposition.InputsLinear
        (Logical.ringInterface interface) offset := by
  intro offset
  refine ⟨?_, ?_⟩
  · intro matrix lane
    exact inputs offset |>.parent matrix (Logical.coefficient lane)
  · intro child matrix lane
    exact inputs offset |>.child child matrix (Logical.coefficient lane)

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 1512
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
      _ = 1512 :=
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition.coordinateCount_eq

theorem freshColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 1512 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 := by
  rw [Logical.localLength_eq, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiDEC.v1_1.EvalARecomposition
