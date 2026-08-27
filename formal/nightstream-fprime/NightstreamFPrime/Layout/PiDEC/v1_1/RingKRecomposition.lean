import NightstreamFPrime.Layout.PiDEC.v1_1.RadixRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition

/-!
Owns typed `c0, c1` physical lowering for the reusable PiDEC `RingK`
recomposition family. It preserves the exact block, ring-lane, and extension
cell order selected by the logical circuit.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.RingKRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.Interface
abbrev circuit :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.circuit
abbrev scalarInterface :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.scalarInterface
abbrev coordinates :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.coordinates
abbrev coordinateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.coordinateCount
abbrev expressionCell :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.expressionCell
abbrev cellCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.cellCount
abbrev localLength_eq :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition.localLength_eq

end Logical

/-- Both extension-field cells contain no multiplication node. -/
structure ValueLinear (value : KExpr) : Prop where
  c0_mulCount : R1CS.mulCount value.c0 = 0
  c1_mulCount : R1CS.mulCount value.c1 = 0

structure InputsLinear {blockCount : Nat}
    (interface : Logical.Interface blockCount) (offset : Nat) : Prop where
  parent : ∀ block lane, ValueLinear (interface.parent offset block lane)
  child : ∀ child block lane,
    ValueLinear (interface.child offset child block lane)

theorem expressionCell_mulCount (cell : Fin Logical.cellCount)
    (value : KExpr) (linear : ValueLinear value) :
    R1CS.mulCount (Logical.expressionCell cell value) = 0 := by
  fin_cases cell
  · simpa [Logical.expressionCell, Logical.cellCount] using linear.c0_mulCount
  · simpa [Logical.expressionCell, Logical.cellCount] using linear.c1_mulCount

def scalarInputs {blockCount : Nat}
    (interface : Logical.Interface blockCount)
    (inputs : ∀ offset, InputsLinear interface offset) :
    ∀ offset,
      PiDEC.v1_1.RadixRecomposition.InputsLinear
        (Logical.scalarInterface interface) offset := by
  intro offset
  refine ⟨?_, ?_⟩
  · intro coordinate
    simpa [Logical.scalarInterface] using
      expressionCell_mulCount (Logical.coordinates coordinate).2.2
        (interface.parent offset (Logical.coordinates coordinate).1
          (Logical.coordinates coordinate).2.1)
        (inputs offset |>.parent (Logical.coordinates coordinate).1
          (Logical.coordinates coordinate).2.1)
  · intro child coordinate
    simpa [Logical.scalarInterface] using
      expressionCell_mulCount (Logical.coordinates coordinate).2.2
        (interface.child offset child (Logical.coordinates coordinate).1
          (Logical.coordinates coordinate).2.1)
        (inputs offset |>.child child (Logical.coordinates coordinate).1
          (Logical.coordinates coordinate).2.1)

def footprint {blockCount : Nat}
    (interface : Logical.Interface blockCount)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) :=
  PiDEC.v1_1.RadixRecomposition.footprint
    (Logical.scalarInterface interface) (scalarInputs interface inputs)

theorem freshColumnCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalFreshCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    R1CS.totalRowCount
      (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) =
      Logical.coordinateCount blockCount :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq {blockCount : Nat}
    (interface : Logical.Interface blockCount)
    (inputs : ∀ offset, InputsLinear interface offset) (offset : Nat) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount
        (flatConstraints (Circuit.ops (Logical.circuit interface).main offset)) = 0 := by
  rw [Logical.localLength_eq, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiDEC.v1_1.RingKRecomposition
