import NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain

/-!
Owns the exact physical footprint composition of the 17-sampler PiRLC chain.

The logical operation spine stays opaque. This module proves that its rows are
the scalar child rows in `K + k` order and sums the certified scalar physical
footprints without evaluating the million-row chain.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.circuit
abbrev main :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.main
abbrev opsAt :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.opsAt
abbrev childOp :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.childOp
abbrev childInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.childInterface
abbrev stateAtExpr :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.stateAtExpr
abbrev sourceOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.sourceOffset
abbrev sourceCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount
abbrev logicalRowCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalRowCount
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.Assumptions
abbrev RelationHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.RelationHolds
abbrev soundness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.soundness
abbrev completeness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.localLength_eq
abbrev main_ops :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.main_ops
abbrev opsAt_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.opsAt_eq
abbrev flatConstraints_varsBelow_of_rows :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.flatConstraints_varsBelow_of_rows

end Logical

/-- The chain's only external expressions are its incoming transcript state. -/
structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

/-- Every scalar child receives affine state. The first state is external;
each successor is the preceding sampler's fresh final digest-window output. -/
def childInputs (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) (source : Nat) :
    ∀ current,
      Sampler.InputsAffine (Logical.childInterface interface offset source)
        current := by
  intro current
  refine ⟨?_⟩
  cases source with
  | zero =>
      simpa [Logical.childInterface, Logical.stateAtExpr] using
        inputs.initialState
  | succ previous =>
      simpa [Logical.childInterface, Logical.stateAtExpr] using
        (Sampler.outputState_fresh
          (Logical.childInterface interface offset previous) previous
          (Logical.sourceOffset offset previous)).affine

def childConstraints (interface : Logical.Interface) (offset source : Nat) :
    List Expr :=
  Sampler.logicalConstraints (Logical.childInterface interface offset source)
    source (Logical.sourceOffset offset source)

def childConstraintLists (interface : Logical.Interface) (offset : Nat) :
    List (List Expr) :=
  (List.range Logical.sourceCount).map
    (childConstraints interface offset)

/-- The exact physical-owner order is the logical `K + k` child order. -/
def orderedConstraints (interface : Logical.Interface) (offset : Nat) :
    List Expr :=
  (childConstraintLists interface offset).flatten

/-- The unchanged logical rows of the sole 17-sampler parent. -/
def logicalConstraints (interface : Logical.Interface) (offset : Nat) :
    List Expr :=
  flatConstraints (Circuit.ops (Logical.main interface) offset)

private theorem childOp_flatConstraints (interface : Logical.Interface)
    (offset source : Nat) :
    (Logical.childOp interface offset source).flatConstraints =
      childConstraints interface offset source := by
  rfl

private theorem flatConstraints_childOps (interface : Logical.Interface)
    (offset : Nat) (sources : List Nat) :
    flatConstraints (sources.map (Logical.childOp interface offset)) =
      (sources.map (childConstraints interface offset)).flatten := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.map_cons, flatConstraints, List.flatMap_cons,
        List.flatten_cons, childOp_flatConstraints]
      exact congrArg (fun tail =>
        childConstraints interface offset source ++ tail) inductionHypothesis

/-- Exact equality between the rows owned here and all 17 opaque scalar child
lists, in source order. -/
theorem logicalConstraints_eq_ordered (interface : Logical.Interface)
    (offset : Nat) :
    logicalConstraints interface offset =
      orderedConstraints interface offset := by
  unfold logicalConstraints
  rw [Logical.main_ops, Logical.opsAt_eq]
  unfold orderedConstraints childConstraintLists
  exact flatConstraints_childOps interface offset _

private theorem childFreshCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) (source : Nat) :
    R1CS.totalFreshCount (childConstraints interface offset source) =
      43743 := by
  exact Sampler.totalFreshCount_eq
    (Logical.childInterface interface offset source) source
    (Logical.sourceOffset offset source) (childInputs interface offset inputs source)

private theorem childRowCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) (source : Nat) :
    R1CS.totalRowCount (childConstraints interface offset source) =
      59344 := by
  exact Sampler.totalRowCount_eq
    (Logical.childInterface interface offset source) source
    (Logical.sourceOffset offset source) (childInputs interface offset inputs source)

private theorem totalFreshCount_sources (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (sources : List Nat) :
    R1CS.totalFreshCount
        ((sources.map (childConstraints interface offset)).flatten) =
      sources.length * 43743 := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.map_cons, List.flatten_cons,
        R1CS.totalFreshCount_append, childFreshCount_eq interface offset inputs,
        inductionHypothesis, List.length_cons]
      omega

private theorem totalRowCount_sources (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (sources : List Nat) :
    R1CS.totalRowCount
        ((sources.map (childConstraints interface offset)).flatten) =
      sources.length * 59344 := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.map_cons, List.flatten_cons,
        R1CS.totalRowCount_append, childRowCount_eq interface offset inputs,
        inductionHypothesis, List.length_cons]
      omega

/-- Exact R1CS multiplication-column count for 17 scalar samplers. -/
theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      743631 := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints childConstraintLists
  rw [totalFreshCount_sources interface offset inputs]
  simp only [List.length_range]
  change 17 * 43743 = 743631
  norm_num

/-- Exact physical-row count for 17 scalar samplers. -/
theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      1008848 := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints childConstraintLists
  rw [totalRowCount_sources interface offset inputs]
  simp only [List.length_range]
  change 17 * 59344 = 1008848
  norm_num

/-- Exact logical-plus-R1CS private-column count for the chain. -/
theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    localLength (Circuit.ops (Logical.main interface) offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) =
      1007199 := by
  rw [Logical.localLength_eq, totalFreshCount_eq interface offset inputs]
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount_eq]

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 743631
  physicalRowCount := fun _ => 1008848
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface offset (inputs offset)

end NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
