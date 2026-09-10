import NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.SignedSplitScalar
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

/-!
Owns the exact physical footprint composition of the 270-coordinate PiDEC
public-input split.

The logical operation spine stays opaque. This module proves that its rows are
the signed-scalar child rows in coordinate order and sums the certified child
footprints without evaluating all child constraints in the parent proof.
-/

namespace NightstreamFPrime.Layout.PiDEC.v1_1.PublicInputSplit

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.Interface
abbrev circuit :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.circuit
abbrev main :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.main
abbrev opsAt :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.opsAt
abbrev childOp :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childOp
abbrev childInterface :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childInterface
abbrev sourceOffset :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.sourceOffset
abbrev coordinateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.logicalPrivateCount
abbrev localLength_eq :=
  @NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.localLength_eq

abbrev exactCoordinateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.exactCoordinateCount
abbrev exactPrivateCount :=
  NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.exactPrivateCount

end Logical

/-- Coordinate multiplicity and scalar physical cost are separate from the
number of logical private cells used by each scalar child. -/
def freshColumnCount : Nat :=
  Logical.exactCoordinateCount * Leaves.SignedSplitScalar.freshColumnCount

def physicalRowCount : Nat :=
  Logical.exactCoordinateCount * Leaves.SignedSplitScalar.physicalRowCount

/-- Exact affine/nonconstant shape of the parent and child public-input
wires supplied by the PiDEC phase. -/
structure InputsLinear
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) : Prop where
  parent_mulCount : ∀ coordinate,
    R1CS.mulCount (interface.parent offset coordinate) = 0
  digit_mulCount : ∀ child coordinate,
    R1CS.mulCount (interface.digit offset child coordinate) = 0
  digit_nonconstant : ∀ child coordinate,
    Nonconstant (interface.digit offset child coordinate)

/-- Every coordinate child receives the same parent-owned affine wires. -/
def childInputs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (source : Nat) (sourceLt : source < Logical.coordinateCount logicalWidth publicFits) :
    ∀ current,
      Leaves.SignedSplitScalar.InputsLinear
        (Logical.childInterface interface offset source sourceLt) current := by
  intro current
  refine ⟨?_, ?_, ?_⟩
  · simpa [Logical.childInterface] using
      inputs.parent_mulCount ⟨source, sourceLt⟩
  · intro child
    simpa [Logical.childInterface] using
      inputs.digit_mulCount child ⟨source, sourceLt⟩
  · intro child
    simpa [Logical.childInterface] using
      inputs.digit_nonconstant child ⟨source, sourceLt⟩

def childConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset source : Nat) : List Expr :=
  (Logical.childOp interface offset source).flatConstraints

def childConstraintLists
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) : List (List Expr) :=
  (List.range (Logical.coordinateCount logicalWidth publicFits)).map
    (childConstraints interface offset)

def orderedConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  (childConstraintLists interface offset).flatten

def logicalConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (Logical.main interface) offset)

private theorem flatConstraints_childOps
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (sources : List Nat) :
    flatConstraints (sources.map (Logical.childOp interface offset)) =
      (sources.map (childConstraints interface offset)).flatten := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.map_cons, flatConstraints, List.flatMap_cons,
        List.flatten_cons, childConstraints]
      exact congrArg (fun tail =>
        (Logical.childOp interface offset source).flatConstraints ++ tail)
        inductionHypothesis

/-- Exact equality between parent rows and the 270 opaque scalar row lists. -/
theorem logicalConstraints_eq_ordered
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) :
    logicalConstraints interface offset = orderedConstraints interface offset := by
  unfold logicalConstraints
  change flatConstraints
      ((List.range (Logical.coordinateCount logicalWidth publicFits)).map
        (Logical.childOp interface offset)) = _
  unfold orderedConstraints childConstraintLists
  exact flatConstraints_childOps interface offset _

private theorem childFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (source : Nat) (sourceLt : source < Logical.coordinateCount logicalWidth publicFits) :
    R1CS.totalFreshCount (childConstraints interface offset source) = Leaves.SignedSplitScalar.freshColumnCount := by
  unfold childConstraints
  have childEq : Logical.childOp interface offset source =
      NightstreamFPrime.Circuit.Sequence.childOp
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childName source)
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit
          (Logical.childInterface interface offset source sourceLt))
        (Logical.sourceOffset offset source) := by
    simp [Logical.childOp,
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childOp,
      sourceLt]
  rw [childEq]
  exact Leaves.SignedSplitScalar.freshColumnCount_eq
    (Logical.childInterface interface offset source sourceLt)
    (childInputs interface offset inputs source sourceLt)
    (Logical.sourceOffset offset source)

private theorem childRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (source : Nat) (sourceLt : source < Logical.coordinateCount logicalWidth publicFits) :
    R1CS.totalRowCount (childConstraints interface offset source) = Leaves.SignedSplitScalar.physicalRowCount := by
  unfold childConstraints
  have childEq : Logical.childOp interface offset source =
      NightstreamFPrime.Circuit.Sequence.childOp
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childName source)
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar.circuit
          (Logical.childInterface interface offset source sourceLt))
        (Logical.sourceOffset offset source) := by
    simp [Logical.childOp,
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.childOp,
      sourceLt]
  rw [childEq]
  exact Leaves.SignedSplitScalar.physicalRowCount_eq
    (Logical.childInterface interface offset source sourceLt)
    (childInputs interface offset inputs source sourceLt)
    (Logical.sourceOffset offset source)

private theorem totalFreshCount_sources
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (sources : List Nat)
    (bounded : ∀ source ∈ sources,
      source < Logical.coordinateCount logicalWidth publicFits) :
    R1CS.totalFreshCount
        ((sources.map (childConstraints interface offset)).flatten) =
      sources.length * Leaves.SignedSplitScalar.freshColumnCount := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      have sourceLt := bounded source (by simp)
      have restBounded : ∀ current ∈ rest,
          current < Logical.coordinateCount logicalWidth publicFits := by
        intro current member
        exact bounded current (by simp [member])
      simp only [List.map_cons, List.flatten_cons,
        R1CS.totalFreshCount_append,
        childFreshCount_eq interface offset inputs source sourceLt,
        inductionHypothesis restBounded, List.length_cons]
      simp only [Nat.succ_mul, Nat.add_comm]

private theorem totalRowCount_sources
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset)
    (sources : List Nat)
    (bounded : ∀ source ∈ sources,
      source < Logical.coordinateCount logicalWidth publicFits) :
    R1CS.totalRowCount
        ((sources.map (childConstraints interface offset)).flatten) =
      sources.length * Leaves.SignedSplitScalar.physicalRowCount := by
  induction sources with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      have sourceLt := bounded source (by simp)
      have restBounded : ∀ current ∈ rest,
          current < Logical.coordinateCount logicalWidth publicFits := by
        intro current member
        exact bounded current (by simp [member])
      simp only [List.map_cons, List.flatten_cons,
        R1CS.totalRowCount_append,
        childRowCount_eq interface offset inputs source sourceLt,
        inductionHypothesis restBounded, List.length_cons]
      simp only [Nat.succ_mul, Nat.add_comm]

/-- Exact fresh count from the ordered coordinate children. -/
theorem totalFreshCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) =
      freshColumnCount := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints childConstraintLists
  rw [totalFreshCount_sources interface offset inputs]
  · rw [List.length_range]
    change NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
        logicalWidth publicFits * Leaves.SignedSplitScalar.freshColumnCount =
      Logical.exactCoordinateCount * Leaves.SignedSplitScalar.freshColumnCount
    rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq_exact]
  · intro source member
    exact List.mem_range.mp member

/-- Exact physical rows from the same ordered coordinate children. -/
theorem totalRowCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) =
      physicalRowCount := by
  rw [logicalConstraints_eq_ordered]
  unfold orderedConstraints childConstraintLists
  rw [totalRowCount_sources interface offset inputs]
  · rw [List.length_range]
    change NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
        logicalWidth publicFits * Leaves.SignedSplitScalar.physicalRowCount =
      Logical.exactCoordinateCount * Leaves.SignedSplitScalar.physicalRowCount
    rw [NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq_exact]
  · intro source member
    exact List.mem_range.mp member

/-- Logical cells and R1CS intermediates have distinct owner counts. -/
theorem physicalPrivateColumnCount_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    localLength (Circuit.ops (Logical.main interface) offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) =
      Logical.exactPrivateCount + freshColumnCount := by
  rw [Logical.localLength_eq, totalFreshCount_eq interface offset inputs,
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.logicalPrivateCount_eq]

def footprint
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Logical.Interface logicalWidth publicFits)
    (inputs : ∀ offset, InputsLinear interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => freshColumnCount
  physicalRowCount := fun _ => physicalRowCount
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface offset (inputs offset)

end NightstreamFPrime.Layout.PiDEC.v1_1.PublicInputSplit
