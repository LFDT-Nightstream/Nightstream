import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane
import NightstreamFPrime.Layout.Poseidon2.PermutationOwned
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow

/-!
Owns physical lowering for one complete PiRLC sampler digest block.

Four physical lane children are followed by one direct owned Poseidon2
permutation. The physical theorem reaches the concrete Lean `digestBlock`
relation; backend acceptance is not used.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.circuit
abbrev laneInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneInterface
abbrev laneOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneOffset
abbrev permutationInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.permutationInterface
abbrev permutationOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.permutationOffset
abbrev rateLane :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.rateLane
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.SpecHolds
abbrev soundness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.soundness
abbrev completeness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.localLength_eq
abbrev flatConstraints_varsBelow :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.flatConstraints_varsBelow
abbrev flatConstraints_opsAt :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.flatConstraints_opsAt
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.logicalPrivateCount
abbrev parentCoverage :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.parentCoverage

end Logical

structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  initialState : ∀ lane, R1CS.IsAffine (interface.initialState offset lane)

private def laneInputs (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) (lane : Fin 4) :
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.InputsAffine
      (Logical.laneInterface interface offset lane)
      (Logical.laneOffset offset lane) where
  source := by
    simpa [Logical.laneInterface] using inputs.initialState (Logical.rateLane lane)

private def permutationInputs (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    NightstreamFPrime.Layout.Poseidon2.PermutationOwned.InputsAffine
      (Logical.permutationInterface interface offset)
      (Logical.permutationOffset offset) where
  initialState := by
    intro lane
    simpa [Logical.permutationInterface] using inputs.initialState lane

def logicalConstraints (interface : Logical.Interface) (offset : Nat) :
    List Expr :=
  flatConstraints (Circuit.ops (Logical.circuit interface).main offset)

theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 1212 := by
  unfold logicalConstraints
  change R1CS.totalFreshCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.opsAt
      interface offset)) = 1212
  rw [Logical.flatConstraints_opsAt,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append]
  change
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 0)
          (Logical.laneOffset offset 0)) +
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 1)
          (Logical.laneOffset offset 1)) +
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 2)
          (Logical.laneOffset offset 2)) +
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 3)
          (Logical.laneOffset offset 3)) +
    R1CS.totalFreshCount
        (NightstreamFPrime.Layout.Poseidon2.PermutationOwned.logicalConstraints
          (Logical.permutationInterface interface offset)
          (Logical.permutationOffset offset)) = 1212
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      _ _ (laneInputs interface offset inputs 0),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      _ _ (laneInputs interface offset inputs 1),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      _ _ (laneInputs interface offset inputs 2),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalFreshCount_eq
      _ _ (laneInputs interface offset inputs 3),
    NightstreamFPrime.Layout.Poseidon2.PermutationOwned.totalFreshCount_eq
      _ _ (permutationInputs interface offset inputs)]

theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 2216 := by
  unfold logicalConstraints
  change R1CS.totalRowCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.opsAt
      interface offset)) = 2216
  rw [Logical.flatConstraints_opsAt,
    R1CS.totalRowCount_append, R1CS.totalRowCount_append,
    R1CS.totalRowCount_append, R1CS.totalRowCount_append]
  change
    R1CS.totalRowCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 0)
          (Logical.laneOffset offset 0)) +
    R1CS.totalRowCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 1)
          (Logical.laneOffset offset 1)) +
    R1CS.totalRowCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 2)
          (Logical.laneOffset offset 2)) +
    R1CS.totalRowCount
        (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
          (Logical.laneInterface interface offset 3)
          (Logical.laneOffset offset 3)) +
    R1CS.totalRowCount
        (NightstreamFPrime.Layout.Poseidon2.PermutationOwned.logicalConstraints
          (Logical.permutationInterface interface offset)
          (Logical.permutationOffset offset)) = 2216
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalRowCount_eq
      _ _ (laneInputs interface offset inputs 0),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalRowCount_eq
      _ _ (laneInputs interface offset inputs 1),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalRowCount_eq
      _ _ (laneInputs interface offset inputs 2),
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalRowCount_eq
      _ _ (laneInputs interface offset inputs 3),
    NightstreamFPrime.Layout.Poseidon2.PermutationOwned.totalRowCount_eq
      _ _ (permutationInputs interface offset inputs)]

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 1212
  physicalRowCount := fun _ => 2216
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface offset (inputs offset)

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) = 2204 := by
  have lengthEq : localLength
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.opsAt
        interface offset) = Logical.logicalPrivateCount := by
    simpa using Logical.localLength_eq interface offset
  change localLength
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.opsAt
        interface offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) = 2204
  rw [lengthEq, totalFreshCount_eq interface offset inputs]
  rfl

def plan (interface : Logical.Interface) (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + Logical.logicalPrivateCount

def PhysicalHolds (interface : Logical.Interface) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface offset).rows

theorem physical_implies_spec (interface : Logical.Interface) (offset : Nat)
    (env : Env) (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Logical.SpecHolds interface offset env := by
  apply Logical.soundness interface env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact R1CS.LoweringPlan.sound (plan interface offset) env physical

theorem physical_implies_digestBlock (interface : Logical.Interface)
    (offset : Nat) (env : Env)
    (assumptions : Logical.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) (counter : Nat) :
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.evalState env
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.output
          interface offset),
      fun position =>
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.evalChunk
          env offset position) =
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.digestBlock
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.evalState env
          (interface.initialState offset)) counter := by
  exact Logical.parentCoverage interface offset env
    (physical_implies_spec interface offset env assumptions physical) counter

set_option maxRecDepth 100000 in -- fixed-size: one four-lane digest window, not artifact data
theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 2204 ∧
      PhysicalHolds interface offset completed := by
  rcases Logical.completeness interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset Logical.logicalPrivateCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have scope : ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) := by
    exact Logical.flatConstraints_varsBelow interface offset assumptions
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset)
      (offset + Logical.logicalPrivateCount) scope logicalRows with
    ⟨completed, physicalAgrees, rows⟩
  refine ⟨completed, ?_, rows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [totalFreshCount_eq interface offset inputs] at combined
  simpa [Logical.logicalPrivateCount] using combined

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindow
