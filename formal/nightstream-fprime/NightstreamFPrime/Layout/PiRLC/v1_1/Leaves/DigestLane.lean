import NightstreamFPrime.Layout.Range.CanonicalU64
import NightstreamFPrime.Layout.Sampling.Candidate16Five
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane

/-!
Owns physical lowering for one PiRLC sampler rate lane.

The logical parent contains one canonical-u64 child and two candidate-decoder
children. Their flattened rows lower in the same order. This owner adds no
boundary-copy row.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

namespace Logical

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.Interface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.circuit
abbrev canonicalInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalInterface
abbrev canonicalOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.canonicalOffset
abbrev decoderOffset :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.decoderOffset
abbrev lowPart :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.lowPart
abbrev highPart :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.highPart
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.SpecHolds
abbrev soundness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.soundness
abbrev completeness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.localLength_eq
abbrev flatConstraints_varsBelow :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.flatConstraints_varsBelow
abbrev flatConstraints_opsAt :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.flatConstraints_opsAt
abbrev logicalPrivateCount :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.logicalPrivateCount

end Logical

structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  source : R1CS.IsAffine (interface.source offset)

private def canonicalInputs (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    NightstreamFPrime.Layout.Range.CanonicalU64.InputsAffine
      (Logical.canonicalInterface interface offset)
      (Logical.canonicalOffset offset) where
  source := by
    simpa [Logical.canonicalInterface, Logical.canonicalOffset] using inputs.source

def logicalConstraints (interface : Logical.Interface) (offset : Nat) :
    List Expr :=
  flatConstraints (Circuit.ops (Logical.circuit interface).main offset)

theorem totalFreshCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 303 := by
  unfold logicalConstraints
  change R1CS.totalFreshCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.opsAt
      interface offset)) = 303
  rw [Logical.flatConstraints_opsAt,
    R1CS.totalFreshCount_append, R1CS.totalFreshCount_append]
  change R1CS.totalFreshCount
        (NightstreamFPrime.Layout.Range.CanonicalU64.logicalConstraints
          (Logical.canonicalInterface interface offset)
          (Logical.canonicalOffset offset)) +
      R1CS.totalFreshCount
        (NightstreamFPrime.Layout.Sampling.Candidate16Five.logicalConstraints
          (Logical.canonicalOffset offset) Logical.lowPart
          (Logical.decoderOffset offset Logical.lowPart)) +
      R1CS.totalFreshCount
        (NightstreamFPrime.Layout.Sampling.Candidate16Five.logicalConstraints
          (Logical.canonicalOffset offset) Logical.highPart
          (Logical.decoderOffset offset Logical.highPart)) = 303
  rw [NightstreamFPrime.Layout.Range.CanonicalU64.totalFreshCount_eq
      (Logical.canonicalInterface interface offset)
      (Logical.canonicalOffset offset) (canonicalInputs interface offset inputs),
    NightstreamFPrime.Layout.Sampling.Candidate16Five.totalFreshCount_eq,
    NightstreamFPrime.Layout.Sampling.Candidate16Five.totalFreshCount_eq]

theorem totalRowCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 406 := by
  unfold logicalConstraints
  change R1CS.totalRowCount (flatConstraints
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.opsAt
      interface offset)) = 406
  rw [Logical.flatConstraints_opsAt,
    R1CS.totalRowCount_append, R1CS.totalRowCount_append]
  change R1CS.totalRowCount
        (NightstreamFPrime.Layout.Range.CanonicalU64.logicalConstraints
          (Logical.canonicalInterface interface offset)
          (Logical.canonicalOffset offset)) +
      R1CS.totalRowCount
        (NightstreamFPrime.Layout.Sampling.Candidate16Five.logicalConstraints
          (Logical.canonicalOffset offset) Logical.lowPart
          (Logical.decoderOffset offset Logical.lowPart)) +
      R1CS.totalRowCount
        (NightstreamFPrime.Layout.Sampling.Candidate16Five.logicalConstraints
          (Logical.canonicalOffset offset) Logical.highPart
          (Logical.decoderOffset offset Logical.highPart)) = 406
  rw [NightstreamFPrime.Layout.Range.CanonicalU64.totalRowCount_eq
      (Logical.canonicalInterface interface offset)
      (Logical.canonicalOffset offset) (canonicalInputs interface offset inputs),
    NightstreamFPrime.Layout.Sampling.Candidate16Five.totalRowCount_eq,
    NightstreamFPrime.Layout.Sampling.Candidate16Five.totalRowCount_eq]

def footprint (interface : Logical.Interface)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Logical.circuit interface) where
  freshColumnCount := fun _ => 303
  physicalRowCount := fun _ => 406
  freshColumnCount_eq := fun offset =>
    totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := fun offset =>
    totalRowCount_eq interface offset (inputs offset)

theorem physicalPrivateColumnCount_eq (interface : Logical.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    localLength (Circuit.ops (Logical.circuit interface).main offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) = 403 := by
  have lengthEq : localLength
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.opsAt
        interface offset) = Logical.logicalPrivateCount := by
    simpa using Logical.localLength_eq interface offset
  change localLength
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestLane.opsAt
        interface offset) +
      R1CS.totalFreshCount (logicalConstraints interface offset) = 403
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

theorem physical_complete (interface : Logical.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Logical.Assumptions interface offset env)
    (specification : Logical.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 403 ∧
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

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane
