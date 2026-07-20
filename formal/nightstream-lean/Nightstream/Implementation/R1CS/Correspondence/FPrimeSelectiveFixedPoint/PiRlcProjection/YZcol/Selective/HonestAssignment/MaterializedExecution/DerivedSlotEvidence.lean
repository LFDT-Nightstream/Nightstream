import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.MaterializedExecution.SourceValues

/-!
Typed evidence boundary for one exact derived centered-word decoder equation.

Owns: pointwise derived-slot decoding and the generic selected-assignment
evidence for derived and predecessor values.

Does not own: source-value transport, full rewrite recurrence composition,
retained checks, selected-row completeness, or producer authority.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `materialized.derived_slot` | the centered-word decoder equals its abstract derived field | derived |
| `materialized.predecessor` | optional predecessor lookup preserves the abstract state value | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness

structure DerivedSlotDecodeEvidence (values : SlotOwner → Nat)
    (slot : DecodedDerivedSlot) : Prop where
  decoded :
    lcEval (materializeAssignment values)
        (SourceDecode.slotExpansionTerms slot.start slot.width) =
      values (.derived slot.compilerIndex)

theorem derivedSlotDecodeEvidence
    (values : SlotOwner → Nat) (slot : DecodedDerivedSlot)
    (member : slot ∈ decodedDerivedSlots)
    (valueCanonical : values (.derived slot.compilerIndex) < goldilocksP) :
    DerivedSlotDecodeEvidence values slot :=
  ⟨Completeness.derivedSlot_decodes values slot member valueCanonical⟩

structure MaterializedDerivedValueEvidence (source derived : Nat → Nat)
    (slot : DecodedDerivedSlot) : Prop where
  valueEq :
    derivedValue (materializedAssignment source derived) slot =
      Materialized.Semantics.fieldResidue (derived slot.compilerIndex)

theorem materializedDerivedValueEvidence
    (source derived : Nat → Nat) (slot : DecodedDerivedSlot)
    (member : slot ∈ decodedDerivedSlots) :
    MaterializedDerivedValueEvidence source derived slot := by
  have canonical :
      materializedValues source derived (.derived slot.compilerIndex) <
        goldilocksP := by
    change derived slot.compilerIndex % goldilocksP < goldilocksP
    exact Nat.mod_lt _ (by decide)
  have decoded := (derivedSlotDecodeEvidence
    (materializedValues source derived) slot member canonical).decoded
  constructor
  unfold derivedValue materializedAssignment
  calc
    Materialized.Semantics.fieldResidue
          (lcEval (materializeAssignment (materializedValues source derived))
            (SourceDecode.slotExpansionTerms slot.start slot.width)) =
        Materialized.Semantics.fieldResidue
          (materializedValues source derived (.derived slot.compilerIndex)) :=
      congrArg Materialized.Semantics.fieldResidue decoded
    _ = Materialized.Semantics.fieldResidue (derived slot.compilerIndex) := by
      change Materialized.Semantics.fieldResidue
          (derived slot.compilerIndex % goldilocksP) =
        Materialized.Semantics.fieldResidue (derived slot.compilerIndex)
      apply Fin.ext
      simp only [Materialized.Semantics.fieldResidue]
      rw [← Materialized.Semantics.modulus_eq, Nat.mod_mod]

def derivedNat (derived : Nat → F) : Nat → Nat :=
  fun compilerIndex => (derived compilerIndex).val

def selectedAssignment (source : Nat → Nat) (derived : Nat → F) : Nat → Nat :=
  materializedAssignment source (derivedNat derived)

structure ConstructedValuesEvidence (source : Nat → Nat)
    (derived : Nat → F) : Prop where
  derivedEq : ∀ slot, slot ∈ decodedDerivedSlots →
    derivedValue (selectedAssignment source derived) slot =
      derived slot.compilerIndex
  previousEq : ∀ previous,
    (match previous with
      | none => True
      | some slot => slot ∈ decodedDerivedSlots) →
    previousValue (selectedAssignment source derived) previous =
      derivedPreviousValue derived previous

theorem constructedValuesEvidence (source : Nat → Nat) (derived : Nat → F) :
    ConstructedValuesEvidence source derived := by
  have derivedEq : ∀ slot, slot ∈ decodedDerivedSlots →
      derivedValue (selectedAssignment source derived) slot =
        derived slot.compilerIndex := by
    intro slot member
    change derivedValue
        (materializedAssignment source (derivedNat derived)) slot = _
    calc
      _ = Materialized.Semantics.fieldResidue
            (derivedNat derived slot.compilerIndex) :=
        (materializedDerivedValueEvidence source (derivedNat derived)
          slot member).valueEq
      _ = derived slot.compilerIndex := by
        unfold derivedNat
        exact Materialized.Semantics.fieldResidue_val
          (derived slot.compilerIndex)
  refine ⟨derivedEq, ?_⟩
  intro previous covered
  cases previous with
  | none => rfl
  | some slot =>
      exact derivedEq slot covered

structure MaterializedStepsEvidence (assignment : Nat → Nat) : Prop where
  holds : ∀ step ∈ decodedRewriteSteps, StepHolds assignment step

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
