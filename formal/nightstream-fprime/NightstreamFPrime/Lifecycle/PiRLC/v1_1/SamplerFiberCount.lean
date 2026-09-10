import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDecoderFiberCount

/-!
Owns the exact scalar-fiber consumer of the field-pair count recurrences.
The complete 54-coefficient list uses raw five-symbol indices. Totalization
is comparison-only and scalarwise; actual protocol failure is unchanged.
No inverse-sampling, runtime, or Poseidon2 distribution law is supplied.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerFiberCount

open NightstreamFPrime.Spec
open Sampling Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-- Exact-length lists do not use the scalar conversion's centered-zero default. -/
theorem ofFn_scalarOfList (values : List Coefficient) (length : values.length = coefficientCount) :
    List.ofFn (Transcript.PiRlcSampler.scalarOfList values) = values := by
  apply List.ext_get
  · simpa only [List.length_ofFn] using length.symm
  · intro index leftBound rightBound
    simp only [List.get_eq_getElem, List.getElem_ofFn, Transcript.PiRlcSampler.scalarOfList]
    exact List.getD_eq_getElem (l := values) (d := (⟨2, by decide⟩ : Coefficient)) rightBound

private theorem map_scalar_eq_some_iff (output : Option (List Coefficient))
    (lengths : ∀ values, output = some values → values.length = coefficientCount) (scalar : Scalar) :
    output.map Transcript.PiRlcSampler.scalarOfList = some scalar ↔
      output = some (List.ofFn scalar) := by
  cases output with
  | none => simp
  | some values =>
      constructor
      · intro same
        have scalarEq : Transcript.PiRlcSampler.scalarOfList values = scalar := Option.some.inj same
        have valuesEq : values = List.ofFn scalar :=
          (ofFn_scalarOfList values (lengths values rfl)).symm.trans (congrArg List.ofFn scalarEq)
        exact congrArg some valuesEq
      · intro same
        have valuesEq : values = List.ofFn scalar := Option.some.inj same
        rw [valuesEq, Option.map_some, SamplerOutputLaw.scalarOfList_ofFn]

private theorem map_eq_none_iff {Input Output : Type*} (map : Input → Output)
    (input : Option Input) : input.map map = none ↔ input = none := by
  cases input <;> simp

private theorem getD_eq_iff {Value : Type*} (input : Option Value) (fallback target : Value) :
    input.getD fallback = target ↔ input = some target ∨ (input = none ∧ fallback = target) := by
  cases input <;> simp [eq_comm]

/-- This is the actual scalar conversion, with the complete output list recovered. -/
theorem fieldDecode_eq_some_iff (fields : FieldShortfall.FieldWindow) (scalar : Scalar) :
    SamplerOutputLaw.fieldDecode fields = some scalar ↔
      FirstAccepted.boundedSample verifier coefficientCount
        (FieldDecoderFiberCount.candidateList fields) = some (List.ofFn scalar) := by
  unfold SamplerOutputLaw.fieldDecode
  rw [← FieldDecoderFiberCount.candidateList_eq_fieldCandidates]
  exact map_scalar_eq_some_iff _
    (fun _ success => FirstAccepted.bounded_success_length success) scalar

/-- Exact successful 32-field fibers for every full 54-coordinate scalar. -/
theorem field_success_fiber_card (scalar : Scalar) :
    Nat.card {fields : FieldShortfall.FieldWindow // SamplerOutputLaw.fieldDecode fields = some scalar} =
      FieldDecoderFiberCount.successCount 32 (List.ofFn scalar) := by
  have same := Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields : FieldShortfall.FieldWindow => fieldDecode_eq_some_iff fields scalar))
  calc
    _ = Nat.card (FieldDecoderFiberCount.SuccessFiber FieldShortfall.fieldLaneCount
        (List.ofFn scalar)) := by simpa only [List.length_ofFn] using same
    _ = _ := FieldDecoderFiberCount.success_fiber_card _ _

/-- Every failed window is counted, regardless of its accepted digit values. -/
theorem field_abort_fiber_card :
    Nat.card {fields : FieldShortfall.FieldWindow // SamplerOutputLaw.fieldDecode fields = none} =
      FieldDecoderFiberCount.abortCount 32 54 := by
  have same := Nat.card_congr (Equiv.subtypeEquivRight
    (fun fields : FieldShortfall.FieldWindow =>
      show SamplerOutputLaw.fieldDecode fields = none ↔
        FirstAccepted.boundedSample verifier coefficientCount
          (FieldDecoderFiberCount.candidateList fields) = none by
        unfold SamplerOutputLaw.fieldDecode
        rw [← FieldDecoderFiberCount.candidateList_eq_fieldCandidates]
        exact map_eq_none_iff _ _))
  exact same.trans (FieldDecoderFiberCount.abort_fiber_card FieldShortfall.fieldLaneCount coefficientCount)

/-- The fallback's fiber is the disjoint sum of successful fallback windows
and all aborting windows. Other scalar fibers contain only successful windows. -/
theorem totalized_fiber_card (fallback scalar : Scalar) :
    Nat.card {fields : FieldShortfall.FieldWindow //
      SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = scalar} =
      FieldDecoderFiberCount.successCount 32 (List.ofFn scalar) +
        if scalar = fallback then FieldDecoderFiberCount.abortCount 32 54 else 0 := by
  classical
  by_cases sameScalar : scalar = fallback
  · have pointwise (fields : FieldShortfall.FieldWindow) :
        SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = scalar ↔
          SamplerOutputLaw.fieldDecode fields = some scalar ∨ SamplerOutputLaw.fieldDecode fields = none := by
      simpa only [SamplerTotalizedOutputLaw.totalizedFieldDecode, sameScalar, and_true] using
        getD_eq_iff (SamplerOutputLaw.fieldDecode fields) fallback scalar
    rw [Nat.card_congr (Equiv.subtypeEquivRight pointwise),
      FieldDecoderFiberCount.disjoint_event_card
        (fun fields => SamplerOutputLaw.fieldDecode fields = some scalar)
        (fun fields => SamplerOutputLaw.fieldDecode fields = none)
        (fun _ success failure => Option.some_ne_none scalar (success.symm.trans failure)),
      field_success_fiber_card, field_abort_fiber_card, if_pos sameScalar]
  · have different : fallback ≠ scalar := Ne.symm sameScalar
    have pointwise (fields : FieldShortfall.FieldWindow) :
        SamplerTotalizedOutputLaw.totalizedFieldDecode fallback fields = scalar ↔
          SamplerOutputLaw.fieldDecode fields = some scalar := by
      simpa only [SamplerTotalizedOutputLaw.totalizedFieldDecode, different, and_false, or_false] using
        getD_eq_iff (SamplerOutputLaw.fieldDecode fields) fallback scalar
    rw [Nat.card_congr (Equiv.subtypeEquivRight pointwise), field_success_fiber_card,
      if_neg sameScalar, Nat.add_zero]

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerFiberCount
